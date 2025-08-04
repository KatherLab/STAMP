import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

import lightning
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from lightning.pytorch.accelerators.accelerator import Accelerator

from stamp.modeling.data import (
    detect_feature_type,
    filter_complete_patient_data_,
    load_patient_level_data,
    patient_feature_dataloader,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
    tile_bag_dataloader,
)
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.mlp_classifier import LitMLPClassifier
from stamp.types import GroundTruth, PandasLabel, PatientId

__all__ = ["deploy_categorical_model_"]

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024-2025 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

Logit: TypeAlias = float


def deploy_categorical_model_(
    *,
    output_dir: Path,
    checkpoint_paths: Sequence[Path],
    clini_table: Path | None,
    slide_table: Path | None,
    feature_dir: Path,
    ground_truth_label: PandasLabel | None,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    num_workers: int,
    accelerator: str | Accelerator,
) -> None:
    # --- Detect feature type and load correct model ---
    feature_type = detect_feature_type(feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type == "tile":
        ModelClass = LitVisionTransformer
    elif feature_type == "patient":
        ModelClass = LitMLPClassifier
    else:
        raise RuntimeError(
            f"Unsupported feature type for deployment: {feature_type}. Only 'tile' and 'patient' are supported."
        )

    models = [
        ModelClass.load_from_checkpoint(checkpoint_path=checkpoint_path).eval()
        for checkpoint_path in checkpoint_paths
    ]

    # Ensure all models were trained on the same ground truth label
    if (
        len(ground_truth_labels := set(model.ground_truth_label for model in models))
        != 1
    ):
        raise RuntimeError(
            f"ground truth labels differ between models: {ground_truth_labels}"
        )
    # Ensure the categories were the same between all models
    if len(categories := set(tuple(model.categories) for model in models)) != 1:
        raise RuntimeError(f"categories differ between models: {categories}")

    model_ground_truth_label = models[0].ground_truth_label
    model_categories = list(models[0].categories)

    if (
        ground_truth_label is not None
        and ground_truth_label != model_ground_truth_label
    ):
        _logger.warning(
            "deployment ground truth label differs from training: "
            f"{ground_truth_label} vs {model_ground_truth_label}"
        )
    ground_truth_label = ground_truth_label or model_ground_truth_label

    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Data loading logic ---
    if feature_type == "tile":
        if slide_table is None:
            raise ValueError("A slide table is required for tile-level modeling")
        slide_to_patient = slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )
        if clini_table is not None:
            patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
                clini_table_path=clini_table,
                ground_truth_label=ground_truth_label,
                patient_label=patient_label,
            )
        else:
            patient_to_ground_truth = {
                patient_id: None for patient_id in set(slide_to_patient.values())
            }
        patient_to_data = filter_complete_patient_data_(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=False,
        )
        test_dl, _ = tile_bag_dataloader(
            patient_data=list(patient_to_data.values()),
            bag_size=None,  # We want all tiles to be seen by the model
            categories=list(models[0].categories),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            transform=None,
        )
        patient_ids = list(patient_to_data.keys())
    elif feature_type == "patient":
        if slide_table is not None:
            _logger.warning(
                "slide_table is ignored for patient-level features during deployment."
            )
        if clini_table is None:
            raise ValueError(
                "clini_table is required for patient-level feature deployment."
            )
        patient_to_data = load_patient_level_data(
            clini_table=clini_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )
        test_dl, _ = patient_feature_dataloader(
            patient_data=list(patient_to_data.values()),
            categories=list(models[0].categories),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            transform=None,
        )
        patient_ids = list(patient_to_data.keys())
        patient_to_ground_truth = {
            pid: pd.ground_truth for pid, pd in patient_to_data.items()
        }
    else:
        raise RuntimeError(f"Unsupported feature type: {feature_type}")

    all_predictions: list[Mapping[PatientId, Float[torch.Tensor, "category"]]] = []  # noqa: F821
    for model_i, model in enumerate(models):
        predictions = _predict(
            model=model,
            test_dl=test_dl,
            patient_ids=patient_ids,
            accelerator=accelerator,
        )
        all_predictions.append(predictions)

        # Only save individual model files when deploying multiple models (ensemble)
        if len(models) > 1:
            _to_prediction_df(
                categories=model_categories,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
            ).to_csv(output_dir / f"patient-preds-{model_i}.csv", index=False)

    # TODO we probably also want to save the 95% confidence interval in addition to the mean
    _to_prediction_df(
        categories=model_categories,
        patient_to_ground_truth=patient_to_ground_truth,
        predictions={
            # Mean prediction
            patient_id: torch.stack(
                [predictions[patient_id] for predictions in all_predictions]
            ).mean(dim=0)
            for patient_id in patient_ids
        },
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
    ).to_csv(output_dir / "patient-preds.csv", index=False)


def _predict(
    *,
    model: lightning.LightningModule,
    test_dl: torch.utils.data.DataLoader,
    patient_ids: Sequence[PatientId],
    accelerator: str | Accelerator,
) -> Mapping[PatientId, Float[torch.Tensor, "category"]]:  # noqa: F821
    model = model.eval()
    torch.set_float32_matmul_precision("medium")

    # Check for data leakage
    patients_used_for_training: set[PatientId] = set(
        getattr(model, "train_patients", [])
    ) | set(getattr(model, "valid_patients", []))
    if overlap := patients_used_for_training & set(patient_ids):
        raise ValueError(
            f"some of the patients in the validation set were used during training: {overlap}"
        )

    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices=1,  # Needs to be 1, otherwise half the predictions are missing for some reason
        logger=False,
    )
    predictions = torch.softmax(
        torch.concat(
            cast(
                list[torch.Tensor],
                trainer.predict(model, test_dl),
            )
        ),
        dim=1,
    )

    return dict(zip(patient_ids, predictions, strict=True))


def _to_prediction_df(
    *,
    categories: Sequence[GroundTruth],
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
    predictions: Mapping[PatientId, torch.Tensor],
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
) -> pd.DataFrame:
    """Compiles deployment results into a DataFrame."""
    return pd.DataFrame(
        [
            {
                patient_label: patient_id,
                ground_truth_label: patient_to_ground_truth.get(patient_id),
                "pred": categories[int(prediction.argmax())],
                **{
                    f"{ground_truth_label}_{category}": prediction[i_cat].item()
                    for i_cat, category in enumerate(categories)
                },
                "loss": (
                    torch.nn.functional.cross_entropy(
                        prediction.reshape(1, -1),
                        torch.tensor(np.where(np.array(categories) == ground_truth)[0]),
                    ).item()
                    if (ground_truth := patient_to_ground_truth.get(patient_id))
                    is not None
                    else None
                ),
            }
            for patient_id, prediction in predictions.items()
        ]
    ).sort_values(by="loss")
