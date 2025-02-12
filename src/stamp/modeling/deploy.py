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
    GroundTruth,
    PandasLabel,
    PatientData,
    PatientId,
    dataloader_from_patient_data,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.lightning_cobra import LitCobra

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
    slide_table: Path,
    feature_dir: Path,
    ground_truth_label: PandasLabel | None,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    num_workers: int,
    accelerator: str | Accelerator,
    use_cobra: bool,
) -> None:
    if use_cobra:
        models = [
            LitCobra.load_from_checkpoint(checkpoint_path=checkpoint_path).eval() 
                  for checkpoint_path in checkpoint_paths
        ]
    else:
        model = [
            LitVisionTransformer.load_from_checkpoint(
                checkpoint_path=checkpoint_path
            ).eval()
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

    slide_to_patient = slide_to_patient_from_slide_table_(
        slide_table_path=slide_table,
        feature_dir=feature_dir,
        patient_label=patient_label,
        filename_label=filename_label,
    )

    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None]
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

    all_predictions: list[Mapping[PatientId, Float[torch.Tensor, "category"]]] = []  # noqa: F821
    for model_i, model in enumerate(models):
        predictions = _predict(
            model=model,
            patient_to_data=patient_to_data,
            num_workers=num_workers,
            accelerator=accelerator,
        )
        all_predictions.append(predictions)

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
            for patient_id in patient_to_data.keys()
        },
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
    ).to_csv(output_dir / "patient-preds.csv", index=False)


def _predict(
    *,
    model: LitVisionTransformer | LitCobra,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth | None]],
    num_workers: int,
    accelerator: str | Accelerator,
) -> Mapping[PatientId, Float[torch.Tensor, "category"]]:  # noqa: F821
    model = model.eval()
    torch.set_float32_matmul_precision("medium")

    patients_used_for_training: set[PatientId] = set(model.train_patients) | set(
        model.valid_patients
    )
    if overlap := patients_used_for_training & set(patient_to_data.keys()):
        raise ValueError(
            f"some of the patients in the validation set were used during training: {overlap}"
        )

    test_dl, _ = dataloader_from_patient_data(
        patient_data=list(patient_to_data.values()),
        bag_size=None,  # Use all the tiles for deployment
        # Use same encoding scheme as during training
        categories=list(model.categories),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
    )

    if isinstance(model, LitCobra) and torch.cuda.get_device_capability()[0] < 8:
        _logger.warning(
            f"\033[93mCOBRA (Mamba2) is designed to run on GPUs with compute capability 8.0 or higher!! "
            f"Your GPU has compute capability {torch.cuda.get_device_capability()[0]}. "
            f"We are forced to switch to mixed FP16 precision. This may lead to numerical instability and reduced performance!!\033[0m"
        )
        precision = "16-mixed"
    else:
        precision = "32-true"
        torch.set_float32_matmul_precision("high")
    
    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices=1,  # Needs to be 1, otherwise half the predictions are missing for some reason
        logger=False,
        precision=precision,
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

    return dict(zip(patient_to_data, predictions, strict=True))


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
