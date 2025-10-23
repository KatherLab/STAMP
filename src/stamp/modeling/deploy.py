import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, Union, cast

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
    patient_to_survival_from_clini_table_,
    slide_to_patient_from_slide_table_,
    tile_bag_dataloader,
)
from stamp.modeling.registry import ModelName, load_model_class
from stamp.types import GroundTruth, PandasLabel, PatientId

__all__ = ["deploy_categorical_model_"]

__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2024-2025 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

Logit: TypeAlias = float


def load_model_from_ckpt(path: Union[str, Path]):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    LitModelClass, ModelClass = load_model_class(
        hparams["task"], hparams["supported_features"], ModelName(hparams["model_name"])
    )

    return LitModelClass.load_from_checkpoint(path, model_class=ModelClass)


def deploy_categorical_model_(
    *,
    output_dir: Path,
    checkpoint_paths: Sequence[Path],
    clini_table: Path | None,
    slide_table: Path | None,
    feature_dir: Path,
    ground_truth_label: PandasLabel | None,
    time_label: PandasLabel | None,
    status_label: PandasLabel | None,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    num_workers: int,
    accelerator: str | Accelerator,
) -> None:
    """Deploy categorical model(s) and save predictions.

    For single model deployment, creates:
    - patient-preds.csv (main prediction file)

    For ensemble deployment (multiple checkpoints), creates:
    - patient-preds-{i}.csv (individual model predictions)
    - patient-preds.csv (mean predictions across models)
    """
    # --- Detect feature type and load correct model ---
    feature_type = detect_feature_type(feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    models = [load_model_from_ckpt(p).eval() for p in checkpoint_paths]
    # task consistency
    tasks = {model.hparams["task"] for model in models}

    if len(tasks) != 1:
        raise RuntimeError(f"Mixed tasks in ensemble: {tasks}")
    task = tasks.pop()

    if models[0].hparams["supported_features"] != feature_type:
        print(getattr(models[0], "supported_features"), feature_type)
        raise RuntimeError(
            f"Unsupported feature type for deployment: {feature_type}. Only 'tile' and 'patient' are supported."
        )

    # Ensure all models were trained on the same ground truth label
    if (
        len(ground_truth_labels := set(model.ground_truth_label for model in models))
        != 1
    ):
        raise RuntimeError(
            f"ground truth labels differ between models: {ground_truth_labels}"
        )

    model_ground_truth_label = models[0].ground_truth_label

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

    model_categories = None
    if task == "classification":
        # Ensure the categories were the same between all models
        category_sets = {tuple(m.categories) for m in models}
        if len(category_sets) != 1:
            raise RuntimeError(f"Categories differ between models: {category_sets}")
        model_categories = list(models[0].categories)

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
            if task == "survival":
                patient_to_ground_truth = patient_to_survival_from_clini_table_(
                    clini_table_path=clini_table,
                    patient_label=patient_label,
                    time_label=models[0].time_label,
                    status_label=models[0].status_label,
                )
            else:
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
            task=task,
            bag_size=None,  # We want all tiles to be seen by the model
            categories=model_categories,
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

    df_builder = {
        "classification": _to_prediction_df,
        "regression": _to_regression_prediction_df,
        "survival": _to_survival_prediction_df,
    }[task]
    all_predictions: list[Mapping[PatientId, Float[torch.Tensor, "category"]]] = []  # noqa: F821
    for model_i, model in enumerate(models):
        predictions = _predict(
            model=model,
            test_dl=test_dl,
            patient_ids=patient_ids,
            accelerator=accelerator,
        )
        all_predictions.append(predictions)

        # cut-off values from survival ckpt
        cut_off = (
            model.hparams["train_pred_mean"]
            if model.hparams["train_pred_mean"] is not None
            else None
        )

        # Only save individual model files when deploying multiple models (ensemble)
        if len(models) > 1:
            df_builder(
                categories=model_categories,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
                cut_off=cut_off,
            ).to_csv(output_dir / f"patient-preds-{model_i}.csv", index=False)
        else:
            df_builder(
                categories=model_categories,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
                cut_off=cut_off,
            ).to_csv(output_dir / "patient-preds.csv", index=False)

    if task == "classification":
        # TODO we probably also want to save the 95% confidence interval in addition to the mean
        df_builder(
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
        ).to_csv(output_dir / "patient-preds_95_confidence_interval.csv", index=False)


def _predict(
    *,
    model: lightning.LightningModule,
    test_dl: torch.utils.data.DataLoader,
    patient_ids: Sequence[PatientId],
    accelerator: str | Accelerator,
) -> Mapping[PatientId, Float[torch.Tensor, "..."]]:
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

    raw_preds = torch.concat(cast(list[torch.Tensor], trainer.predict(model, test_dl)))

    if getattr(model.hparams, "task", None) == "classification":
        predictions = torch.softmax(raw_preds, dim=1)
    elif getattr(model.hparams, "task", None) == "survival":
        predictions = raw_preds.squeeze(-1)  # (N,) risk scores
    else:  # regression
        predictions = raw_preds

    return dict(zip(patient_ids, predictions, strict=True))


def _to_prediction_df(
    *,
    categories: Sequence[GroundTruth],
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
    predictions: Mapping[PatientId, torch.Tensor],
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
    **kwargs,
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


def _to_regression_prediction_df(
    *,
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
    predictions: Mapping[PatientId, torch.Tensor],
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
    **kwargs,
) -> pd.DataFrame:
    """Compiles deployment results into a DataFrame for regression.

    Columns:
      - patient_label
      - ground_truth_label (numeric if available)
      - pred (float)
      - loss (per-sample L1 loss if GT available, else None)
    """
    import torch.nn.functional as F

    return pd.DataFrame(
        [
            {
                patient_label: patient_id,
                ground_truth_label: patient_to_ground_truth.get(patient_id),
                "pred": float(prediction.flatten().item())
                if prediction.numel() == 1
                else prediction.cpu().tolist(),
                "loss": (
                    F.l1_loss(
                        prediction.flatten(),
                        torch.tensor(
                            [float(ground_truth)],
                            dtype=prediction.dtype,
                            device=prediction.device,
                        ),
                        reduction="mean",
                    ).item()
                    if (
                        (ground_truth := patient_to_ground_truth.get(patient_id))
                        is not None
                        and str(ground_truth).lower() != "nan"
                        and prediction.numel() == 1
                    )
                    else None
                ),
            }
            for patient_id, prediction in predictions.items()
        ]
    ).sort_values(by="loss", na_position="last")


def _to_survival_prediction_df(
    *,
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
    predictions: Mapping[PatientId, torch.Tensor],
    patient_label: PandasLabel,
    cut_off: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compiles deployment results into a DataFrame for survival analysis.

    Ground truth values should be either:
      - a string "time status" (e.g. "302 dead"), or
      - a tuple/list (time, event).

    Predictions are assumed to be risk scores (Cox model), shape [1].
    """
    rows: list[dict] = []

    for patient_id, pred in predictions.items():
        pred = -pred.detach().flatten()

        gt = patient_to_ground_truth.get(patient_id)

        row: dict = {patient_label: patient_id}

        # Prediction: risk score
        if pred.numel() == 1:
            row["pred_score"] = float(pred.item())
        else:
            row["pred_score"] = pred.cpu().tolist()

        # Ground truth: time + event
        if gt is not None:
            if isinstance(gt, str) and " " in gt:
                time_str, status_str = gt.split(" ", 1)
                row["time"] = float(time_str) if time_str.lower() != "nan" else None
                if status_str.lower() in {"dead", "event", "1"}:
                    row["event"] = 1
                elif status_str.lower() in {"alive", "censored", "0"}:
                    row["event"] = 0
                else:
                    row["event"] = None
            elif isinstance(gt, (tuple, list)) and len(gt) == 2:
                row["time"], row["event"] = gt
            else:
                row["time"], row["event"] = None, None
        else:
            row["time"], row["event"] = None, None

        rows.append(row)

    df = pd.DataFrame(rows)
    if cut_off is not None:
        df[f"cut_off={cut_off}"] = None

    return df
