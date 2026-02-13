import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, Union, cast

import lightning
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.accelerators.accelerator import Accelerator

from stamp.modeling.data import (
    create_dataloader,
    detect_feature_type,
    filter_complete_patient_data_,
    load_patient_level_data,
    patient_to_ground_truth_from_clini_table_,
    patient_to_survival_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.registry import ModelName, load_model_class
from stamp.types import Category, GroundTruth, PandasLabel, PatientId

__all__ = ["deploy_categorical_model_"]

__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2024-2025 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

Logit: TypeAlias = float

# Prediction type aliases
PredictionSingle: TypeAlias = torch.Tensor
PredictionMulti: TypeAlias = dict[str, torch.Tensor]
PredictionsType: TypeAlias = Mapping[
    PatientId, Union[PredictionSingle, PredictionMulti]
]


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
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None,
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
    # Detect feature type and load correct model
    feature_type = detect_feature_type(feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    models = [load_model_from_ckpt(p).eval() for p in checkpoint_paths]
    # Task consistency
    tasks = {model.hparams["task"] for model in models}

    if len(tasks) != 1:
        raise RuntimeError(f"Mixed tasks in ensemble: {tasks}")
    task = tasks.pop()

    # Feature type consistency
    model_supported = models[0].hparams["supported_features"]

    # tile-based models are strict; patient/slide models are interchangeable
    if model_supported == "tile":
        if feature_type != "tile":
            raise RuntimeError(
                f"Model trained on tile-level features cannot be deployed on {feature_type}-level features."
            )
    elif model_supported in ("slide", "patient"):
        if feature_type not in ("slide", "patient"):
            raise RuntimeError(
                f"Model trained on {model_supported}-level features cannot be deployed on tile-level features."
            )
    else:
        raise RuntimeError(f"Unknown supported_features value: {model_supported}")

    # Task-specific label consistency
    if task == "survival":
        # survival models use time_label + status_label
        time_labels = {getattr(model, "time_label", None) for model in models}
        status_labels = {getattr(model, "status_label", None) for model in models}

        if len(time_labels) != 1 or len(status_labels) != 1:
            raise RuntimeError(
                f"Survival label mismatch between models: "
                f"time_labels={time_labels}, status_labels={status_labels}"
            )

        model_time_label = next(iter(time_labels))
        model_status_label = next(iter(status_labels))

        if (time_label and time_label != model_time_label) or (
            status_label and status_label != model_status_label
        ):
            _logger.warning(
                "deployment time/status labels differ from training: "
                f"{(time_label, status_label)} vs {(model_time_label, model_status_label)}"
            )

        time_label = time_label or model_time_label
        status_label = status_label or model_status_label

    else:
        # classification/regression: still use ground_truth_label
        if (
            len(
                ground_truth_labels := {
                    tuple(model.ground_truth_label)
                    if isinstance(model.ground_truth_label, list)
                    else (model.ground_truth_label,)
                    for model in models
                }
            )
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

        ground_truth_label = ground_truth_label or cast(
            PandasLabel, model_ground_truth_label
        )

    output_dir.mkdir(exist_ok=True, parents=True)

    model_categories = None
    if task == "classification":
        # Ensure the categories were the same between all models
        category_sets = {
            tuple(cast(Sequence[GroundTruth], m.categories)) for m in models
        }
        if len(category_sets) != 1:
            raise RuntimeError(f"Categories differ between models: {category_sets}")
        model_categories = list(cast(Sequence[GroundTruth], models[0].categories))

    # Data loading logic
    if feature_type in ("tile", "slide"):
        if slide_table is None:
            raise ValueError(
                "A slide table is required for deployment of slide-level or tile-level features."
            )
        slide_to_patient = slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )
        if clini_table is not None:
            if task == "survival":
                if not hasattr(models[0], "time_label") or not isinstance(
                    models[0].time_label, str
                ):
                    raise AttributeError("Model is missing valid 'time_label' (str).")
                if not hasattr(models[0], "status_label") or not isinstance(
                    models[0].status_label, str
                ):
                    raise AttributeError("Model is missing valid 'status_label' (str).")

                patient_to_ground_truth = patient_to_survival_from_clini_table_(
                    clini_table_path=clini_table,
                    patient_label=patient_label,
                    time_label=models[0].time_label,
                    status_label=models[0].status_label,
                )
            else:
                if ground_truth_label is None:
                    raise ValueError(
                        "Ground truth label is required for deployment of classification/regression models."
                    )
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
            patient_to_ground_truth=cast(
                Mapping[PatientId, GroundTruth | None],
                patient_to_ground_truth,
            ),
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=False,
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
            task=task,
            clini_table=clini_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
            time_label=time_label,
            status_label=status_label,
        )

        patient_ids = list(patient_to_data.keys())
        patient_to_ground_truth = {
            pid: pd.ground_truth for pid, pd in patient_to_data.items()
        }
    else:
        raise RuntimeError(f"Unsupported feature type: {feature_type}")

    test_dl, _ = create_dataloader(
        feature_type=feature_type,
        task=task,
        patient_data=list(patient_to_data.values()),
        bag_size=None,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
        categories=model_categories,
    )

    df_builder = {
        "classification": _to_prediction_df,
        "regression": _to_regression_prediction_df,
        "survival": _to_survival_prediction_df,
    }[task]
    all_predictions: list[PredictionsType] = []
    categories_for_export: (
        Sequence[Category] | Mapping[str, Sequence[Category]] | None
    ) = cast(Sequence[Category] | Mapping[str, Sequence[Category]] | None, None)
    for model_i, model in enumerate(models):
        predictions = _predict(
            model=model,
            test_dl=test_dl,  # pyright: ignore[reportPossiblyUnboundVariable]
            patient_ids=patient_ids,
            accelerator=accelerator,
        )
        all_predictions.append(predictions)

        if isinstance(next(iter(predictions.values())), dict):
            # Multi-target case: gather categories across all targets for export (use model categories if available, else infer from GT)
            categories_accum: dict[str, set[GroundTruth]] = {}

            for pd_item in patient_to_data.values():
                gt = pd_item.ground_truth
                if isinstance(gt, dict):
                    for k, v in gt.items():
                        if v is not None:
                            categories_accum.setdefault(k, set()).add(v)

            categories_for_export = {k: sorted(v) for k, v in categories_accum.items()}

        else:
            # Single-target case: use categories from model if available, else infer from GT
            if task == "classification":
                categories_for_export = models[0].categories
            else:
                categories_for_export = []

        # cut-off values from survival ckpt
        cut_off = (
            getattr(model.hparams, "train_pred_median", None)
            if getattr(model.hparams, "train_pred_median", None) is not None
            else None
        )

        # Only save individual model files when deploying multiple models (ensemble)
        if len(models) > 1:
            df_builder(
                categories=categories_for_export,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
                cut_off=cut_off,
            ).to_csv(output_dir / f"patient-preds-{model_i}.csv", index=False)
        else:
            df_builder(
                categories=categories_for_export,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
                cut_off=cut_off,
            ).to_csv(output_dir / "patient-preds.csv", index=False)

    if task == "classification":
        # compute mean prediction across models (supports single- and multi-target)
        mean_preds: dict[PatientId, object] = {}
        for pid in patient_ids:
            model_preds = cast(
                list[torch.Tensor], [preds[pid] for preds in all_predictions]
            )
            firstp = model_preds[0]
            if isinstance(firstp, dict):
                # per-target averaging
                mean_preds[pid] = {
                    t: torch.stack([p[t] for p in model_preds]).mean(dim=0)
                    for t in firstp.keys()
                }
            else:
                mean_preds[pid] = torch.stack(model_preds).mean(dim=0)

        assert categories_for_export is not None, (
            "categories_for_export must be set before use"
        )
        df_builder(
            categories=categories_for_export,
            patient_to_ground_truth=patient_to_ground_truth,
            predictions=mean_preds,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        ).to_csv(output_dir / "patient-preds_95_confidence_interval.csv", index=False)


def _predict(
    *,
    model: lightning.LightningModule,
    test_dl: torch.utils.data.DataLoader,
    patient_ids: Sequence[PatientId],
    accelerator: str | Accelerator,
) -> PredictionsType:
    model = model.eval()
    torch.set_float32_matmul_precision("medium")

    # Check for data leakage
    patients_used_for_training: set[PatientId] = set(
        getattr(model, "train_patients", [])
    ) | set(getattr(model, "valid_patients", []))
    if overlap := patients_used_for_training & set(patient_ids):
        _logger.critical(
            "DATA LEAKAGE DETECTED: %d patient(s) in deployment set were used "
            "during training/validation. Overlapping IDs: %s",
            len(overlap),
            sorted(overlap),
        )

    trainer = lightning.Trainer(
        accelerator=accelerator,
        devices=1,  # Needs to be 1, otherwise half the predictions are missing for some reason
        logger=False,
    )

    outs = trainer.predict(model, test_dl)

    if not outs:
        return {}

    first = outs[0]

    # Multi-target case: each element of outs is a dict[target_label -> tensor]
    if isinstance(first, dict):
        per_target_lists: dict[str, list[torch.Tensor]] = {}
        for out in outs:
            if not isinstance(out, dict):
                raise RuntimeError("Mixed prediction output types from model")
            for k, v in out.items():
                per_target_lists.setdefault(k, []).append(v)

        per_target_tensors: dict[str, torch.Tensor] = {
            k: torch.cat(vlist, dim=0) for k, vlist in per_target_lists.items()
        }

        if getattr(model.hparams, "task", None) == "classification":
            for k in list(per_target_tensors.keys()):
                per_target_tensors[k] = torch.softmax(per_target_tensors[k], dim=1)

        # build per-patient dicts
        num_preds = next(iter(per_target_tensors.values())).shape[0]
        predictions: dict[PatientId, dict[str, torch.Tensor]] = {}
        for i, pid in enumerate(patient_ids[:num_preds]):
            predictions[pid] = {
                k: per_target_tensors[k][i] for k in per_target_tensors.keys()
            }

        return predictions

    # Single-target case: each element of outs is a tensor
    outs_single = cast(list[torch.Tensor], outs)

    raw_preds = torch.cat(outs_single, dim=0)

    if getattr(model.hparams, "task", None) == "classification":
        raw_preds = torch.softmax(raw_preds, dim=1)
    elif getattr(model.hparams, "task", None) == "survival":
        raw_preds = raw_preds.squeeze(-1)

    result: dict[PatientId, torch.Tensor] = {
        pid: raw_preds[i] for i, pid in enumerate(patient_ids)
    }

    return result


def _to_prediction_df(
    *,
    categories: Sequence[GroundTruth] | Mapping[str, Sequence[GroundTruth]],
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None]
    | Mapping[PatientId, dict[str, GroundTruth | None]],
    predictions: Mapping[PatientId, torch.Tensor]
    | Mapping[PatientId, dict[str, torch.Tensor]],
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel | Sequence[PandasLabel],
    **kwargs,
) -> pd.DataFrame:
    """Compiles deployment results into a DataFrame.

    Supports single-target and multi-target classification.
    - Single-target: `predictions` maps patient -> tensor and `categories` is a sequence.
    - Multi-target: `predictions` maps patient -> dict[target_label -> tensor] and
      `categories` is a mapping from target_label -> sequence of category names.
    """
    first_pred = next(iter(predictions.values()))

    # Multi-target predictions: dict per patient
    if isinstance(first_pred, dict):
        # determine target labels
        target_labels = list(cast(dict, first_pred).keys())

        # prepare categories mapping
        if isinstance(categories, dict):
            cats_map = categories
        else:
            # try infer categories list ordering: assume categories is a sequence-of-sequences
            cats_map = {}
            if isinstance(categories, Sequence):
                try:
                    for i, t in enumerate(target_labels):
                        cats_map[t] = list(
                            cast(Sequence[Sequence[GroundTruth]], categories)[i]
                        )
                except Exception:
                    cats_map = {}

        # infer missing category lists from ground truth
        if any(t not in cats_map for t in target_labels):
            inferred: dict[str, set] = {t: set() for t in target_labels}
            for pid, gt in patient_to_ground_truth.items():
                if isinstance(gt, dict):
                    for t in target_labels:
                        val = gt.get(t)
                        if val is not None:
                            inferred[t].add(val)
            for t in target_labels:
                if t not in cats_map:
                    cats_map[t] = sorted(inferred.get(t, []))

        rows = []
        for pid, pred_dict in predictions.items():
            row: dict = {patient_label: pid}
            gt_entry = patient_to_ground_truth.get(pid)
            # ground truths per target
            for t in target_labels:
                if isinstance(gt_entry, dict):
                    row[t] = gt_entry.get(t)
                else:
                    row[t] = gt_entry

            total_loss = 0.0
            has_loss = False
            for t in target_labels:
                tensor = cast(dict[str, torch.Tensor], pred_dict)[t]
                probs = tensor.detach().cpu()
                cats: Sequence[GroundTruth] = cast(
                    Sequence[GroundTruth],
                    cats_map.get(t, []),
                )
                if probs.numel() == 1:
                    row[f"pred_{t}"] = float(probs.item())
                else:
                    pred_idx = int(probs.argmax().item())
                    row[f"pred_{t}"] = (
                        cats[pred_idx] if pred_idx < len(cats) else pred_idx
                    )
                for i_cat, cat in enumerate(cats):
                    if i_cat < probs.shape[0]:
                        row[f"{t}_{cat}"] = float(probs[i_cat].item())
                    else:
                        row[f"{t}_{cat}"] = None

                if isinstance(gt_entry, dict) and (gt := gt_entry.get(t)) is not None:
                    try:
                        target_index = int(np.where(np.array(cats) == gt)[0][0])
                        loss = torch.nn.functional.cross_entropy(
                            probs.reshape(1, -1), torch.tensor([target_index])
                        ).item()
                        total_loss += loss
                        has_loss = True
                    except Exception:
                        pass

            row["loss"] = total_loss if has_loss else None
            rows.append(row)

        return pd.DataFrame(rows)

    # Single-target (original behaviour)
    if not all(isinstance(p, torch.Tensor) for p in predictions.values()):
        raise TypeError("Single-target block received multi-target dict predictions.")

    predictions = cast(Mapping[PatientId, torch.Tensor], predictions)

    rows = []
    for pid, prediction in predictions.items():
        gt = patient_to_ground_truth.get(pid)
        cats = cast(Sequence[GroundTruth], categories)
        pred_idx = int(prediction.argmax())
        row = {
            patient_label: pid,
            ground_truth_label: gt,
            "pred": cats[pred_idx],
            **{
                f"{ground_truth_label}_{category}": float(prediction[i_cat].item())
                for i_cat, category in enumerate(cats)
            },
            "loss": (
                torch.nn.functional.cross_entropy(
                    prediction.reshape(1, -1),
                    torch.tensor(np.where(np.array(cats) == gt)[0]),
                ).item()
                if gt is not None
                else None
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="loss")


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
        pred = pred.detach().flatten()

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
