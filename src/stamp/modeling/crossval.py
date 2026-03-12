import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.model_selection import KFold, StratifiedKFold

from stamp.modeling.config import AdvancedConfig, CrossvalConfig
from stamp.modeling.data import (
    PatientData,
    create_dataloader,
    load_patient_data_,
    log_patient_class_summary,
)
from stamp.modeling.deploy import (
    _predict,
    _to_prediction_df,
    _to_regression_prediction_df,
    _to_survival_prediction_df,
    load_model_from_ckpt,
)
from stamp.modeling.train import setup_model_from_dataloaders, train_model_
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.types import (
    GroundTruth,
    PatientId,
)

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


class _Split(BaseModel):
    train_patients: set[PatientId]
    test_patients: set[PatientId]


class _Splits(BaseModel):
    splits: Sequence[_Split]


def categorical_crossval_(
    config: CrossvalConfig,
    advanced: AdvancedConfig,
) -> None:
    if config.task is None:
        raise ValueError(
            "task must be set to 'classification' | 'regression' | 'survival'"
        )

    patient_to_data, feature_type = load_patient_data_(
        feature_dir=config.feature_dir,
        clini_table=config.clini_table,
        slide_table=config.slide_table,
        task=config.task,
        ground_truth_label=config.ground_truth_label,
        time_label=config.time_label,
        status_label=config.status_label,
        patient_label=config.patient_label,
        filename_label=config.filename_label,
        drop_patients_with_missing_ground_truth=True,
    )
    _logger.info(f"Detected feature type: {feature_type}")

    patient_to_ground_truth = {
        pid: pd.ground_truth for pid, pd in patient_to_data.items()
    }

    if feature_type not in ("tile", "slide", "patient"):
        raise ValueError(f"Unknown feature type: {feature_type}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = config.output_dir / "splits.json"

    # Generate the splits, or load them from the splits file if they already exist
    if not splits_file.exists():
        # Detect multi-target classification (ground_truth is a dict)
        is_multitarget = any(
            isinstance(pd.ground_truth, dict) for pd in patient_to_data.values()
        )

        # Use KFold for regression or multi-target classification; otherwise StratifiedKFold.
        # For survival we want StratifiedKFold so folds are balanced by event status.
        spliter = (
            KFold
            if (config.task == "regression" or is_multitarget)
            else StratifiedKFold
        )

        _logger.info(f"Using {spliter.__name__} for cross-validation splits")

        splits = _get_splits(
            patient_to_data=patient_to_data,
            n_splits=config.n_splits,
            spliter=spliter,
            task=config.task,
        )

        with open(splits_file, "w") as fp:
            fp.write(splits.model_dump_json(indent=4))
    else:
        _logger.debug(f"reading splits from {splits_file}")
        with open(splits_file, "r") as fp:
            splits = _Splits.model_validate_json(fp.read())

    patients_in_splits = {
        patient
        for split in splits.splits
        for patient in [*split.train_patients, *split.test_patients]
    }

    if patients_without_ground_truth := patients_in_splits - patient_to_data.keys():
        raise RuntimeError(
            "The splits file contains some patients we don't have information for in the clini / slide table: "
            f"{patients_without_ground_truth}"
        )

    if ground_truths_not_in_split := patient_to_data.keys() - patients_in_splits:
        _logger.warning(
            "Some of the entries in the clini / slide table are not in the crossval split: "
            f"{ground_truths_not_in_split}"
        )

    categories_for_export: (
        dict[str, list] | list
    ) = []  # declare upfront to avoid unbound variable warnings
    categories: Sequence[GroundTruth] | list | None = []

    if config.task == "classification":
        # Determine categories for training (single-target) and for export (supports multi-target)
        if isinstance(config.ground_truth_label, str):
            categories = config.categories or sorted(
                {
                    patient_data.ground_truth
                    for patient_data in patient_to_data.values()
                    if patient_data.ground_truth is not None
                }
            )
            log_patient_class_summary(
                patient_to_data={pid: patient_to_data[pid] for pid in patient_to_data},
                categories=categories,
            )
            categories_for_export = cast(list, categories)
        else:
            # Multi-target: build a mapping from target label -> sorted list of categories
            categories_accum: dict[str, set[GroundTruth]] = {}
            for patient_data in patient_to_data.values():
                gt = patient_data.ground_truth
                if isinstance(gt, dict):
                    for k, v in gt.items():
                        if v is not None:
                            categories_accum.setdefault(k, set()).add(v)
            categories_for_export = {k: sorted(v) for k, v in categories_accum.items()}
            # Log summary per target
            for t, cats in categories_for_export.items():
                ground_truths = [
                    pd.ground_truth.get(t)
                    for pd in patient_to_data.values()
                    if isinstance(pd.ground_truth, dict)
                    and pd.ground_truth.get(t) is not None
                ]
                counter = Counter(ground_truths)
                _logger.info(
                    f"{t} | Total patients: {len(ground_truths)} | "
                    + " | ".join([f"Class {c}: {counter.get(c, 0)}" for c in cats])
                )
            # For training, categories can remain None (inferred later)
            categories = config.categories or None
    else:
        categories = []

    for split_i, split in enumerate(splits.splits):
        split_dir = config.output_dir / f"split-{split_i}"

        if (split_dir / "patient-preds.csv").exists():
            _logger.info(
                f"skipping training for split {split_i}, "
                "as a model checkpoint is already present"
            )
            continue

        if config.task is None:
            raise ValueError(
                "config.task must be set to 'classification' | 'regression' | 'survival'"
            )

        # Train the model
        if not (split_dir / "model.ckpt").exists():
            # Build train and test dataloaders directly (pure 2-way k-fold split)
            train_patient_ids = [
                pid for pid in split.train_patients if pid in patient_to_data
            ]
            test_patient_ids = [
                pid for pid in split.test_patients if pid in patient_to_data
            ]
            train_patient_data = [patient_to_data[pid] for pid in train_patient_ids]
            test_patient_data = [patient_to_data[pid] for pid in test_patient_ids]

            fold_categories = (
                categories
                if categories is not None
                else (
                    sorted(
                        {
                            patient_data.ground_truth
                            for patient_data in patient_to_data.values()
                            if patient_data.ground_truth is not None
                            and not isinstance(patient_data.ground_truth, dict)
                        }
                    )
                    if not isinstance(config.ground_truth_label, Sequence)
                    else None
                )
            )

            train_transform = (
                VaryPrecisionTransform(min_fraction_bits=1)
                if config.use_vary_precision_transform
                else None
            )

            train_dl, train_categories = create_dataloader(
                feature_type=feature_type,
                task=config.task,
                patient_data=train_patient_data,
                bag_size=advanced.bag_size,
                batch_size=advanced.batch_size,
                shuffle=True,
                num_workers=advanced.num_workers,
                transform=train_transform,
                categories=fold_categories,
            )
            test_dl, _ = create_dataloader(
                feature_type=feature_type,
                task=config.task,
                patient_data=test_patient_data,
                bag_size=None,
                batch_size=1,
                shuffle=False,
                num_workers=advanced.num_workers,
                transform=None,
                categories=train_categories,
            )

            # Infer feature dimension
            batch = next(iter(train_dl))
            if feature_type == "tile":
                bags, _, _, _ = batch
                dim_feats = bags.shape[-1]
            else:
                feats, _ = batch
                dim_feats = feats.shape[-1]

            model = setup_model_from_dataloaders(
                train_dl=train_dl,
                valid_dl=test_dl,
                task=config.task,
                train_categories=train_categories,
                dim_feats=dim_feats,
                train_patients=train_patient_ids,
                valid_patients=test_patient_ids,
                feature_type=feature_type,
                advanced=advanced,
                ground_truth_label=config.ground_truth_label,
                time_label=config.time_label,
                status_label=config.status_label,
                clini_table=config.clini_table,
                slide_table=config.slide_table,
                feature_dir=config.feature_dir,
            )
            model = train_model_(
                output_dir=split_dir,
                model=model,
                train_dl=train_dl,
                valid_dl=test_dl,
                max_epochs=advanced.max_epochs,
                patience=advanced.patience,
                accelerator=advanced.accelerator,
            )
        else:
            if feature_type == "tile":
                model = load_model_from_ckpt(split_dir / "model.ckpt")
            else:
                model = load_model_from_ckpt(split_dir / "model.ckpt")

        # Deploy on test fold (used as validation/prediction set)
        if not (split_dir / "patient-preds.csv").exists():
            # Prepare validation dataloader for predictions
            test_patients = [
                pid for pid in split.test_patients if pid in patient_to_data
            ]
            test_patient_data = [patient_to_data[pid] for pid in test_patients]
            test_dl, _ = create_dataloader(
                feature_type=feature_type,
                task=config.task,
                patient_data=test_patient_data,
                bag_size=None,
                batch_size=1,
                shuffle=False,
                num_workers=advanced.num_workers,
                transform=None,
                categories=categories,
            )

            predictions = _predict(
                model=model,
                test_dl=test_dl,
                patient_ids=test_patients,
                accelerator=advanced.accelerator,
            )

            if config.task == "survival":
                # Export only when patients have single-target survival labels ("time status").
                # Don't rely on `ground_truth_label` for survival — check the loaded patient data.
                if any(isinstance(gt, dict) for gt in patient_to_ground_truth.values()):
                    _logger.warning(
                        "Multi-target survival prediction export not yet supported; skipping CSV save"
                    )
                else:
                    _to_survival_prediction_df(
                        patient_to_ground_truth=cast(
                            Mapping[
                                PatientId, str | tuple[float | None, int | None] | None
                            ],
                            patient_to_ground_truth,
                        ),
                        predictions=cast(Mapping[PatientId, torch.Tensor], predictions),
                        patient_label=config.patient_label,
                        cut_off=getattr(model.hparams, "train_pred_median", None),
                    ).to_csv(split_dir / "patient-preds.csv", index=False)
            elif config.task == "regression":
                if config.ground_truth_label is None:
                    raise RuntimeError("Grounf truth label is required for regression")
                if isinstance(config.ground_truth_label, str):
                    _to_regression_prediction_df(
                        patient_to_ground_truth=cast(
                            Mapping[PatientId, str | None], patient_to_ground_truth
                        ),
                        predictions=cast(Mapping[PatientId, torch.Tensor], predictions),
                        patient_label=config.patient_label,
                        ground_truth_label=config.ground_truth_label,
                    ).to_csv(split_dir / "patient-preds.csv", index=False)
                else:
                    _logger.warning(
                        "Multi-target regression prediction export not yet supported; skipping CSV save"
                    )
            else:
                if config.ground_truth_label is None:
                    raise RuntimeError(
                        "Grounf truth label is required for classification"
                    )
                _to_prediction_df(
                    categories=categories_for_export,
                    patient_to_ground_truth=patient_to_ground_truth,
                    predictions=cast(
                        Mapping[PatientId, torch.Tensor]
                        | Mapping[PatientId, dict[str, torch.Tensor]],
                        predictions,
                    ),
                    patient_label=config.patient_label,
                    ground_truth_label=config.ground_truth_label,
                ).to_csv(split_dir / "patient-preds.csv", index=False)


def _get_splits(
    *,
    patient_to_data: Mapping[PatientId, PatientData[Any]],
    n_splits: int,
    spliter,
    task: str | None = None,
) -> _Splits:
    patients = np.array(list(patient_to_data.keys()))

    # Build stratification labels depending on the task
    gts = [patient.ground_truth for patient in patient_to_data.values()]

    if task == "survival":
        # use event status (0/1) for stratification
        # Ground-truths are expected to be pre-parsed into (time, status) tuples
        # by the data loading pipeline; extract the status element directly.
        statuses: list[int] = []
        for gt in gts:
            # support multi-target fallback: use first target
            val = next(iter(gt.values())) if isinstance(gt, dict) else gt
            # If structured (time, status) use second element, otherwise assume val is status
            if isinstance(val, (tuple, list)) and len(val) == 2:
                status_val = val[1]
            else:
                status_val = val
            statuses.append(int(cast(int, status_val)) if status_val is not None else 0)

        y_strat = np.array(statuses)
    elif task == "classification":
        # For multi-target (dict), use the first target's value
        y_strat = np.array(
            [next(iter(gt.values())) if isinstance(gt, dict) else gt for gt in gts]
        )
    else:
        # regression or unknown: do not stratify (KFold will ignore y)
        y_strat = None

    skf = spliter(n_splits=n_splits, shuffle=True, random_state=0)

    if y_strat is None:
        splits_iter = skf.split(patients)
    else:
        splits_iter = skf.split(patients, y_strat)

    splits = _Splits(
        splits=[
            _Split(
                train_patients=set(patients[train_indices]),
                test_patients=set(patients[test_indices]),
            )
            for train_indices, test_indices in splits_iter
        ]
    )
    return splits
