import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import lightning
import lightning.pytorch
import lightning.pytorch.accelerators
import lightning.pytorch.accelerators.accelerator
import numpy as np
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold

from stamp.modeling.data import (
    Category,
    FeaturePath,
    GroundTruth,
    PandasLabel,
    PatientData,
    PatientId,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.train import setup_model_for_training

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
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    output_dir: Path,
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
    filename_label: PandasLabel,
    categories: Sequence[Category] | None,
    n_splits: int,
    # Dataset and -loader parameters
    bag_size: int,
    num_workers: int,
    # Training paramenters
    batch_size: int,
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
) -> None:
    patient_to_ground_truth: Final[dict[PatientId, GroundTruth]] = (
        patient_to_ground_truth_from_clini_table_(
            clini_table_path=clini_table,
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
        )
    )
    slide_to_patient: Final[dict[FeaturePath, PatientId]] = (
        slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )
    )

    # Clean data (remove slides without ground truth, missing features, etc.)
    patient_to_data: Final[Mapping[Category, PatientData]] = (
        filter_complete_patient_data_(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=True,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = output_dir / "splits.json"

    # Generate the splits, or load them from the splits file if they already exist
    if not splits_file.exists():
        splits = _get_splits(patient_to_data=patient_to_data, n_splits=n_splits)
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

    for split_i, split in enumerate(splits.splits):
        split_dir = output_dir / f"split-{split_i}"

        if (split_dir / "patient-preds.csv").exists():
            _logger.info(
                "skipping training for split {split_i}, "
                "as a model checkpoint is already present"
            )
            continue

        # Train the model
        if not (split_dir / "model.ckpt").exists():
            trainer = _train_model(
                output_dir=split_dir,
                clini_table=clini_table,
                slide_table=slide_table,
                feature_dir=feature_dir,
                ground_truth_label=ground_truth_label,
                bag_size=bag_size,
                num_workers=num_workers,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                accelerator=accelerator,
                patient_to_data={
                    patient_id: patient_data
                    for patient_id, patient_data in patient_to_data.items()
                    if patient_id in split.train_patients
                },
                categories=(
                    categories
                    or sorted(
                        {
                            patient_data.ground_truth
                            for patient_data in patient_to_data.values()
                            if patient_data.ground_truth is not None
                        }
                    )
                ),
            )
            trainer.save_checkpoint(split_dir / "model.ckpt")
            model = cast(LitVisionTransformer, trainer.model)
        else:
            model = LitVisionTransformer.load_from_checkpoint(split_dir / "model.ckpt")

        # Deploy on test set
        if not (split_dir / "patient-preds.csv").exists():
            predictions = _predict(
                model=model,
                patient_to_data={
                    patient_id: patient_data
                    for patient_id, patient_data in patient_to_data.items()
                    if patient_id in split.test_patients
                },
                num_workers=num_workers,
                accelerator=accelerator,
            )

            _to_prediction_df(
                model=model,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=ground_truth_label,
            ).to_csv(split_dir / "patient-preds.csv", index=False)


def _get_splits(
    *, patient_to_data: Mapping[PatientId, PatientData[Any]], n_splits: int
) -> _Splits:
    patients = np.array(list(patient_to_data.keys()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    splits = _Splits(
        splits=[
            _Split(
                train_patients=set(patients[train_indices]),
                test_patients=set(patients[test_indices]),
            )
            for train_indices, test_indices in skf.split(
                patients,
                np.array(
                    [patient.ground_truth for patient in patient_to_data.values()]
                ),
            )
        ]
    )
    return splits


def _train_model(
    *,
    output_dir: Path,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth]],
    categories: Sequence[Category],
    # Dataset and -loader parameters
    bag_size: int,
    num_workers: int,
    # Training paramenters
    batch_size: int,
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
    # Metadata stored in model
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    ground_truth_label: PandasLabel,
) -> lightning.Trainer:
    model, train_dl, valid_dl = setup_model_for_training(
        patient_to_data=patient_to_data,
        categories=categories,
        bag_size=bag_size,
        batch_size=batch_size,
        num_workers=num_workers,
        ground_truth_label=ground_truth_label,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
    )

    torch.set_float32_matmul_precision("high")

    model_checkpoint = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:0.3f}",
    )
    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping(monitor="validation_loss", mode="min", patience=patience),
            model_checkpoint,
        ],
        max_epochs=max_epochs,
        # FIXME The number of accelerators is currently fixed to one for the
        # following reasons:
        #  1. `trainer.predict()` does not return any predictions if used with
        #     the default strategy no multiple GPUs
        #  2. `barspoon.model.SafeMulticlassAUROC` breaks on multiple GPUs
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=output_dir),
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    LitVisionTransformer.load_from_checkpoint(model_checkpoint.best_model_path)
    return trainer
