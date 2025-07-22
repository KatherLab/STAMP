import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold

from stamp.modeling.config import AdvancedConfig, CrossvalConfig
from stamp.modeling.data import (
    PatientData,
    detect_feature_type,
    filter_complete_patient_data_,
    load_patient_level_data,
    patient_feature_dataloader,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
    tile_bag_dataloader,
)
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.mlp_classifier import LitMLPClassifier
from stamp.modeling.train import setup_model_for_training, train_model_
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.types import (
    FeaturePath,
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
    feature_type = detect_feature_type(config.feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type == "tile":
        if config.slide_table is None:
            raise ValueError("A slide table is required for tile-level modeling")
        patient_to_ground_truth: dict[PatientId, GroundTruth] = (
            patient_to_ground_truth_from_clini_table_(
                clini_table_path=config.clini_table,
                ground_truth_label=config.ground_truth_label,
                patient_label=config.patient_label,
            )
        )
        slide_to_patient: Final[dict[FeaturePath, PatientId]] = (
            slide_to_patient_from_slide_table_(
                slide_table_path=config.slide_table,
                feature_dir=config.feature_dir,
                patient_label=config.patient_label,
                filename_label=config.filename_label,
            )
        )
        patient_to_data: Mapping[PatientId, PatientData] = (
            filter_complete_patient_data_(
                patient_to_ground_truth=patient_to_ground_truth,
                slide_to_patient=slide_to_patient,
                drop_patients_with_missing_ground_truth=True,
            )
        )
    elif feature_type == "patient":
        patient_to_data: Mapping[PatientId, PatientData] = load_patient_level_data(
            clini_table=config.clini_table,
            feature_dir=config.feature_dir,
            patient_label=config.patient_label,
            ground_truth_label=config.ground_truth_label,
        )
        patient_to_ground_truth: dict[PatientId, GroundTruth] = {
            pid: pd.ground_truth for pid, pd in patient_to_data.items()
        }
    else:
        raise RuntimeError(f"Unsupported feature type: {feature_type}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = config.output_dir / "splits.json"

    # Generate the splits, or load them from the splits file if they already exist
    if not splits_file.exists():
        splits = _get_splits(patient_to_data=patient_to_data, n_splits=config.n_splits)
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

    categories = config.categories or sorted(
        {
            patient_data.ground_truth
            for patient_data in patient_to_data.values()
            if patient_data.ground_truth is not None
        }
    )

    for split_i, split in enumerate(splits.splits):
        split_dir = config.output_dir / f"split-{split_i}"

        if (split_dir / "patient-preds.csv").exists():
            _logger.info(
                f"skipping training for split {split_i}, "
                "as a model checkpoint is already present"
            )
            continue

        # Train the model
        if not (split_dir / "model.ckpt").exists():
            model, train_dl, valid_dl = setup_model_for_training(
                clini_table=config.clini_table,
                slide_table=config.slide_table,
                feature_dir=config.feature_dir,
                ground_truth_label=config.ground_truth_label,
                advanced=advanced,
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
                train_transform=(
                    VaryPrecisionTransform(min_fraction_bits=1)
                    if config.use_vary_precision_transform
                    else None
                ),
                feature_type=feature_type,
            )
            model = train_model_(
                output_dir=split_dir,
                model=model,
                train_dl=train_dl,
                valid_dl=valid_dl,
                max_epochs=advanced.max_epochs,
                patience=advanced.patience,
                accelerator=advanced.accelerator,
            )
        else:
            if feature_type == "tile":
                model = LitVisionTransformer.load_from_checkpoint(
                    split_dir / "model.ckpt"
                )
            else:
                model = LitMLPClassifier.load_from_checkpoint(split_dir / "model.ckpt")

        # Deploy on test set
        if not (split_dir / "patient-preds.csv").exists():
            # Prepare test dataloader
            test_patients = [
                pid for pid in split.test_patients if pid in patient_to_data
            ]
            test_patient_data = [patient_to_data[pid] for pid in test_patients]
            if feature_type == "tile":
                test_dl, _ = tile_bag_dataloader(
                    patient_data=test_patient_data,
                    bag_size=None,
                    categories=categories,
                    batch_size=1,
                    shuffle=False,
                    num_workers=advanced.num_workers,
                    transform=None,
                )
            elif feature_type == "patient":
                test_dl, _ = patient_feature_dataloader(
                    patient_data=test_patient_data,
                    categories=categories,
                    batch_size=1,
                    shuffle=False,
                    num_workers=advanced.num_workers,
                    transform=None,
                )
            else:
                raise RuntimeError(f"Unsupported feature type: {feature_type}")

            predictions = _predict(
                model=model,
                test_dl=test_dl,
                patient_ids=test_patients,
                accelerator=advanced.accelerator,
            )

            _to_prediction_df(
                categories=categories,
                patient_to_ground_truth=patient_to_ground_truth,
                predictions=predictions,
                patient_label=config.patient_label,
                ground_truth_label=config.ground_truth_label,
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
