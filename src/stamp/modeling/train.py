import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import lightning
import lightning.pytorch
import lightning.pytorch.accelerators
import lightning.pytorch.accelerators.accelerator
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from stamp.modeling.data import (
    BagDataset,
    PatientData,
    PatientFeatureDataset,
    detect_feature_type,
    filter_complete_patient_data_,
    load_patient_level_data,
    patient_feature_dataloader,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
    tile_bag_dataloader,
)
from stamp.modeling.lightning_model import (
    Bags,
    BagSizes,
    EncodedTargets,
    LitVisionTransformer,
)
from stamp.modeling.mlp_classifier import LitMLPClassifier
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.types import Category, CoordinatesBatch, GroundTruth, PandasLabel, PatientId

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


def train_categorical_model_(
    *,
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    output_dir: Path,
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
    filename_label: PandasLabel,
    categories: Sequence[Category] | None,
    # Dataset and -loader parameters
    bag_size: int,
    num_workers: int,
    # Training paramenters
    batch_size: int,
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
    # Experimental features
    use_vary_precision_transform: bool,
    use_alibi: bool,
) -> None:
    """Trains a model based on the feature type.

    Args:
        clini_table:
            An excel or csv file to read the clinical information from.
            Must at least have the columns specified in the arguments

            `patient_label` (containing a unique patient ID)
            and `ground_truth_label` (containing the ground truth to train for).
        slide_table:
            An excel or csv file to read the patient-slide associations from.
            Must at least have the columns specified in the arguments
            `patient_label` (containing the patient ID)
            and `filename_label`
            (containing a filename relative to `feature_dir`
            in which some of the patient's features are stored).
        feature_dir:
            See `slide_table`.
        output_dir:
            Path into which to output the artifacts (trained model etc.)
            generated during training.
        patient_label:
            See `clini_table`, `slide_table`.
        ground_truth_label:
            See `clini_table`.
        filename_label:
            See `slide_table`.
        categories:
            Categories of the ground truth.
            Set to `None` to automatically infer.
    """
    feature_type = detect_feature_type(feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type == "tile":
        # Tile-level: use slide_table
        patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
            clini_table_path=clini_table,
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
        )
        slide_to_patient = slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )
        patient_to_data = filter_complete_patient_data_(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=True,
        )
    elif feature_type == "patient":
        # Patient-level: ignore slide_table
        if slide_table is not None:
            _logger.warning("slide_table is ignored for patient-level features.")
        patient_to_data = load_patient_level_data(
            clini_table=clini_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )
    elif feature_type == "slide":
        raise RuntimeError(
            "Slide-level features are not supported for training."
            "Please rerun the encoding step with patient-level encoding."
        )
    else:
        raise RuntimeError(f"Unknown feature type: {feature_type}")

    # Train the model (the rest of the logic is unchanged)
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
        train_transform=(
            VaryPrecisionTransform(min_fraction_bits=1)
            if use_vary_precision_transform
            else None
        ),
        use_alibi=use_alibi,
        feature_type=feature_type,
    )
    train_model_(
        output_dir=output_dir,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
    )


def setup_model_for_training(
    *,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth]],
    categories: Sequence[Category] | None,
    bag_size: int,
    batch_size: int,
    num_workers: int,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    use_alibi: bool,
    # Metadata, has no effect on model training
    ground_truth_label: PandasLabel,
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    feature_type: str,
) -> tuple[
    lightning.LightningModule,
    DataLoader,
    DataLoader,
]:
    """Creates a model and dataloaders for training"""

    train_dl, valid_dl, train_categories, dim_feats, train_patients, valid_patients = (
        setup_dataloaders_for_training(
            patient_to_data=patient_to_data,
            categories=categories,
            bag_size=bag_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            feature_type=feature_type,
        )
    )

    category_weights = _compute_class_weights_and_check_categories(
        train_dl=train_dl,
        feature_type=feature_type,
        train_categories=train_categories,
    )

    # Model selection
    if feature_type == "tile":
        model = LitVisionTransformer(
            categories=train_categories,
            category_weights=category_weights,
            dim_input=dim_feats,
            dim_model=512,
            dim_feedforward=2048,
            n_heads=8,
            n_layers=2,
            dropout=0.25,
            use_alibi=use_alibi,
            ground_truth_label=ground_truth_label,
            train_patients=train_patients,
            valid_patients=valid_patients,
            clini_table=clini_table,
            slide_table=slide_table,
            feature_dir=feature_dir,
        )
    else:
        model = LitMLPClassifier(
            categories=train_categories,
            category_weights=category_weights,
            dim_input=dim_feats,
            dim_hidden=512,
            num_layers=2,
            dropout=0.25,
            ground_truth_label=ground_truth_label,
            train_patients=train_patients,
            valid_patients=valid_patients,
            clini_table=clini_table,
            slide_table=slide_table,
            feature_dir=feature_dir,
        )

    return model, train_dl, valid_dl


def setup_dataloaders_for_training(
    *,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth]],
    categories: Sequence[Category] | None,
    bag_size: int,
    batch_size: int,
    num_workers: int,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    feature_type: str,
) -> tuple[
    DataLoader,
    DataLoader,
    Sequence[Category],
    int,
    Sequence[PatientId],
    Sequence[PatientId],
]:
    """
    Creates train/val dataloaders for tile-level or patient-level features.

    Returns:
        train_dl, valid_dl, categories, feature_dim, train_patients, valid_patients
    """
    # Stratified split
    ground_truths = [
        patient_data.ground_truth
        for patient_data in patient_to_data.values()
        if patient_data.ground_truth is not None
    ]
    if len(ground_truths) != len(patient_to_data):
        raise ValueError(
            "patient_to_data must have a ground truth defined for all targets!"
        )

    train_patients, valid_patients = cast(
        tuple[Sequence[PatientId], Sequence[PatientId]],
        train_test_split(
            list(patient_to_data), stratify=ground_truths, shuffle=True, random_state=0
        ),
    )

    if feature_type == "tile":
        # Use existing BagDataset logic
        train_dl, train_categories = tile_bag_dataloader(
            patient_data=[patient_to_data[pid] for pid in train_patients],
            categories=categories,
            bag_size=bag_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=train_transform,
        )
        valid_dl, _ = tile_bag_dataloader(
            patient_data=[patient_to_data[pid] for pid in valid_patients],
            bag_size=None,
            categories=train_categories,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            transform=None,
        )
        bags, _, _, _ = next(iter(train_dl))
        dim_feats = bags.shape[-1]
        return (
            train_dl,
            valid_dl,
            train_categories,
            dim_feats,
            train_patients,
            valid_patients,
        )

    elif feature_type == "patient":
        train_dl, train_categories = patient_feature_dataloader(
            patient_data=[patient_to_data[pid] for pid in train_patients],
            categories=categories,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=train_transform,
        )
        valid_dl, _ = patient_feature_dataloader(
            patient_data=[patient_to_data[pid] for pid in valid_patients],
            categories=train_categories,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            transform=None,
        )
        feats, _ = next(iter(train_dl))
        dim_feats = feats.shape[-1]
        return (
            train_dl,
            valid_dl,
            train_categories,
            dim_feats,
            train_patients,
            valid_patients,
        )
    else:
        raise RuntimeError(
            f"Unsupported feature type: {feature_type}. Only 'tile' and 'patient' are supported."
        )
    

def train_model_(
    *,
    output_dir: Path,
    model: lightning.LightningModule,
    train_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    valid_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
) -> lightning.LightningModule:
    """Trains a model.

    Returns:
        The model with the best validation loss during training.
    """
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
        log_every_n_steps=len(train_dl),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
    shutil.copy(model_checkpoint.best_model_path, output_dir / "model.ckpt")

    # Reload the best model using the same class as the input model
    ModelClass = type(model)
    return ModelClass.load_from_checkpoint(model_checkpoint.best_model_path)


def _compute_class_weights_and_check_categories(
    *,
    train_dl: DataLoader,
    feature_type: str,
    train_categories: Sequence[str],
) -> torch.Tensor:
    """
    Computes class weights and checks for category issues.
    Logs warnings if there are too few or underpopulated categories.
    Returns normalized category weights as a torch.Tensor.
    """
    if feature_type == "tile":
        category_counts = cast(BagDataset, train_dl.dataset).ground_truths.sum(dim=0)
    else:
        category_counts = cast(
            PatientFeatureDataset, train_dl.dataset
        ).ground_truths.sum(dim=0)
    cat_ratio_reciprocal = category_counts.sum() / category_counts
    category_weights = cat_ratio_reciprocal / cat_ratio_reciprocal.sum()

    if len(train_categories) <= 1:
        raise ValueError(f"not enough categories to train on: {train_categories}")
    elif any(category_counts < 16):
        underpopulated_categories = {
            category: int(count)
            for category, count in zip(train_categories, category_counts, strict=True)
            if count < 16
        }
        _logger.warning(
            f"Some categories do not have enough samples to meaningfully train a model: {underpopulated_categories}. "
            "You may want to consider removing these categories; the model will likely overfit on the few samples available."
        )
    return category_weights
