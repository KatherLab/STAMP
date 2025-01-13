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
    Category,
    CoordinatesBatch,
    GroundTruth,
    PandasLabel,
    PatientData,
    PatientId,
    dataloader_from_patient_data,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.lightning_model import (
    Bags,
    BagSizes,
    EncodedTargets,
    LitVisionTransformer,
)
from stamp.modeling.transforms import VaryPrecisionTransform

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024 Marko van Treeck"
__license__ = "MIT"


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
    """Trains a model.

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
    # Read and parse data from out clini and slide table
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

    # Clean data (remove slides without ground truth, missing features, etc.)
    patient_to_data = filter_complete_patient_data_(
        patient_to_ground_truth=patient_to_ground_truth,
        slide_to_patient=slide_to_patient,
        drop_patients_with_missing_ground_truth=True,
    )

    # Train the model
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


def train_model_(
    *,
    output_dir: Path,
    model: LitVisionTransformer,
    train_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    valid_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
) -> LitVisionTransformer:
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
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
    shutil.copy(model_checkpoint.best_model_path, output_dir / "model.ckpt")

    return LitVisionTransformer.load_from_checkpoint(model_checkpoint.best_model_path)


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
) -> tuple[
    LitVisionTransformer,
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
]:
    """Creates a model and dataloaders for training"""

    # Do a stratified train-validation split
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

    train_dl, train_categories = dataloader_from_patient_data(
        patient_data=[patient_to_data[patient] for patient in train_patients],
        categories=categories,
        bag_size=bag_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_transform,
    )
    del categories  # Let's not accidentally reuse the original categories
    valid_dl, _ = dataloader_from_patient_data(
        patient_data=[patient_to_data[patient] for patient in valid_patients],
        bag_size=None,  # Use all the patient data for validation
        categories=train_categories,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
    )
    if overlap := set(train_patients) & set(valid_patients):
        raise RuntimeError(
            f"unreachable: unexpected overlap between training and validation set: {overlap}"
        )

    # Sample one bag to infer the input dimensions of the model
    bags, coords, bag_sizes, targets = cast(
        tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets], next(iter(train_dl))
    )
    _, _, dim_feats = bags.shape

    # Weigh classes inversely to their occurrence
    category_counts = cast(BagDataset, train_dl.dataset).ground_truths.sum(dim=0)
    cat_ratio_reciprocal = category_counts.sum() / category_counts
    category_weights = cat_ratio_reciprocal / cat_ratio_reciprocal.sum()

    if len(train_categories) <= 1:
        raise ValueError(f"not enough categories to train on: {train_categories}")
    elif any(category_counts < 16):
        underpopulated_categories = {
            category: count
            for category, count in zip(train_categories, category_counts, strict=True)
            if count < 16
        }
        raise ValueError(
            f"some categories do not have enough samples to meaningfully train a model: {underpopulated_categories}"
        )

    # Train the model
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
        # Metadata, has no effect on model training
        ground_truth_label=ground_truth_label,
        train_patients=train_patients,
        valid_patients=valid_patients,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
    )

    return model, train_dl, valid_dl
