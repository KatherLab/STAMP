import logging
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import lightning
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from stamp.modeling.config import AdvancedConfig, Seed, TrainConfig
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
from stamp.modeling.registry import MODEL_REGISTRY, ModelName
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.types import (
    Bags,
    BagSizes,
    Category,
    CoordinatesBatch,
    EncodedTargets,
    GroundTruth,
    PandasLabel,
    PatientId,
)

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")



def train_categorical_model_(
    *,
    config: TrainConfig,
    advanced: AdvancedConfig,
) -> None:
    """Trains a model based on the feature type."""
    feature_type = detect_feature_type(config.feature_dir)
    _logger.info(f"Detected feature type: {feature_type}")

    if feature_type == "tile":
        if config.slide_table is None:
            raise ValueError("A slide table is required for tile-level modeling")
        patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
            clini_table_path=config.clini_table,
            ground_truth_label=config.ground_truth_label,
            patient_label=config.patient_label,
        )
        slide_to_patient = slide_to_patient_from_slide_table_(
            slide_table_path=config.slide_table,
            feature_dir=config.feature_dir,
            patient_label=config.patient_label,
            filename_label=config.filename_label,
        )
        patient_to_data = filter_complete_patient_data_(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=True,
        )
    elif feature_type == "patient":
        # Patient-level: ignore slide_table
        if config.slide_table is not None:
            _logger.warning("slide_table is ignored for patient-level features.")
        patient_to_data = load_patient_level_data(
            clini_table=config.clini_table,
            feature_dir=config.feature_dir,
            patient_label=config.patient_label,
            ground_truth_label=config.ground_truth_label,
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
        categories=config.categories,
        advanced=advanced,
        ground_truth_label=config.ground_truth_label,
        clini_table=config.clini_table,
        slide_table=config.slide_table,
        feature_dir=config.feature_dir,
        train_transform=(
            VaryPrecisionTransform(min_fraction_bits=1)
            if config.use_vary_precision_transform
            else None
        ),
        feature_type=feature_type,
    )
    train_model_(
        output_dir=config.output_dir,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        max_epochs=advanced.max_epochs,
        patience=advanced.patience,
        accelerator=advanced.accelerator,
    )


def setup_model_for_training(
    *,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth]],
    categories: Sequence[Category] | None,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    feature_type: str,
    advanced: AdvancedConfig,
    # Metadata, has no effect on model training
    ground_truth_label: PandasLabel,
    clini_table: Path,
    slide_table: Path | None,
    feature_dir: Path,
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
            bag_size=advanced.bag_size,
            batch_size=advanced.batch_size,
            num_workers=advanced.num_workers,
            train_transform=train_transform,
            feature_type=feature_type,
        )
    )

    _logger.info(
        "Training dataloaders: bag_size=%s, batch_size=%s, num_workers=%s",
        advanced.bag_size,
        advanced.batch_size,
        advanced.num_workers,
    )

    category_weights = _compute_class_weights_and_check_categories(
        train_dl=train_dl,
        feature_type=feature_type,
        train_categories=train_categories,
    )

    # 1. Default to a model if none is specified
    if advanced.model_name is None:
        advanced.model_name = ModelName.VIT if feature_type == "tile" else ModelName.MLP
        _logger.info(
            f"No model specified, defaulting to '{advanced.model_name.value}' for feature type '{feature_type}'"
        )

    # 2. Validate that the chosen model supports the feature type
    model_info = MODEL_REGISTRY[advanced.model_name]
    if feature_type not in model_info["supported_features"]:
        raise ValueError(
            f"Model '{advanced.model_name.value}' does not support feature type '{feature_type}'. "
            f"Supported types are: {model_info['supported_features']}"
        )

    # 3. Get model-specific hyperparameters
    model_specific_params = advanced.model_params.model_dump()[
        advanced.model_name.value
    ]

    # 4. Calculate total steps for scheduler
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * advanced.max_epochs

    # 5. Prepare common parameters
    common_params = {
        "categories": train_categories,
        "category_weights": category_weights,
        "dim_input": dim_feats,
        "total_steps": total_steps,
        "max_lr": advanced.max_lr,
        "div_factor": advanced.div_factor,
        # Metadata, has no effect on model training
        "model_name": advanced.model_name.value,
        "ground_truth_label": ground_truth_label,
        "train_patients": train_patients,
        "valid_patients": valid_patients,
        "clini_table": clini_table,
        "slide_table": slide_table,
        "feature_dir": feature_dir,
    }

    # 6. Instantiate the model dynamically
    ModelClass = model_info["model_class"]

    match advanced.model_name.value:
        case ModelName.VIT:
            from stamp.modeling.classifier.vision_tranformer import (
                VisionTransformer as Classifier,
            )

        case ModelName.TRANS_MIL:
            from stamp.modeling.classifier.trans_mil import TransMIL as Classifier

        case ModelName.MLP:
            from stamp.modeling.classifier.mlp import MLPClassifier as Classifier

        case ModelName.LINEAR:
            from stamp.modeling.classifier.mlp import LinearClassifier as Classifier

        case _:
            raise ValueError(f"Unknown model name: {advanced.model_name.value}")

    # 7. Build the backbone instance
    backbone = Classifier(
        dim_output=len(train_categories),
        dim_input=dim_feats,
        **model_specific_params,
    )

    _logger.info(
        f"Instantiating model '{advanced.model_name.value}' with parameters: {model_specific_params}"
    )
    _logger.info(
        "Other params: max_epochs=%s, patience=%s",
        advanced.max_epochs,
        advanced.patience,
    )

    model = ModelClass(
        **common_params,
        model=backbone,
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
    # Sample count for training
    log_total_class_summary(patient_to_data, categories)

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
    Seed.set(42)

    model_checkpoint = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:0.3f}",
    )
    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        # check_val_every_n_epoch=5,
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
        devices=[1],
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


def log_total_class_summary(
    patient_to_data: Mapping[PatientId, PatientData],
    categories: Sequence[Category] | None,
) -> None:
    ground_truths = [
        patient_data.ground_truth
        for patient_data in patient_to_data.values()
        if patient_data.ground_truth is not None
    ]
    cats = categories or sorted(set(ground_truths))
    counter = Counter(ground_truths)
    _logger.info(
        f"Total samples: {len(ground_truths)} | "
        + " | ".join([f"Class {cls}: {counter.get(cls, 0)}" for cls in cats])
    )
