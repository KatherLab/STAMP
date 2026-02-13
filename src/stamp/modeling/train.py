import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import lightning
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from stamp.modeling.config import AdvancedConfig, TrainConfig
from stamp.modeling.data import (
    BagDataset,
    PatientData,
    PatientFeatureDataset,
    create_dataloader,
    load_patient_data_,
)
from stamp.modeling.registry import ModelName, load_model_class
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
    Task,
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

    # Train the model (the rest of the logic is unchanged)
    model, train_dl, valid_dl = setup_model_for_training(
        patient_to_data=patient_to_data,
        categories=config.categories,
        task=config.task,
        advanced=advanced,
        ground_truth_label=config.ground_truth_label,
        time_label=config.time_label,
        status_label=config.status_label,
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
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth | None | dict]],
    task: Task,
    categories: Sequence[Category] | None,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    feature_type: str,
    advanced: AdvancedConfig,
    # Metadata, has no effect on model training
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None,
    time_label: PandasLabel | None,
    status_label: PandasLabel | None,
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
            task=task,
            categories=categories,
            bag_size=advanced.bag_size,
            batch_size=advanced.batch_size,
            num_workers=advanced.num_workers,
            train_transform=train_transform,
            feature_type=feature_type,
        )
    )

    _logger.info(
        "Training dataloaders: bag_size=%s, batch_size=%s, num_workers=%s, task=%s",
        advanced.bag_size,
        advanced.batch_size,
        advanced.num_workers,
        task,
    )
    ##temopary for test regression
    category_weights = []
    if task == "classification":
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

    # 2. Instantiate the lightning wrapper (based on provided task, feature type) and model backbone dynamically
    LitModelClass, ModelClass = load_model_class(
        task, feature_type, advanced.model_name
    )
    print(f"Using Lightning wrapper class: {LitModelClass}")

    # 3. Validate that the chosen model supports the feature type
    if feature_type not in LitModelClass.supported_features:
        raise ValueError(
            f"Model '{advanced.model_name.value}' does not support feature type '{feature_type}'. "
            f"Supported types are: {LitModelClass.supported_features}"
        )
    elif (
        feature_type in ("slide", "patient")
        and advanced.model_name.value.lower() != "mlp"
    ):
        raise ValueError(
            f"Feature type '{feature_type}' only supports MLP backbones. "
            f"Got '{advanced.model_name.value}'. Please set model_name='mlp'."
        )

    # 4. Get model-specific hyperparameters
    model_specific_params = (
        advanced.model_params.model_dump().get(advanced.model_name.value) or {}
    )

    # 5. Calculate total steps for scheduler
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * advanced.max_epochs

    # 6. Prepare common parameters
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
        "time_label": time_label,
        "status_label": status_label,
        "train_patients": train_patients,
        "valid_patients": valid_patients,
        "clini_table": clini_table,
        "slide_table": slide_table,
        "feature_dir": feature_dir,
    }

    all_params = {**common_params, **model_specific_params}

    _logger.info(
        f"Instantiating model '{advanced.model_name.value}' with parameters: {model_specific_params}"
    )
    _logger.info(
        "Other params: max_epochs=%s, patience=%s",
        advanced.max_epochs,
        advanced.patience,
    )

    model = LitModelClass(model_class=ModelClass, **all_params)

    return model, train_dl, valid_dl


def setup_dataloaders_for_training(
    *,
    patient_to_data: Mapping[PatientId, PatientData[GroundTruth | None | dict]],
    task: Task,
    categories: Sequence[Category] | None,
    bag_size: int,
    batch_size: int,
    num_workers: int,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    feature_type: str,
) -> tuple[
    DataLoader,
    DataLoader,
    Sequence[Category] | Mapping[str, Sequence[Category]],
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

    _logger.info(f"Task: {feature_type} {task}")

    if len(ground_truths) != len(patient_to_data):
        raise ValueError(
            "patient_to_data must have a ground truth defined for all targets!"
        )

    if task == "classification":
        # Handle both single and multi-target cases
        if ground_truths and isinstance(ground_truths[0], dict):
            # Multi-target: use first target for stratification
            first_key = list(ground_truths[0].keys())[0]
            stratify = [cast(dict, gt)[first_key] for gt in ground_truths]
        else:
            stratify = ground_truths
    elif task == "survival":
        # Extract event indicator (status) - handle both single and multi-target
        statuses = []
        for gt in ground_truths:
            if isinstance(gt, dict):
                # Multi-target survival: extract from first target
                first_key = list(gt.keys())[0]
                val = cast(dict, gt)[first_key]
                if val:
                    statuses.append(int(val.split()[1]))
            else:
                statuses.append(int(gt.split()[1]))
        stratify = statuses
    elif task == "regression":
        stratify = None

    train_patients, valid_patients = cast(
        tuple[Sequence[PatientId], Sequence[PatientId]],
        train_test_split(
            list(patient_to_data),
            stratify=cast(Any, stratify),
            shuffle=True,
            random_state=0,
        ),
    )

    if feature_type in ("tile", "slide", "patient"):
        # Build train/valid dataloaders
        train_dl, train_categories = create_dataloader(
            feature_type=feature_type,
            task=task,
            patient_data=[patient_to_data[pid] for pid in train_patients],
            bag_size=bag_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            transform=train_transform,
            categories=categories,
        )

        valid_dl, _ = create_dataloader(
            feature_type=feature_type,
            task=task,
            patient_data=[patient_to_data[pid] for pid in valid_patients],
            bag_size=None,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            transform=None,
            categories=train_categories,
        )

        # Infer feature dimension automatically
        batch = next(iter(train_dl))
        if feature_type == "tile":
            bags, _, _, _ = batch
            dim_feats = bags.shape[-1]
        else:
            feats, _ = batch
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
            f"Unsupported feature type: {feature_type}. "
            "Only 'tile', 'slide', and 'patient' are supported."
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

    # Decide monitor metric based on task
    task = getattr(model.hparams, "task", None)
    if task == "survival":
        monitor_metric, mode = "val_cindex", "max"
    else:  # regression or classification
        monitor_metric, mode = "validation_loss", "min"

    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_metric}:0.3f}}",
    )
    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        # check_val_every_n_epoch=5,
        callbacks=[
            EarlyStopping(monitor=monitor_metric, mode=mode, patience=patience),
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
        # gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=output_dir),
        log_every_n_steps=len(train_dl),
        num_sanity_val_steps=0,
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
    train_categories: Sequence[str] | Mapping[str, Sequence[str]],
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Computes class weights and checks for category issues.
    Logs warnings if there are too few or underpopulated categories.
    Returns normalized category weights as a torch.Tensor.
    """
    if feature_type == "tile":
        dataset = cast(BagDataset, train_dl.dataset)

        if isinstance(dataset.ground_truths, list):
            # Multi-target case: compute weights per target head
            weights_per_target: dict[str, torch.Tensor] = {}

            target_keys = dataset.ground_truths[0].keys()

            for key in target_keys:
                stacked = torch.stack([gt[key] for gt in dataset.ground_truths], dim=0)
                counts = stacked.sum(dim=0)
                w = counts.sum() / counts
                weights_per_target[key] = w / w.sum()

            return weights_per_target
        else:
            category_counts = dataset.ground_truths.sum(dim=0)
    else:
        dataset = cast(PatientFeatureDataset, train_dl.dataset)
        category_counts = dataset.ground_truths.sum(dim=0)
    cat_ratio_reciprocal = category_counts.sum() / category_counts
    category_weights = cat_ratio_reciprocal / cat_ratio_reciprocal.sum()

    if len(train_categories) <= 1:
        raise ValueError(f"not enough categories to train on: {train_categories}")
    elif (category_counts < 16).any():
        category_counts_list = (
            category_counts.tolist()
            if category_counts.dim() > 0
            else [category_counts.item()]
        )
        underpopulated_categories = {
            category: int(count)
            for category, count in zip(
                train_categories, category_counts_list, strict=True
            )
            if count < 16
        }
        _logger.warning(
            f"Some categories do not have enough samples to meaningfully train a model: {underpopulated_categories}. "
            "You may want to consider removing these categories; the model will likely overfit on the few samples available."
        )
    return category_weights
