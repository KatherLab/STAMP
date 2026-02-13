import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from random_data import (
    create_random_dataset,
    create_random_multi_target_dataset,
    create_random_patient_level_dataset,
    create_random_patient_level_survival_dataset,
    create_random_regression_dataset,
    create_random_survival_dataset,
)

from stamp.modeling.config import (
    AdvancedConfig,
    MlpModelParams,
    ModelParams,
    TrainConfig,
    VitModelParams,
)
from stamp.modeling.deploy import deploy_categorical_model_
from stamp.modeling.registry import ModelName
from stamp.modeling.train import train_categorical_model_
from stamp.utils.seed import Seed


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
@pytest.mark.parametrize(
    "use_alibi,use_vary_precision_transform",
    [
        pytest.param(False, False, id="no experimental features"),
        pytest.param(True, False, id="use alibi"),
        pytest.param(False, True, id="use vary_precision_transform"),
    ],
)
def test_train_deploy_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
    use_alibi: bool,
    use_vary_precision_transform: bool,
) -> None:
    Seed.set(42)

    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    train_clini_path, train_slide_path, train_feature_dir, categories = (
        create_random_dataset(
            dir=tmp_path / "train",
            n_categories=3,
            n_patients=400,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = create_random_dataset(
        dir=tmp_path / "deploy",
        categories=categories,
        n_patients=50,
        max_slides_per_patient=3,
        min_tiles_per_slide=20,
        max_tiles_per_slide=600,
        feat_dim=feat_dim,
    )

    config = TrainConfig(
        clini_table=train_clini_path,
        slide_table=train_slide_path,
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        ground_truth_label="ground-truth",
        filename_label="slide_path",
        categories=categories,
        use_vary_precision_transform=use_vary_precision_transform,
    )

    advanced = AdvancedConfig(
        # Dataset and -loader parameters
        bag_size=500,
        num_workers=min(os.cpu_count() or 1, 16),
        # Training paramenters
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(
            vit=VitModelParams(use_alibi=use_alibi), mlp=MlpModelParams()
        ),
    )

    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label="ground-truth",
        time_label=None,
        status_label=None,
        filename_label="slide_path",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
@pytest.mark.parametrize(
    "use_alibi,use_vary_precision_transform",
    [
        pytest.param(False, False, id="no experimental features"),
        pytest.param(True, False, id="use alibi"),
        pytest.param(False, True, id="use vary_precision_transform"),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*violates type hint.*not instance of tuple:UserWarning"
)
def test_train_deploy_patient_level_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
    use_alibi: bool,
    use_vary_precision_transform: bool,
) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    train_clini_path, train_slide_path, train_feature_dir, categories = (
        create_random_patient_level_dataset(
            dir=tmp_path / "train",
            n_categories=3,
            n_patients=400,
            feat_dim=feat_dim,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = (
        create_random_patient_level_dataset(
            dir=tmp_path / "deploy",
            categories=categories,
            n_patients=50,
            feat_dim=feat_dim,
        )
    )

    config = TrainConfig(
        clini_table=train_clini_path,
        slide_table=None,  # Not needed for patient-level
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        ground_truth_label="ground-truth",
        filename_label="slide_path",  # Not used for patient-level
        categories=categories,
        use_vary_precision_transform=use_vary_precision_transform,
    )

    advanced = AdvancedConfig(
        # Dataset and -loader parameters
        bag_size=1,  # Not used for patient-level, but required by signature
        num_workers=min(os.cpu_count() or 1, 16),
        # Training paramenters
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(
            vit=VitModelParams(use_alibi=use_alibi), mlp=MlpModelParams()
        ),
    )

    train_categorical_model_(
        config=config,
        advanced=advanced,
    )

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=None,  # Not needed for patient-level
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label="ground-truth",
        time_label=None,
        status_label=None,
        filename_label="slide_path",  # Not used for patient-level
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
def test_train_deploy_regression_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
) -> None:
    """Integration test: train + deploy a tile-level regression model."""
    Seed.set(42)

    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    # --- Create random tile-level regression dataset ---
    train_clini_path, train_slide_path, train_feature_dir, _ = (
        create_random_regression_dataset(
            dir=tmp_path / "train",
            n_patients=400,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = (
        create_random_regression_dataset(
            dir=tmp_path / "deploy",
            n_patients=50,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
        )
    )

    # --- Build config objects ---
    config = TrainConfig(
        task="regression",
        clini_table=train_clini_path,
        slide_table=train_slide_path,
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        ground_truth_label="target",  # numeric regression target
        filename_label="slide_path",
        categories=None,
    )

    advanced = AdvancedConfig(
        bag_size=500,
        num_workers=min(os.cpu_count() or 1, 16),
        batch_size=1,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(
            vit=VitModelParams(),
            mlp=MlpModelParams(),
        ),
    )

    # --- Train + deploy regression model ---
    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label="target",
        time_label=None,
        status_label=None,
        filename_label="slide_path",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
def test_train_deploy_survival_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
) -> None:
    """Integration test: train + deploy a tile-level survival model."""
    Seed.set(42)

    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    # --- Create random tile-level survival dataset ---
    train_clini_path, train_slide_path, train_feature_dir, _ = (
        create_random_survival_dataset(
            dir=tmp_path / "train",
            n_patients=400,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = (
        create_random_survival_dataset(
            dir=tmp_path / "deploy",
            n_patients=50,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
        )
    )

    # --- Build config objects ---
    config = TrainConfig(
        task="survival",
        clini_table=train_clini_path,
        slide_table=train_slide_path,
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        time_label="day",  # raw ground-truth columns
        status_label="status",
        filename_label="slide_path",
    )

    advanced = AdvancedConfig(
        bag_size=500,
        num_workers=min(os.cpu_count() or 1, 16),
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(
            vit=VitModelParams(),
            mlp=MlpModelParams(),
        ),
    )

    # --- Train + deploy survival model ---
    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label=None,
        time_label="day",
        status_label="status",
        filename_label="slide_path",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:.*violates type hint.*not instance of tuple:UserWarning"
)
def test_train_deploy_patient_level_regression_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
) -> None:
    """Integration test: train + deploy a patient-level regression model."""
    Seed.set(42)

    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    # --- Create patient-level regression datasets ---
    train_clini_path = tmp_path / "train" / "clini.csv"
    deploy_clini_path = tmp_path / "deploy" / "clini.csv"
    train_slide_path = tmp_path / "train" / "slide.csv"
    deploy_slide_path = tmp_path / "deploy" / "slide.csv"
    train_feat_dir = tmp_path / "train" / "feats"
    deploy_feat_dir = tmp_path / "deploy" / "feats"
    train_feat_dir.mkdir(parents=True, exist_ok=True)
    deploy_feat_dir.mkdir(parents=True, exist_ok=True)

    n_train, n_deploy = 300, 60
    train_rows, deploy_rows = [], []

    # --- Generate random patient-level features and numeric targets ---
    for i in range(n_train):
        patient_id = f"train_pt_{i:04d}"
        feats = torch.randn(1, feat_dim)
        with h5py.File(train_feat_dir / f"{patient_id}.h5", "w") as f:
            f["feats"] = feats.numpy()
            f.attrs["extractor"] = "random-test-generator"
            f.attrs["feat_type"] = "patient"
        target = float(np.random.uniform(0.0, 100.0))  # ensure float
        train_rows.append((patient_id, target))

    for i in range(n_deploy):
        patient_id = f"deploy_pt_{i:04d}"
        feats = torch.randn(1, feat_dim)
        with h5py.File(deploy_feat_dir / f"{patient_id}.h5", "w") as f:
            f["feats"] = feats.numpy()
            f.attrs["extractor"] = "random-test-generator"
            f.attrs["feat_type"] = "patient"
        target = float(np.random.uniform(0.0, 100.0))  # ensure float
        deploy_rows.append((patient_id, target))

    # --- Write clini tables (force float dtype) ---
    train_df = pd.DataFrame(train_rows, columns=["patient", "target"])
    deploy_df = pd.DataFrame(deploy_rows, columns=["patient", "target"])
    train_df["target"] = train_df["target"].astype(float)
    deploy_df["target"] = deploy_df["target"].astype(float)
    train_df.to_csv(train_clini_path, index=False, float_format="%.6f")
    deploy_df.to_csv(deploy_clini_path, index=False, float_format="%.6f")

    # --- Dummy slide tables (required by current code) ---
    pd.DataFrame(
        {
            "slide_path": [f"{pid}.h5" for pid, _ in train_rows],
            "patient": [pid for pid, _ in train_rows],
        }
    ).to_csv(train_slide_path, index=False)
    pd.DataFrame(
        {
            "slide_path": [f"{pid}.h5" for pid, _ in deploy_rows],
            "patient": [pid for pid, _ in deploy_rows],
        }
    ).to_csv(deploy_slide_path, index=False)

    # --- Build train + advanced configs ---
    config = TrainConfig(
        task="regression",
        clini_table=train_clini_path,
        slide_table=train_slide_path,  # dummy table
        feature_dir=train_feat_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        ground_truth_label="target",
        filename_label="slide_path",
    )

    advanced = AdvancedConfig(
        bag_size=1,
        num_workers=min(os.cpu_count() or 1, 16),
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(vit=VitModelParams(), mlp=MlpModelParams()),
    )

    # --- Train + deploy ---
    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,  # dummy table
        feature_dir=deploy_feat_dir,
        patient_label="patient",
        ground_truth_label="target",
        time_label=None,
        status_label=None,
        filename_label="slide_path",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:.*violates type hint.*not instance of tuple:UserWarning"
)
def test_train_deploy_patient_level_survival_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
) -> None:
    """Integration test: train + deploy a patient-level survival model."""
    Seed.set(42)
    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    # --- Create patient-level survival dataset ---
    train_clini_path, train_slide_path, train_feature_dir, _ = (
        create_random_patient_level_survival_dataset(
            dir=tmp_path / "train",
            n_patients=300,
            feat_dim=feat_dim,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = (
        create_random_patient_level_survival_dataset(
            dir=tmp_path / "deploy",
            n_patients=60,
            feat_dim=feat_dim,
        )
    )

    # --- Train config ---
    config = TrainConfig(
        task="survival",
        clini_table=train_clini_path,
        slide_table=train_slide_path,  # dummy slide.csv (empty)
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        time_label="day",
        status_label="status",
        filename_label="slide_path",  # unused, for API compatibility
    )

    advanced = AdvancedConfig(
        bag_size=1,
        num_workers=min(os.cpu_count() or 1, 16),
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(vit=VitModelParams(), mlp=MlpModelParams()),
    )

    # --- Train + deploy ---
    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,  # dummy slide.csv (empty)
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label=None,
        time_label="day",
        status_label="status",
        filename_label="slide_path",  # unused
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
def test_train_deploy_multi_target_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
) -> None:
    """Integration test: train + deploy a multi-target tile-level classification model."""
    Seed.set(42)

    (tmp_path / "train").mkdir()
    (tmp_path / "deploy").mkdir()

    # Define multi-target setup: subtype (2 categories) and grade (3 categories)
    target_labels = ["subtype", "grade"]
    categories_per_target = [["A", "B"], ["1", "2", "3"]]

    # Create random multi-target tile-level dataset
    train_clini_path, train_slide_path, train_feature_dir, _ = (
        create_random_multi_target_dataset(
            dir=tmp_path / "train",
            n_patients=400,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
            target_labels=target_labels,
            categories_per_target=categories_per_target,
        )
    )
    deploy_clini_path, deploy_slide_path, deploy_feature_dir, _ = (
        create_random_multi_target_dataset(
            dir=tmp_path / "deploy",
            n_patients=50,
            max_slides_per_patient=3,
            min_tiles_per_slide=20,
            max_tiles_per_slide=600,
            feat_dim=feat_dim,
            target_labels=target_labels,
            categories_per_target=categories_per_target,
        )
    )

    # Build config objects
    config = TrainConfig(
        task="classification",
        clini_table=train_clini_path,
        slide_table=train_slide_path,
        feature_dir=train_feature_dir,
        output_dir=tmp_path / "train_output",
        patient_label="patient",
        ground_truth_label=target_labels,
        filename_label="slide_path",
        categories=[cat for cats in categories_per_target for cat in cats],
    )

    advanced = AdvancedConfig(
        bag_size=500,
        num_workers=min(os.cpu_count() or 1, 16),
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_params=ModelParams(),
        model_name=ModelName.BARSPOON,
    )

    # Train + deploy multi-target model
    train_categorical_model_(config=config, advanced=advanced)

    deploy_categorical_model_(
        output_dir=tmp_path / "deploy_output",
        checkpoint_paths=[tmp_path / "train_output" / "model.ckpt"],
        clini_table=deploy_clini_path,
        slide_table=deploy_slide_path,
        feature_dir=deploy_feature_dir,
        patient_label="patient",
        ground_truth_label=target_labels,
        time_label=None,
        status_label=None,
        filename_label="slide_path",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )
