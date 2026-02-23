import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from random_data import create_random_dataset, create_random_patient_level_dataset

from stamp.modeling.config import (
    AdvancedConfig,
    MlpModelParams,
    ModelParams,
    TrainConfig,
    VitModelParams,
)
from stamp.modeling.deploy import deploy_categorical_model_
from stamp.modeling.train import train_categorical_model_


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
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

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
def test_train_deploy_patient_level_integration(
    *,
    tmp_path: Path,
    feat_dim: int = 25,
    use_alibi: bool,
    use_vary_precision_transform: bool,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

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
        filename_label="slide_path",  # Not used for patient-level
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_workers=min(os.cpu_count() or 1, 16),
    )
