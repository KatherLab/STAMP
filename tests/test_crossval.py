import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from random_data import create_random_dataset, create_random_patient_level_dataset

from stamp.modeling.config import (
    AdvancedConfig,
    CrossvalConfig,
    MlpModelParams,
    ModelParams,
    VitModelParams,
)
from stamp.modeling.crossval import categorical_crossval_


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
@pytest.mark.parametrize("feature_type", ["tile", "patient"])
def test_crossval_integration(
    tmp_path: Path,
    feature_type: str,
    n_patients: int = 80,
    max_slides_per_patient: int = 3,
    min_tiles_per_slide: int = 8,
    max_tiles_per_slide: int = 32,
    feat_dim: int = 8,
    n_categories: int = 3,
    use_alibi: bool = False,
    use_vary_precision_transform: bool = False,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    if feature_type == "tile":
        clini_path, slide_path, feature_dir, categories = create_random_dataset(
            dir=tmp_path,
            n_categories=n_categories,
            n_patients=n_patients,
            max_slides_per_patient=max_slides_per_patient,
            min_tiles_per_slide=min_tiles_per_slide,
            max_tiles_per_slide=max_tiles_per_slide,
            feat_dim=feat_dim,
        )
    elif feature_type == "patient":
        clini_path, slide_path, feature_dir, categories = (
            create_random_patient_level_dataset(
                dir=tmp_path,
                n_categories=n_categories,
                n_patients=n_patients,
                feat_dim=feat_dim,
            )
        )
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    output_dir = tmp_path / "output"

    config = CrossvalConfig(
        clini_table=clini_path,
        slide_table=slide_path,
        output_dir=output_dir,
        patient_label="patient",
        ground_truth_label="ground-truth",
        filename_label="slide_path",
        categories=categories,
        feature_dir=feature_dir,
        n_splits=2,
        use_vary_precision_transform=use_vary_precision_transform,
    )

    advanced = AdvancedConfig(
        # Dataset and -loader parameters
        bag_size=max_tiles_per_slide // 2,
        num_workers=min(os.cpu_count() or 1, 7),
        # Training paramenters
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # Experimental features
        model_params=ModelParams(
            vit=VitModelParams(
                use_alibi=use_alibi,
            ),
            mlp=MlpModelParams(),
        ),
    )

    categorical_crossval_(
        config=config,
        advanced=advanced,
    )
