import os
import random
import string
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
import pandas as pd
import torch

from stamp.modeling.crossval import categorical_crossval_
from stamp.modeling.data import Category, PatientId

CliniPath: TypeAlias = Path
SlidePath: TypeAlias = Path
FeatureDir: TypeAlias = Path


def test_crossval_integration(
    n_patients: int = 800,
    max_slides_per_patient: int = 3,
    min_tiles_per_slide: int = 8,
    max_tiles_per_slide: int = 2**10,
    feat_dim: int = 25,
    n_categories: int = 3,
    use_alibi: bool = False,
    use_vary_precision_transform: bool = False,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory(prefix="stamp_test_train_") as tmp_dir:
        clini_path, slide_path, feature_dir, categories = _create_random_dataset(
            dir=Path(tmp_dir),
            n_categories=n_categories,
            n_patients=n_patients,
            max_slides_per_patient=max_slides_per_patient,
            min_tiles_per_slide=min_tiles_per_slide,
            max_tiles_per_slide=max_tiles_per_slide,
            feat_dim=feat_dim,
        )

        output_dir = Path(tmp_dir) / "output"

        categorical_crossval_(
            clini_table=clini_path,
            slide_table=slide_path,
            feature_dir=feature_dir,
            output_dir=output_dir,
            patient_label="patient",
            ground_truth_label="ground_truth",
            filename_label="slide_path",
            categories=categories,
            # Dataset and -loader parameters
            bag_size=max_tiles_per_slide // 2,
            num_workers=min(os.cpu_count() or 1, 7),
            # Training paramenters
            batch_size=8,
            max_epochs=2,
            patience=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            n_splits=3,
            # Experimental features
            use_vary_precision_transform=use_vary_precision_transform,
            use_alibi=use_alibi,
        )


def _create_random_dataset(
    *,
    dir: Path,
    n_patients: int,
    max_slides_per_patient: int,
    min_tiles_per_slide: int,
    max_tiles_per_slide: int,
    feat_dim: int,
    n_categories: int,
) -> tuple[CliniPath, SlidePath, FeatureDir, Sequence[Category]]:
    slide_path_to_patient: Mapping[Path, PatientId] = {}
    patient_to_ground_truth: Mapping[PatientId, str] = {}
    clini_path = dir / "clini.csv"
    slide_path = dir / "slide.csv"

    feat_dir = dir / "feats"
    feat_dir.mkdir()

    categories = [_random_string(8) for _ in range(n_categories)]

    for _ in range(n_patients):
        # Random patient ID
        patient_id = _random_string(16)

        patient_to_ground_truth[patient_id] = random.choice(categories)

        # Generate some slides
        for _ in range(random.randint(1, max_slides_per_patient)):
            slide_path_to_patient[
                _create_random_feature_file(
                    dir=feat_dir,
                    min_tiles_per_slide=min_tiles_per_slide,
                    max_tiles_per_slide=max_tiles_per_slide,
                    feat_dim=feat_dim,
                ).relative_to(feat_dir)
            ] = patient_id

    clini_df = pd.DataFrame(
        patient_to_ground_truth.items(),
        columns=["patient", "ground_truth"],  # pyright: ignore[reportArgumentType]
    )
    clini_df.to_csv(clini_path, index=False)

    slide_df = pd.DataFrame(
        slide_path_to_patient.items(),
        columns=["slide_path", "patient"],  # pyright: ignore[reportArgumentType]
    )
    slide_df.to_csv(slide_path, index=False)

    return clini_path, slide_path, feat_dir, categories


def _create_random_feature_file(
    *, dir: Path, min_tiles_per_slide: int, max_tiles_per_slide: int, feat_dim: int
) -> Path:
    n_tiles = random.randint(min_tiles_per_slide, max_tiles_per_slide)
    with (
        tempfile.NamedTemporaryFile(dir=dir, suffix=".h5", delete=False) as tmp_file,
        h5py.File(tmp_file, "w") as h5_file,
    ):
        h5_file["feats"] = torch.rand(n_tiles, feat_dim)
        h5_file["coords"] = torch.rand(n_tiles, 2)
        return Path(tmp_file.name)


def _random_string(len: int):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=len))
