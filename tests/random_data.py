"""Routines to create random data"""

import io
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
from jaxtyping import Float
from torch import Tensor

import stamp
from stamp.modeling.data import (
    Category,
    PatientId,
)
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.tiling import Microns, TilePixels

CliniPath: TypeAlias = Path
SlidePath: TypeAlias = Path
FeatureDir: TypeAlias = Path


def seed_rng(seed: int) -> None:
    """Seeds all the random number generators"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_random_dataset(
    *,
    dir: Path,
    n_patients: int,
    max_slides_per_patient: int,
    min_tiles_per_slide: int,
    max_tiles_per_slide: int,
    feat_dim: int,
    categories: Sequence[str] | None = None,
    n_categories: int | None = None,
    extractor_name: ExtractorName | str = "random-test-generator",
    min_slides_per_patient: int = 1,
) -> tuple[CliniPath, SlidePath, FeatureDir, Sequence[Category]]:
    slide_path_to_patient: Mapping[Path, PatientId] = {}
    patient_to_ground_truth: Mapping[PatientId, str] = {}
    clini_path = dir / "clini.csv"
    slide_path = dir / "slide.csv"

    feat_dir = dir / "feats"
    feat_dir.mkdir()

    if categories is not None:
        if n_categories is not None:
            raise ValueError("only one of `categories` and `n_categories` can be set")
    else:
        if n_categories is None:
            raise ValueError(
                "either `categories` or `n_categories` has to be specified"
            )
        categories = [random_string(8) for _ in range(n_categories)]

    for _ in range(n_patients):
        # Random patient ID
        patient_id = random_string(16)

        patient_to_ground_truth[patient_id] = random.choice(categories)

        # Generate some slides
        for _ in range(random.randint(min_slides_per_patient, max_slides_per_patient)):
            slide_path_to_patient[
                create_random_feature_file(
                    tmp_path=feat_dir,
                    min_tiles=min_tiles_per_slide,
                    max_tiles=max_tiles_per_slide,
                    feat_dim=feat_dim,
                    extractor_name=extractor_name,
                ).relative_to(feat_dir)
            ] = patient_id

    clini_df = pd.DataFrame(
        patient_to_ground_truth.items(),
        columns=["patient", "ground-truth"],  # pyright: ignore[reportArgumentType]
    )
    clini_df.to_csv(clini_path, index=False)

    slide_df = pd.DataFrame(
        slide_path_to_patient.items(),
        columns=["slide_path", "patient"],  # pyright: ignore[reportArgumentType]
    )
    slide_df.to_csv(slide_path, index=False)

    return clini_path, slide_path, feat_dir, categories


def create_random_feature_file(
    *,
    tmp_path: Path,
    min_tiles: int,
    max_tiles: int,
    feat_dim: int,
    tile_size_um: Microns = Microns(2508),
    extractor_name: ExtractorName | str = "random-test-generator",
    feat_filename: str | None = None,
) -> Path:
    """Creates a h5 file with random contents.

    Args:
        dir:
            Directory to create the file in.

    Returns:
        Path to the feature file.
    """
    n_tiles = random.randint(min_tiles, max_tiles)
    if feat_filename is None:
        feat_filename = random_string(16)  # Generate a random filename
    feature_file_path = tmp_path / f"{feat_filename}.h5"
    with h5py.File(feature_file_path, "w") as h5_file:
        h5_file["feats"] = torch.rand(n_tiles, feat_dim) * 1000 * tile_size_um
        h5_file["coords"] = torch.rand(n_tiles, 2)

        h5_file.attrs["stamp_version"] = stamp.__version__
        h5_file.attrs["extractor"] = str(extractor_name)
        h5_file.attrs["unit"] = "um"
        h5_file.attrs["tile_size"] = tile_size_um
        return Path(feature_file_path)


def create_random_feature_file_with_agg(
    *,
    tmp_path: Path,
    min_tiles: int,
    max_tiles: int,
    feat_dim: int,
    tile_size_um: Microns = Microns(2508),
    extractor_name: ExtractorName | str = "random-test-generator",
    agg_path: Path,
    agg_feat_dim: int,
    agg_extractor_name: ExtractorName | str = "random-agg-test-generator",
) -> tuple[Path, Path]:
    """Creates a random feature file with its aggregated pair. Used
    for testing eagle encoder.

    Args:
        dir:
            Directory to create the file in.

    Returns:
        Path to the feature and aggregated file.
    """
    n_tiles = random.randint(min_tiles, max_tiles)
    random_filename = random_string(16)  # Generate a random filename

    feature_file_path = tmp_path / f"{random_filename}.h5"
    agg_file_path = agg_path / f"{random_filename}.h5"

    with h5py.File(feature_file_path, "w") as h5_file:
        h5_file["feats"] = torch.rand(n_tiles, feat_dim) * 1000 * tile_size_um
        h5_file["coords"] = torch.rand(n_tiles, 2)

        h5_file.attrs["stamp_version"] = stamp.__version__
        h5_file.attrs["extractor"] = extractor_name
        h5_file.attrs["unit"] = "um"
        h5_file.attrs["tile_size"] = tile_size_um

    with h5py.File(agg_file_path, "w") as h5_file:
        h5_file["feats"] = torch.rand(n_tiles, agg_feat_dim) * 1000 * tile_size_um
        h5_file["coords"] = torch.rand(n_tiles, 2)

        h5_file.attrs["stamp_version"] = stamp.__version__
        h5_file.attrs["extractor"] = agg_extractor_name
        h5_file.attrs["unit"] = "um"
        h5_file.attrs["tile_size"] = tile_size_um

    return Path(feature_file_path), Path(agg_file_path)


def random_patient_preds(*, n_patients: int, categories: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": [random_string(8) for _ in range(n_patients)],
            "ground-truth": [random.choice(categories) for _ in range(n_patients)],
            **{
                f"ground-truth_{cat}": scores
                for i, (cat, scores) in enumerate(
                    zip(
                        categories,
                        torch.softmax(torch.rand(len(categories), n_patients), dim=0),
                    )
                )
            },
        }
    )


def random_string(len: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=len))


def make_old_feature_file(
    *,
    feats: Float[Tensor, "tile feat_d"],
    coords: Float[Tensor, "tile 2"],
    tile_size_um: Microns = Microns(2508),
) -> io.BytesIO:
    """Creates a feature file with historic format from the given data"""
    file = io.BytesIO()
    with h5py.File(file, "w") as h5:
        h5["feats"] = feats
        h5["coords"] = coords * tile_size_um
        h5.attrs["stamp_version"] = stamp.__version__
        h5.attrs["extractor"] = "random-test-generator"
        h5.attrs["unit"] = "um"
        h5.attrs["tile_size"] = tile_size_um

    return file


def make_feature_file(
    *,
    feats: Float[Tensor, "tile feat_d"],
    coords: Float[Tensor, "tile 2"],
    tile_size_um: Microns = Microns(2508),
    tile_size_px: TilePixels = TilePixels(512),
) -> io.BytesIO:
    """Creates a feature file from the given data"""
    file = io.BytesIO()
    with h5py.File(file, "w") as h5:
        h5["feats"] = feats
        h5["coords"] = coords * tile_size_um
        h5.attrs["stamp_version"] = stamp.__version__
        h5.attrs["extractor"] = "random-test-generator"
        h5.attrs["unit"] = "um"
        h5.attrs["tile_size_um"] = tile_size_um
        h5.attrs["tile_size_px"] = tile_size_px

    return file
