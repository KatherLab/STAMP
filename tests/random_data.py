"""Routines to create random data"""

import io
import random
import string
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
from stamp.preprocessing.config import ExtractorName
from stamp.types import Category, FeaturePath, Microns, PatientId, TilePixels

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


def create_random_patient_level_dataset(
    *,
    dir: Path,
    n_patients: int,
    feat_dim: int,
    categories: Sequence[str] | None = None,
    n_categories: int | None = None,
) -> tuple[Path, Path, Path, Sequence[str]]:
    """
    Creates a random dataset with one .h5 file per patient (patient-level features).
    Returns (clini_path, slide_path, feat_dir, categories).
    slide_path is a dummy file (not used for patient-level).
    """
    clini_path = dir / "clini.csv"
    slide_path = dir / "slide.csv"  # Not used, but keep interface consistent
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

    patient_to_ground_truth = {}
    for _ in range(n_patients):
        patient_id = random_string(16)
        patient_to_ground_truth[patient_id] = random.choice(categories)
        # Create a single feature vector per patient
        create_random_patient_level_feature_file(
            tmp_path=feat_dir,
            feat_dim=feat_dim,
            feat_filename=patient_id,
        )

    pd.DataFrame(
        patient_to_ground_truth.items(),
        columns=["patient", "ground-truth"],
    ).to_csv(clini_path, index=False)

    # slide_path is not used for patient-level, but return a dummy file for API compatibility
    pd.DataFrame(columns=["slide_path", "patient"]).to_csv(slide_path, index=False)

    return clini_path, slide_path, feat_dir, categories


def create_random_feature_file(
    *,
    tmp_path: Path,
    min_tiles: int,
    max_tiles: int,
    feat_dim: int,
    tile_size_um: Microns = Microns(256),
    tile_size_px: TilePixels = TilePixels(224),
    extractor_name: ExtractorName | str = "random-test-generator",
    feat_filename: str | None = None,
    coords: np.ndarray | None = None,
) -> FeaturePath:
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
        rand_feats = torch.rand(n_tiles, feat_dim) * 1000 * tile_size_um
        mean = rand_feats.mean()
        std = rand_feats.std()
        norm_feats = (rand_feats - mean) / std
        h5_file["feats"] = norm_feats.numpy()
        if coords is not None:
            h5_file["coords"] = coords
        else:
            h5_file["coords"] = torch.rand(n_tiles, 2).numpy()
        h5_file.attrs["stamp_version"] = stamp.__version__
        h5_file.attrs["extractor"] = str(extractor_name)
        h5_file.attrs["unit"] = "um"
        h5_file.attrs["tile_size_um"] = tile_size_um
        h5_file.attrs["tile_size_px"] = tile_size_px
        return FeaturePath(feature_file_path)


def create_random_patient_level_feature_file(
    *,
    tmp_path: Path,
    feat_dim: int,
    feat_filename: str | None = None,
    encoder: str = "test-encoder",
    precision: str = "float32",
    feat_type: str = "patient",
    code_hash: str = "testhash",
    version: str | None = None,
) -> FeaturePath:
    """
    Creates a random patient-level feature .h5 file with the correct metadata.
    Returns the path to the created file.
    """
    if feat_filename is None:
        feat_filename = random_string(16)
    feature_file_path = tmp_path / f"{feat_filename}.h5"
    feats = torch.rand(1, feat_dim)
    version = version or stamp.__version__
    with h5py.File(feature_file_path, "w") as h5:
        h5["feats"] = feats.numpy()
        h5.attrs["version"] = version
        h5.attrs["encoder"] = encoder
        h5.attrs["precision"] = precision
        h5.attrs["stamp_version"] = version
        h5.attrs["code_hash"] = code_hash
        h5.attrs["feat_type"] = feat_type
    return FeaturePath(feature_file_path)


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
        h5.attrs["feat_type"] = "tile"

    return file


def make_patient_level_feature_file(
    *,
    feats: torch.Tensor,
    encoder: str = "test-encoder",
    precision: str = "float32",
    code_hash: str = "testhash",
    version: str | None = None,
) -> io.BytesIO:
    """
    Creates an in-memory patient-level feature .h5 file with the correct metadata.
    Returns a BytesIO object.
    """
    version = version or stamp.__version__
    file = io.BytesIO()
    with h5py.File(file, "w") as h5:
        h5["feats"] = feats.numpy()
        h5.attrs["version"] = version
        h5.attrs["encoder"] = encoder
        h5.attrs["precision"] = precision
        h5.attrs["stamp_version"] = version
        h5.attrs["code_hash"] = code_hash
        h5.attrs["feat_type"] = "patient"
    file.seek(0)
    return file


def create_good_and_bad_slide__tables(*, tmp_path: Path) -> tuple[Path, Path]:
    """
    Manually creates two slide tables for testing
    slide_to_patient_from_slide_table_ in data.py. Good slide tables
    contain .h5 extensions and bad slide tables do not.
    """

    # Create good slide table (with .h5 extension)
    good_slide_df = pd.DataFrame(
        {
            "PATIENT": ["pat1", "pat2", "pat3"],
            "FILENAME": ["slide1.h5", "slide2.h5", "slide3.h5"],
        }
    )
    good_slide_path = tmp_path / "good_slide.csv"
    good_slide_df.to_csv(good_slide_path, index=False)

    # Create bad slide table (no .h5 extension)
    bad_slide_df = pd.DataFrame(
        {
            "PATIENT": ["pat_bad1", "pat_bad2", "pat_bad3"],
            "FILENAME": ["slide1.jpg", "slide2.png", "slide3.tiff"],
        }
    )
    bad_slide_path = tmp_path / "bad_slide.csv"
    bad_slide_df.to_csv(bad_slide_path, index=False)

    return good_slide_path, bad_slide_path


def create_random_slide_tables(
        *,
        n_patients: int,
        tmp_path: Path) -> tuple[Path, Path]:
    """
    Randomly creates two slide tables for testing
    slide_to_patient_from_slide_table_ in data.py. Good slide tables
    contain .h5 extensions and bad slide tables do not.
    """
    bad_extensions = [
        ".jpg",
        ".pdf",
        ".png",
        ".bmp",
        ".gif",
        ".jpeg",
        ".svg",
        ".webp",
        ".tff",
        ".cur",
    ]

    names = []
    for i in range(n_patients):
        names.append("pat" + str(i))

    good_files = []
    bad_files = []
    for _ in range(n_patients):
        word = random_string(random.randint(5, 12))
        good_files.append(word + ".h5")
        bad_files.append(word + random.choice(bad_extensions))

    good_slide_df = pd.DataFrame({"PATIENT": names, "FILENAME": good_files})
    good_slide_path = tmp_path / "good_random_slide.csv"
    good_slide_df.to_csv(good_slide_path, index=False)

    bad_slide_df = pd.DataFrame({"PATIENT": names, "FILENAME": bad_files})
    bad_slide_path = tmp_path / "bad_random_slide.csv"
    bad_slide_df.to_csv(bad_slide_path, index=False)

    return good_slide_path, bad_slide_path
