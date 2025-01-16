import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from stamp.cache import download_file
from stamp.preprocessing import ExtractorName, Microns, TilePixels, extract_


def test_if_feature_extraction_crashes(extractor=ExtractorName.CTRANSPATH) -> None:
    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    with tempfile.TemporaryDirectory(prefix="stamp_test_preprocessing_") as tmp_dir:
        dir = Path(tmp_dir)
        wsi_dir = dir / "wsis"
        wsi_dir.mkdir()
        (wsi_dir / "slide.svs").symlink_to(example_slide_path)

        try:
            extract_(
                wsi_dir=wsi_dir,
                output_dir=dir / "output",
                extractor=extractor,
                cache_dir=None,
                tile_size_px=TilePixels(224),
                tile_size_um=Microns(256.0),
                max_workers=min(os.cpu_count() or 1, 16),
                brightness_cutoff=224,
                canny_cutoff=0.02,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except ModuleNotFoundError:
            pytest.skip(f"dependencies for {extractor} not installed")
        except GatedRepoError:
            pytest.skip(f"cannot access gated repo for {extractor}")

        # Check if the file has any contents
        with h5py.File(next((dir / "output").glob("*/*.h5"))) as h5_file:
            just_extracted_feats = np.array(h5_file["feats"][:])  # pyright: ignore[reportIndexIssue]

        assert len(just_extracted_feats) > 0


def test_if_conch_feature_extraction_crashes() -> None:
    test_if_feature_extraction_crashes(ExtractorName.CONCH)


def test_if_uni_feature_extraction_crashes() -> None:
    test_if_feature_extraction_crashes(ExtractorName.UNI)


def test_if_dino_bloom_feature_extraction_crashes() -> None:
    test_if_feature_extraction_crashes(ExtractorName.DINO_BLOOM)


def test_if_virchow2_feature_extraction_crashes() -> None:
    test_if_feature_extraction_crashes(ExtractorName.VIRCHOW2)


def test_if_empty_feature_extraction_crashes() -> None:
    test_if_feature_extraction_crashes(ExtractorName.EMPTY)


def test_backward_compatability(extractor=ExtractorName.CTRANSPATH) -> None:
    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    with tempfile.TemporaryDirectory(prefix="stamp_test_preprocessing_") as tmp_dir:
        dir = Path(tmp_dir)
        wsi_dir = dir / "wsis"
        wsi_dir.mkdir()
        (wsi_dir / "slide.svs").symlink_to(example_slide_path)

        try:
            extract_(
                wsi_dir=wsi_dir,
                output_dir=dir / "output",
                extractor=extractor,
                cache_dir=None,
                tile_size_px=TilePixels(224),
                tile_size_um=Microns(256.0),
                max_workers=min(os.cpu_count() or 1, 16),
                brightness_cutoff=224,
                canny_cutoff=0.02,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except ModuleNotFoundError:
            pytest.skip(f"dependencies for {extractor} not installed")
        except GatedRepoError:
            pytest.skip(f"cannot access gated repo for {extractor}")

        reference_feature_path = download_file(
            url=f"https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4-{extractor}.h5",
            file_name=f"TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4-{extractor}.h5",
            sha256sum="f3f33b069c3ed860d2bdb7d65ca5db64936d7acee3ba1061a457a8cdb1bc67e3",
        )

        with h5py.File(reference_feature_path) as h5_file:
            reference_feats = h5_file["feats"][:]  # pyright: ignore[reportIndexIssue]
            reference_version = h5_file.attrs["stamp_version"]

        with h5py.File(next((dir / "output").glob("*/*.h5"))) as h5_file:
            just_extracted_feats = h5_file["feats"][:]  # pyright: ignore[reportIndexIssue]

        assert torch.allclose(
            torch.tensor(just_extracted_feats), torch.tensor(reference_feats)
        ), (
            f"extracted {extractor} features differ from those made with stamp version {reference_version}"
        )

def check_uni_backward_compatability() -> None:
    test_backward_compatability(ExtractorName.UNI)