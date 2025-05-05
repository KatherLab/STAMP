import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from stamp.cache import download_file
from stamp.preprocessing import ExtractorName, Microns, TilePixels, extract_


@pytest.mark.slow
@pytest.mark.parametrize("extractor", ExtractorName)
@pytest.mark.filterwarnings("ignore:Importing from timm.models.layers is deprecated")
@pytest.mark.filterwarnings(
    "ignore:You are using `torch.load` with `weights_only=False`"
)
@pytest.mark.filterwarnings("ignore:xFormers is available")
def test_if_feature_extraction_crashes(
    *, tmp_path: Path, extractor: ExtractorName
) -> None:
    """
    Test if the feature extraction process crashes for a given extractor.

    This test downloads an example slide file, sets up a temporary directory,
    and attempts to extract features using the specified extractor. If the
    necessary dependencies for the extractor are not installed or if access
    to a gated repository is required, the test will be skipped.

    Args:
        extractor (ExtractorName): The name of the extractor to use for feature extraction.

    Raises:
        AssertionError: If the extracted features file is empty.
    """
    if extractor == ExtractorName.DINO_BLOOM and not torch.cuda.is_available():
        pytest.skip(
            "Skipping test for ExtractorName.DINO_BLOOM as CUDA is not available"
        )

    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "slide.svs").symlink_to(example_slide_path)

    try:
        extract_(
            wsi_dir=wsi_dir,
            output_dir=tmp_path / "output",
            extractor=extractor,
            cache_dir=None,
            cache_tiles_ext="png",
            tile_size_px=TilePixels(224),
            tile_size_um=Microns(256.0),
            max_workers=min(os.cpu_count() or 1, 16),
            brightness_cutoff=224,
            canny_cutoff=0.02,
            device="cuda" if torch.cuda.is_available() else "cpu",
            default_slide_mpp=None,
            generate_hash=True,
        )
    except ModuleNotFoundError:
        pytest.skip(f"dependencies for {extractor} not installed")
    except GatedRepoError:
        pytest.skip(f"cannot access gated repo for {extractor}")

    # Check if the file has any contents
    with h5py.File(next((tmp_path / "output").glob("*/*.h5"))) as h5_file:
        just_extracted_feats = np.array(h5_file["feats"][:])  # pyright: ignore[reportIndexIssue]

    assert len(just_extracted_feats) > 0


@pytest.mark.slow
def test_backward_compatability(tmp_path: Path) -> None:
    """
    Test the backward compatibility of feature extraction.

    This test downloads a sample slide image and a reference feature file,
    extracts features from the slide using the specified extractor, and
    compares the extracted features with the reference features to ensure
    they match.

    Raises:
        pytest.skip: If the dependencies for the extractor are not installed
            or if the gated repository for the extractor cannot be accessed.

    Asserts:
        torch.allclose: Asserts that the coordinates and features extracted
            from the slide match the reference coordinates and features within
            a tolerance.
    """
    extractor = ExtractorName.CTRANSPATH

    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "slide.svs").symlink_to(example_slide_path)

    try:
        extract_(
            wsi_dir=wsi_dir,
            output_dir=tmp_path / "output",
            extractor=extractor,
            cache_dir=None,
            cache_tiles_ext="png",
            tile_size_px=TilePixels(224),
            tile_size_um=Microns(256.0),
            max_workers=min(os.cpu_count() or 1, 16),
            brightness_cutoff=224,
            canny_cutoff=0.02,
            device="cuda" if torch.cuda.is_available() else "cpu",
            default_slide_mpp=None,
            generate_hash=True,
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
        reference_coords = torch.from_numpy(h5_file["coords"][:])  # pyright: ignore[reportIndexIssue]
        # Order coords / feats lexicographically by coords
        reference_order = np.lexsort((reference_coords[:, 1], reference_coords[:, 0]))
        reference_coords = reference_coords[reference_order]
        reference_feats = torch.from_numpy(h5_file["feats"][:])[reference_order]  # pyright: ignore[reportIndexIssue]
        reference_version = h5_file.attrs["stamp_version"]

    with h5py.File(next((tmp_path / "output").glob("*/*.h5"))) as h5_file:
        just_extracted_coords = torch.from_numpy(h5_file["coords"][:])  # pyright: ignore[reportIndexIssue]
        # Order coords / feats lexicographically by coords
        just_extracted_order = np.lexsort(
            (just_extracted_coords[:, 1], just_extracted_coords[:, 0])
        )
        just_extracted_coords = just_extracted_coords[just_extracted_order]
        just_extracted_feats = torch.from_numpy(h5_file["feats"][:])[  # pyright: ignore[reportIndexIssue]
            just_extracted_order
        ]

    assert torch.allclose(just_extracted_coords, reference_coords), (
        f"extracted {extractor} coordinates differ from those made with stamp version {reference_version}"
    )
    assert torch.allclose(just_extracted_feats, reference_feats, atol=1e-3), (
        f"extracted {extractor} features differ from those made with stamp version {reference_version}"
    )
