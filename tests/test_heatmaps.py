from pathlib import Path

import pytest
import torch

from stamp.cache import download_file
from stamp.heatmaps import heatmaps_


@pytest.mark.filterwarnings("ignore:There is a performance drop")
def test_heatmap_integration(tmp_path: Path) -> None:
    example_checkpoint_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.2.0/example-model-v2_3_0.ckpt",
        file_name="example-modelv2_3_0.ckpt",
        sha256sum="eb6225fcdea7f33dee80fd5dc4e7a0da6cd0d91a758e3ee9605d6869b30ab657",
    )
    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    example_feature_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.2.0/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.h5",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.h5",
        sha256sum="c66a63a289bd36d9fd3bdca9226830d0cba59fa1f9791adf60eef39f9c40c49a",
    )

    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "slide.svs").symlink_to(example_slide_path)
    feature_dir = tmp_path / "feats"
    feature_dir.mkdir()
    (feature_dir / "slide.h5").symlink_to(example_feature_path)

    heatmaps_(
        feature_dir=feature_dir,
        wsi_dir=wsi_dir,
        checkpoint_path=example_checkpoint_path,
        output_dir=tmp_path / "output",
        slide_paths=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        topk=2,
        bottomk=2,
        default_slide_mpp=None,
        opacity=0.6,
    )

    assert (tmp_path / "output" / "slide" / "plots" / "overview-slide.png").is_file()
    assert (tmp_path / "output" / "slide" / "raw" / "thumbnail-slide.png").is_file()
    assert (tmp_path / "output" / "slide" / "raw").glob("slide-MSIH=*.png")
    assert any((tmp_path / "output" / "slide" / "raw").glob("slide-nonMSIH=*.png"))
    assert (
        len(
            list(
                (tmp_path / "output" / "slide" / "tiles").glob(
                    "top_*-slide-nonMSIH=*.jpg"
                )
            )
        )
        == 2
    )
    assert (
        len(
            list(
                (tmp_path / "output" / "slide" / "tiles").glob(
                    "bottom_*-slide-nonMSIH=*.jpg"
                )
            )
        )
        == 2
    )
