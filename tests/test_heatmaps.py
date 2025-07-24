from pathlib import Path

import pytest
import torch

from stamp.cache import download_file
from stamp.heatmaps import heatmaps_


@pytest.mark.filterwarnings("ignore:There is a performance drop")
def test_heatmap_integration(tmp_path: Path) -> None:
    example_checkpoint_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0-dev8/example-model.ckpt",
        file_name="example-model.ckpt",
        sha256sum="a71dffd4b5fdb82acd5f84064880efd3382e200b07e5a008cb53e03197b6de56",
    )
    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )
    example_feature_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4-mahmood-uni.h5",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4-mahmood-uni.h5",
        sha256sum="13b1390241e73a3969915d3d01c5c64f1b7c68318a685d8e3bf851067162f0bc",
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
    )

    assert (tmp_path / "output" / "slide" / "plots" /  "overview-slide.png").is_file()
    assert (tmp_path / "output" / "slide" / "raw" / "thumbnail-slide.png").is_file()
    assert (tmp_path / "output" / "slide" / "raw" / "slide-MSIH=0.16.png").is_file()
    assert (tmp_path / "output" / "slide" / "raw" / "slide-nonMSIH=0.84.png").is_file()
    assert (
        len(list((tmp_path / "output" / "slide" / "tiles").glob("top-slide-*.jpg"))) == 2
    )
    assert (
        len(list((tmp_path / "output" / "slide" / "tiles").glob("bottom-slide-*.jpg"))) == 2
    )
