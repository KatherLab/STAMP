from pathlib import Path

import openslide
import torch
from PIL import Image

from stamp.heatmaps import _export_ranked_tiles
from stamp.types import TilePixels


def _dummy_slide() -> openslide.ImageSlide:
    return openslide.ImageSlide(Image.new("RGB", (64, 64), color=(123, 45, 67)))


def test_export_ranked_tiles_writes_top_and_bottom_tiles(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    _export_ranked_tiles(
        slide=_dummy_slide(),
        tiles_dir=tiles_dir,
        stem="slide",
        label="regression",
        tile_scores=torch.tensor([0.1, 0.9, 0.3]),
        coords_tile_slide_px=torch.tensor([[0, 0], [10, 10], [20, 20]]),
        tile_size_slide_px=TilePixels(8),
        topk=2,
        bottomk=2,
    )

    top_tiles = sorted(tiles_dir.glob("top_*-slide-regression=*.jpg"))
    bottom_tiles = sorted(tiles_dir.glob("bottom_*-slide-regression=*.jpg"))

    assert len(top_tiles) == 2
    assert len(bottom_tiles) == 2
    assert top_tiles[0].name.startswith("top_01-slide-regression=0.90")
    assert top_tiles[1].name.startswith("top_02-slide-regression=0.30")
    assert bottom_tiles[0].name.startswith("bottom_01-slide-regression=0.10")
    assert bottom_tiles[1].name.startswith("bottom_02-slide-regression=0.30")


def test_export_ranked_tiles_caps_requested_count(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    _export_ranked_tiles(
        slide=_dummy_slide(),
        tiles_dir=tiles_dir,
        stem="slide",
        label="survival",
        tile_scores=torch.tensor([0.4, 0.2]),
        coords_tile_slide_px=torch.tensor([[0, 0], [10, 10]]),
        tile_size_slide_px=TilePixels(8),
        topk=5,
        bottomk=5,
    )

    assert len(list(tiles_dir.glob("top_*-slide-survival=*.jpg"))) == 2
    assert len(list(tiles_dir.glob("bottom_*-slide-survival=*.jpg"))) == 2
