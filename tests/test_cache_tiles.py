import os
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from stamp.preprocessing import Microns, TilePixels
from stamp.preprocessing.tiling import _Tile, tiles_with_cache
from stamp.types import ImageExtension, SlidePixels
from stamp.utils.cache import download_file


def _get_tiles_and_images(
    iterator: Iterable[_Tile[Microns]],
) -> tuple[list[_Tile[Microns]], list[np.ndarray]]:
    tiles = []
    images = []
    for tile in iterator:
        tiles.append(tile)
        images.append(np.array(tile.image))
    return tiles, images


@pytest.mark.slow
@pytest.mark.parametrize("cache_ext", ["jpg", "png"])
def test_tile_caching(*, tmp_path: Path, cache_ext: ImageExtension) -> None:
    """
    Test if tile caching works correctly for different cache extensions.

    This test downloads an example slide file, extracts tiles with caching enabled,
    and verifies that:
    1. The cache file is created with correct parameters
    2. The tiles can be read back from the cache
    3. The tiles from cache match the original tiles

    Args:
        tmp_path: Temporary directory for the test
        cache_ext: The image extension to use for caching (jpg or png)
    """
    example_slide_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0.dev14/TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        file_name="TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs",
        sha256sum="9b7d2b0294524351bf29229c656cc886af028cb9e7463882289fac43c1347525",
    )

    # First extract tiles with caching
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Extract tiles first time (creates cache)
    tiles_first_pass, images_first_pass = _get_tiles_and_images(
        tiles_with_cache(
            slide_path=example_slide_path,
            cache_dir=cache_dir,
            cache_tiles_ext=cache_ext,
            tile_size_um=Microns(256.0),
            tile_size_px=TilePixels(224),
            max_supertile_size_slide_px=SlidePixels(4096),
            max_workers=min(os.cpu_count() or 1, 16),
            brightness_cutoff=224,
            canny_cutoff=0.02,
            default_slide_mpp=None,
        )
    )

    # Verify cache was created
    cache_files = list(cache_dir.glob("*.zip"))
    assert len(cache_files) == 1, "Expected exactly one cache file"
    cache_file = cache_files[0]

    # Check that the cache file contains images with the correct extension
    with ZipFile(cache_file, "r") as zip_file:
        assert any(name.endswith(f".{cache_ext}") for name in zip_file.namelist()), (
            "Cache file should contain images with the correct extension"
        )

    # Extract tiles second time (should use cache)
    tiles_second_pass, images_second_pass = _get_tiles_and_images(
        tiles_with_cache(
            slide_path=example_slide_path,
            cache_dir=cache_dir,
            cache_tiles_ext=cache_ext,
            tile_size_um=Microns(256.0),
            tile_size_px=TilePixels(224),
            max_supertile_size_slide_px=SlidePixels(4096),
            max_workers=min(os.cpu_count() or 1, 16),
            brightness_cutoff=224,
            canny_cutoff=0.02,
            default_slide_mpp=None,
        )
    )

    # Verify we got the same number of tiles
    assert len(tiles_first_pass) == len(tiles_second_pass), (
        "Number of tiles should match"
    )

    # Compare tiles from both passes
    for tile_first, tile_second in zip(tiles_first_pass, tiles_second_pass):
        # Compare coordinates
        assert tile_first.coordinates.x == tile_second.coordinates.x, (
            "X coordinates should match"
        )
        assert tile_first.coordinates.y == tile_second.coordinates.y, (
            "Y coordinates should match"
        )
        assert tile_first.size == tile_second.size, "Tile sizes should match"

    # Compare images
    if cache_ext == "png":  # don't even bother comparing JPG (lossy compression)
        for img1, img2 in zip(images_first_pass, images_second_pass):
            np.testing.assert_array_equal(img1, img2, "Images should be identical")
