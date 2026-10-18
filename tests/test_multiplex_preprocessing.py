from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from tifffile import imwrite

from stamp.preprocessing import extract_
from stamp.preprocessing.config import (
    MultiplexMarkerConfig,
    PreprocessingMode,
)
from stamp.preprocessing.extractor import (
    Extractor,
)
from stamp.preprocessing.multiplex import _read_slide
from stamp.types import Microns, TilePixels


class _ToyExtractorModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        flat = batch.view(batch.shape[0], -1)
        return torch.stack((flat.mean(dim=1), flat.amax(dim=1)), dim=1)


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)


@pytest.mark.parametrize("fallback_mpp", [None, 0.5])
def test_multiplex_preprocess_writes_notebook_style_h5(
    tmp_path: Path,
    fallback_mpp: float | None,
    monkeypatch,
) -> None:
    import openslide

    def unsupported_slide(_path):
        raise openslide.OpenSlideUnsupportedFormatError("test image")

    monkeypatch.setattr(
        "stamp.preprocessing.multiplex.openslide.open_slide", unsupported_slide
    )
    slide = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[40, 41, 42, 43], [44, 45, 46, 47], [48, 49, 50, 51], [52, 53, 54, 55]],
        ],
        dtype=np.uint8,
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    imwrite(wsi_dir / "slide.qptiff", slide)

    extractor = Extractor(
        model=_ToyExtractorModel(),
        transform=_pil_rgb_to_tensor,
        identifier="toy-multiplex",
    )

    extract_(
        wsi_dir=wsi_dir,
        output_dir=tmp_path / "output",
        wsi_list=None,
        cache_dir=None,
        cache_tiles_ext="png",
        extractor=extractor,
        tile_size_px=TilePixels(2),
        tile_size_um=Microns(256.0),
        max_workers=1,
        device="cpu",
        default_slide_mpp=fallback_mpp,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[
            MultiplexMarkerConfig(name="DAPI"),
            MultiplexMarkerConfig(name="HER2"),
        ],
    )

    h5_path = next((tmp_path / "output").glob("*/*.h5"))
    with h5py.File(h5_path, "r") as h5:
        assert set(h5.keys()) == {
            "coord_x",
            "coord_y",
            "feats",
            "marker_embeddings",
            "patch_embeddings",
            "token_embeddings",
        }
        assert h5["feats"].shape == (4, 2)
        assert h5["patch_embeddings"].shape == (4, 2)
        assert h5["marker_embeddings"].shape == (4, 2, 2)
        assert h5["token_embeddings"].shape == (4, 2, 1, 1, 2)
        assert np.array_equal(h5["coord_x"][:], np.array([0, 2, 0, 2], dtype=np.int32))
        assert np.array_equal(h5["coord_y"][:], np.array([0, 0, 2, 2], dtype=np.int32))
        assert h5.attrs["tile_size_px"] == 2
        if fallback_mpp is None:
            assert "source_mpp" not in h5.attrs
        else:
            assert h5.attrs["source_mpp"] == fallback_mpp
        assert list(h5.attrs["marker_names"]) == ["DAPI", "HER2"]
        assert np.allclose(h5["feats"][:], h5["patch_embeddings"][:])


def test_spimage_uses_official_float_scaling_and_edge_grid() -> None:
    from stamp.preprocessing.sp_image import SPImage

    image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    patches, marker_names, coords = SPImage(
        image, markers=["CD-8"], mpp=1.0
    ).to_patches(patch_size=2)

    assert marker_names == ["cd_8"]
    assert np.array_equal(coords, np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    assert patches.shape == (4, 1, 2, 2)
    assert np.isclose(patches[0, 0, 1, 1], 4.0 / 400.0)


def test_read_slide_trims_extra_channels(tmp_path: Path) -> None:
    slide = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    slide_path = tmp_path / "slide.qptiff"
    imwrite(slide_path, slide)

    trimmed = _read_slide(slide_path, required_channels=3)

    assert trimmed.shape == (3, 2, 2)
    assert np.array_equal(trimmed, slide[:3])
