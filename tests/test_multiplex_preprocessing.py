import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from tifffile import imwrite

from stamp.preprocessing import extract_
from stamp.preprocessing.multiplex import _read_slide
from stamp.preprocessing.config import (
    ExtractorName,
    MultiplexMarkerConfig,
    PreprocessingMode,
)
from stamp.preprocessing.extractor import (
    Extractor,
    MultiplexExtractor,
    MultiplexFeatures,
)
from stamp.types import Microns, TilePixels


class _ToyExtractorModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        flat = batch.view(batch.shape[0], -1)
        return torch.stack((flat.mean(dim=1), flat.amax(dim=1)), dim=1)


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)


class _ToyMultiplexModel(torch.nn.Module):
    pass


def _toy_multiplex_forward(
    _model: _ToyMultiplexModel,
    batch: torch.Tensor,
) -> MultiplexFeatures:
    marker_means = batch.mean(dim=(-2, -1), keepdim=False)
    marker_embeddings = marker_means.unsqueeze(-1)
    feats = marker_embeddings.mean(dim=1)
    token_embeddings = marker_embeddings.unsqueeze(2).unsqueeze(3)
    return MultiplexFeatures(
        feats=feats,
        marker_embeddings=marker_embeddings,
        token_embeddings=token_embeddings,
    )


def test_multiplex_preprocess_writes_notebook_style_h5(tmp_path: Path) -> None:
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
        default_slide_mpp=None,
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
        assert list(h5.attrs["marker_names"]) == ["DAPI", "HER2"]
        assert np.allclose(h5["feats"][:], h5["patch_embeddings"][:])


def test_multiplex_preprocess_autofills_marker_stats_from_bundled_csv(
    tmp_path: Path,
) -> None:
    dapi_mean = 0.083207167266239
    dapi_std = 0.095881901595564
    foxp3_mean = 0.014452395933506
    foxp3_std = 0.038268101524066

    slide = np.array(
        [
            np.full((2, 2), dapi_mean + 2 * dapi_std, dtype=np.float32),
            np.full((2, 2), foxp3_mean + foxp3_std, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    imwrite(wsi_dir / "slide.qptiff", slide)

    extractor = MultiplexExtractor(
        model=_ToyMultiplexModel(),
        identifier="toy-native-multiplex",
        forward=_toy_multiplex_forward,
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
        default_slide_mpp=None,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[
            MultiplexMarkerConfig(name="DAPI"),
            MultiplexMarkerConfig(name="FOXP3"),
        ],
    )

    h5_path = next((tmp_path / "output").glob("*/*.h5"))
    with h5py.File(h5_path, "r") as h5:
        assert np.allclose(h5["marker_embeddings"][0, :, 0], np.array([2.0, 1.0]))
        assert np.allclose(h5["feats"][0], np.array([1.5]))
        assert np.allclose(h5.attrs["marker_means"], np.array([dapi_mean, foxp3_mean]))
        assert np.allclose(h5.attrs["marker_stds"], np.array([dapi_std, foxp3_std]))
        assert h5.attrs["marker_metadata_csv"].endswith(
            "src/stamp/preprocessing/marker_metadata.csv"
        )


def test_multiplex_preprocess_with_kronos_extractor_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import stamp.preprocessing.extractor.kronos as kronos_module

    class _StubKronosModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor):
            batch_size, n_markers = batch.shape[:2]
            feats = torch.full(
                (batch_size, 3), 7.0, dtype=batch.dtype, device=batch.device
            )
            marker_embeddings = torch.arange(
                batch_size * n_markers * 2,
                dtype=batch.dtype,
                device=batch.device,
            ).reshape(batch_size, n_markers, 2)
            token_embeddings = marker_embeddings.unsqueeze(2).unsqueeze(3)
            return feats, marker_embeddings, token_embeddings

    def _fake_create_model_from_pretrained(**_kwargs):
        return _StubKronosModel(), torch.float32, 3

    monkeypatch.setattr(
        kronos_module,
        "create_model_from_pretrained",
        _fake_create_model_from_pretrained,
    )

    slide = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.uint8,
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    imwrite(wsi_dir / "slide.qptiff", slide)

    extract_(
        wsi_dir=wsi_dir,
        output_dir=tmp_path / "output",
        wsi_list=None,
        cache_dir=None,
        cache_tiles_ext="png",
        extractor=ExtractorName.KRONOS,
        tile_size_px=TilePixels(2),
        tile_size_um=Microns(256.0),
        max_workers=1,
        device="cpu",
        default_slide_mpp=None,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[
            MultiplexMarkerConfig(name="DAPI"),
            MultiplexMarkerConfig(name="HER2"),
        ],
    )

    h5_path = next((tmp_path / "output").glob("kronos/*.h5"))
    with h5py.File(h5_path, "r") as h5:
        assert h5["feats"].shape == (1, 3)
        assert h5["marker_embeddings"].shape == (1, 2, 2)
        assert h5["token_embeddings"].shape == (1, 2, 1, 1, 2)
        assert np.allclose(h5["feats"][0], np.array([7.0, 7.0, 7.0], dtype=np.float32))


def test_multiplex_preprocess_with_kronos2_extractor_name_unit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import stamp.preprocessing.extractor.kronos2 as kronos2_module

    class _StubKronos2Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.preprocess_calls: list[tuple[np.ndarray, list[str], str | None]] = []
            self.forward_calls: list[list[str]] = []
            self.additional_marker_csvs: list[str] = []

        def register_additional_markers(self, csv_path: str) -> None:
            self.additional_marker_csvs.append(csv_path)

        def preprocess(
            self,
            batch: np.ndarray,
            marker_names: list[str],
            *,
            preferred_dapi: str | None,
        ) -> np.ndarray:
            self.preprocess_calls.append((batch, marker_names, preferred_dapi))
            return batch + 1.0

        def forward(self, batch: torch.Tensor, marker_names: list[str]) -> torch.Tensor:
            self.forward_calls.append(marker_names)
            return batch.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 3)

    model = _StubKronos2Model()
    monkeypatch.setattr(
        kronos2_module.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    slide = np.array(
        [
            [[0, 255], [127, 64]],
            [[1, 2], [3, 4]],
        ],
        dtype=np.uint8,
    )
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    imwrite(wsi_dir / "slide.qptiff", slide)
    marker_metadata_csv = tmp_path / "additional_markers.csv"
    marker_metadata_csv.write_text("marker_name,mean,std\nDLL3,0.01,0.02\n")

    extract_(
        wsi_dir=wsi_dir,
        output_dir=tmp_path / "output",
        wsi_list=None,
        cache_dir=None,
        cache_tiles_ext="png",
        extractor=ExtractorName.KRONOS2,
        tile_size_px=TilePixels(2),
        tile_size_um=Microns(256.0),
        max_workers=1,
        device="cpu",
        default_slide_mpp=None,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[
            MultiplexMarkerConfig(name="DAPI"),
            MultiplexMarkerConfig(name="HER2"),
        ],
        marker_metadata_csv=marker_metadata_csv,
    )

    assert model.additional_marker_csvs == [str(marker_metadata_csv)]
    assert len(model.preprocess_calls) == 1
    preprocessed, marker_names, preferred_dapi = model.preprocess_calls[0]
    assert marker_names == ["dapi", "her2"]
    assert preferred_dapi == "dapi"
    assert np.isclose(preprocessed[0, 0, 0, 1], 1.0)
    assert model.forward_calls == [["dapi", "her2"]]

    h5_path = next((tmp_path / "output").glob("kronos2/*.h5"))
    with h5py.File(h5_path, "r") as h5:
        assert set(h5.keys()) == {"coord_x", "coord_y", "feats", "patch_embeddings"}
        assert h5["feats"].shape == (1, 3)
        assert h5.attrs["extractor"] == "kronos2"


@pytest.mark.slow
def test_multiplex_preprocess_with_kronos2_extractor_name(tmp_path: Path) -> None:
    """Run KRONOS2 on its official multiplex demo image.

    This is opt-in because KRONOS2 is gated and downloading its model weights is
    too expensive for the regular unit-test suite. Before running, authenticate
    with ``hf auth login``, accept the model terms, and set
    ``STAMP_RUN_KRONOS2_INTEGRATION=1``.
    """

    if os.environ.get("STAMP_RUN_KRONOS2_INTEGRATION") != "1":
        pytest.skip("set STAMP_RUN_KRONOS2_INTEGRATION=1 to run the gated test")

    try:
        tiff_path = hf_hub_download(
            repo_id="MahmoodLab/KRONOS2",
            filename="demo_image/core.ome.tiff",
        )
        names_path = hf_hub_download(
            repo_id="MahmoodLab/KRONOS2",
            filename="demo_image/channel_names.txt",
        )
    except GatedRepoError:
        pytest.skip("the active Hugging Face account cannot access MahmoodLab/KRONOS2")

    marker_names = [
        line.strip()
        for line in Path(names_path).read_text().splitlines()
        if line.strip()
    ]
    assert marker_names, "KRONOS2 demo is missing channel names"

    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "core.ome.tiff").symlink_to(tiff_path)

    extract_(
        wsi_dir=wsi_dir,
        output_dir=tmp_path / "output",
        wsi_list=None,
        cache_dir=None,
        cache_tiles_ext="png",
        extractor=ExtractorName.KRONOS2,
        tile_size_px=TilePixels(256),
        tile_size_um=Microns(256.0),
        max_workers=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_slide_mpp=None,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[MultiplexMarkerConfig(name=name) for name in marker_names],
    )

    h5_path = next((tmp_path / "output").glob("kronos2/*.h5"))
    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["extractor"] == "kronos2"
        assert h5["feats"].shape[0] > 0
        assert h5["feats"].shape[1] == 768


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


def test_read_slide_reports_missing_imagecodecs(monkeypatch, tmp_path: Path) -> None:
    class _FakeSeries:
        def asarray(self):
            raise ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    class _FakeTiffFile:
        def __init__(self, _path: Path) -> None:
            self.series = [_FakeSeries()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("stamp.preprocessing.multiplex.TiffFile", _FakeTiffFile)

    with pytest.raises(ModuleNotFoundError, match="imagecodecs"):
        _read_slide(tmp_path / "slide.qptiff", required_channels=2)


def test_read_slide_trims_extra_channels(tmp_path: Path) -> None:
    slide = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    slide_path = tmp_path / "slide.qptiff"
    imwrite(slide_path, slide)

    trimmed = _read_slide(slide_path, required_channels=3)

    assert trimmed.shape == (3, 2, 2)
    assert np.array_equal(trimmed, slide[:3])


def test_multiplex_preprocess_respects_worker_slide_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for index in range(2):
        slide = np.full((2, 2, 2), index, dtype=np.uint8)
        slide_dir = tmp_path / "wsis"
        slide_dir.mkdir(exist_ok=True)
        imwrite(slide_dir / f"slide{index}.qptiff", slide)

    monkeypatch.setenv("STAMP_PREPROCESS_SLIDE_START", "0")
    monkeypatch.setenv("STAMP_PREPROCESS_SLIDE_END", "1")

    extractor = Extractor(
        model=_ToyExtractorModel(),
        transform=_pil_rgb_to_tensor,
        identifier="toy-multiplex",
    )

    extract_(
        wsi_dir=tmp_path / "wsis",
        output_dir=tmp_path / "output",
        wsi_list=None,
        cache_dir=None,
        cache_tiles_ext="png",
        extractor=extractor,
        tile_size_px=TilePixels(2),
        tile_size_um=Microns(256.0),
        max_workers=1,
        device="cpu",
        default_slide_mpp=None,
        brightness_cutoff=None,
        canny_cutoff=None,
        generate_hash=False,
        mode=PreprocessingMode.MULTIPLEX,
        marker_configs=[
            MultiplexMarkerConfig(name="DAPI"),
            MultiplexMarkerConfig(name="HER2"),
        ],
    )

    h5_paths = sorted((tmp_path / "output").glob("*/*.h5"))
    assert len(h5_paths) == 1
    assert h5_paths[0].stem == "slide0"
