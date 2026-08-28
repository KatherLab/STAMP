from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tifffile import TiffFile
from torch import Tensor
from tqdm import tqdm

import stamp
from stamp.preprocessing.config import ExtractorName, MultiplexMarkerConfig
from stamp.preprocessing.extractor import (
    Extractor,
    MultiplexExtractor,
    MultiplexFeatures,
)
from stamp.types import DeviceLikeType, TilePixels
from stamp.utils.cache import get_processing_code_hash

__author__ = "Marko van Treeck, OpenAI Codex"
__copyright__ = "Copyright (C) 2022-2026 Marko van Treeck"
__license__ = "MIT"


_logger = logging.getLogger("stamp")

_SUPPORTED_MULTIPLEX_EXTENSIONS = {".qptiff", ".tiff", ".tif"}
_MULTIPLEX_BATCH_SIZE = 64
_PACKAGED_MARKER_METADATA_CSV = Path(__file__).with_name("marker_metadata.csv")


def extract_multiplex_(
    *,
    wsi_dir: Path,
    output_dir: Path,
    wsi_list: Path | None,
    extractor: Extractor | MultiplexExtractor,
    tile_size_px: TilePixels,
    max_workers: int,
    device: DeviceLikeType,
    marker_configs: Sequence[MultiplexMarkerConfig],
    marker_metadata_csv: Path | None,
    generate_hash: bool,
    process_step: str | None = None,
    slide_start: int | None = None,
    slide_end: int | None = None,
) -> None:
    wsi_dir = wsi_dir.resolve()
    is_kronos2 = extractor.identifier == ExtractorName.KRONOS2
    if isinstance(extractor, MultiplexExtractor) and extractor.preprocess is not None:
        # Marker-aware models (currently KRONOS2) own their normalization tables.
        # Do not silently combine those statistics with STAMP's generic table.
        marker_configs = list(marker_configs)
        marker_metadata_source = None
    else:
        marker_configs, marker_metadata_source = _autofill_marker_statistics(
            marker_configs=marker_configs,
            marker_metadata_csv=marker_metadata_csv,
        )
    marker_names = [marker.name for marker in marker_configs]
    model_marker_names = marker_names
    marker_normalization = _marker_normalization(marker_configs)

    model = extractor.model.to(device).eval()
    if is_kronos2 and marker_metadata_csv is not None:
        from stamp.preprocessing.extractor.kronos2 import register_additional_markers

        register_additional_markers(model, marker_metadata_csv)

    code_hash = get_processing_code_hash(Path(__file__))[:8]
    extractor_id = extractor.identifier

    feat_output_dir = (
        output_dir / process_step
        if process_step is not None
        else (
            output_dir / f"{extractor_id}-{code_hash}"
            if generate_hash
            else output_dir / extractor_id
        )
    )
    feat_output_dir.mkdir(parents=True, exist_ok=True)

    if wsi_list is not None:
        slide_paths = [
            (wsi_dir / slide_path).resolve()
            for slide_path in _get_slide_paths(wsi_list)
        ]
    else:
        slide_paths = _discover_slide_files(wsi_dir)

    if slide_start is None and slide_end is None:
        rng = np.random.default_rng()
        perm = rng.permutation(len(slide_paths))
        slide_paths = [slide_paths[i] for i in perm]
    else:
        slide_paths = slide_paths[slice(slide_start, slide_end)]

    patch_size = int(tile_size_px)
    if patch_size <= 0:
        raise ValueError("tile_size_px must be positive for multiplex preprocessing")

    if marker_normalization is not None and isinstance(extractor, Extractor):
        _logger.info(
            "Loaded multiplex marker mean/std metadata, but the selected extractor "
            "uses per-marker RGB tiles. Marker normalization is only applied for "
            "MultiplexExtractor implementations."
        )
    elif isinstance(extractor, MultiplexExtractor) and extractor.preprocess is not None:
        _logger.info(
            "The selected multiplex extractor applies its own marker-aware "
            "normalization; STAMP marker mean/std values are not used."
        )

    for slide_path in (progress := tqdm(slide_paths)):
        progress.set_description(str(slide_path.relative_to(wsi_dir)))

        feature_output_path = feat_output_dir / f"{_stem(slide_path)}.h5"
        if feature_output_path.exists():
            _logger.debug(
                f"skipping {slide_path} because {feature_output_path} already exists"
            )
            continue

        feature_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            slide = _read_slide(slide_path, required_channels=len(marker_configs))
            if is_kronos2:
                from stamp.preprocessing.sp_image import SPImage

                patches, model_marker_names, coords = SPImage(
                    slide,
                    markers=marker_names,
                    mpp=1.0,
                ).to_patches(
                    patch_size=patch_size,
                    marker_subset=marker_names,
                )
        except Exception:
            _logger.exception(f"error while reading multiplex slide {slide_path}")
            continue

        tmp_h5_file = None
        try:
            with (
                NamedTemporaryFile(dir=feat_output_dir, delete=False) as tmp_h5_file,
                h5py.File(tmp_h5_file, "w") as h5_fp,
            ):
                h5_fp.attrs["stamp_version"] = stamp.__version__
                h5_fp.attrs["extractor"] = str(extractor.identifier)
                h5_fp.attrs["feat_type"] = "tile"
                h5_fp.attrs["unit"] = "px"
                h5_fp.attrs["tile_size_px"] = patch_size
                h5_fp.attrs["marker_names"] = np.asarray(marker_names, dtype=object)
                h5_fp.attrs["marker_means"] = np.asarray(
                    [
                        np.nan if marker.mean is None else marker.mean
                        for marker in marker_configs
                    ],
                    dtype=np.float32,
                )
                h5_fp.attrs["marker_stds"] = np.asarray(
                    [
                        np.nan if marker.std is None else marker.std
                        for marker in marker_configs
                    ],
                    dtype=np.float32,
                )
                if marker_metadata_source is not None:
                    h5_fp.attrs["marker_metadata_csv"] = str(marker_metadata_source)

                datasets: _Datasets | None = None
                patches_written = 0
                if is_kronos2:
                    patch_batches = _iter_array_patch_batches(
                        patches, coords, batch_size=_MULTIPLEX_BATCH_SIZE
                    )
                else:
                    patch_batches = _iter_patch_batches(
                        slide, patch_size=patch_size, batch_size=_MULTIPLEX_BATCH_SIZE
                    )
                for patch_batch, xs, ys in patch_batches:
                    outputs = _encode_batch(
                        extractor=extractor,
                        model=model,
                        patch_batch=patch_batch,
                        marker_normalization=marker_normalization,
                        marker_names=model_marker_names,
                    )
                    outputs = _normalize_outputs(outputs)

                    if datasets is None:
                        datasets = _create_datasets(h5_fp, outputs)

                    _append_batch(
                        datasets=datasets,
                        outputs=outputs,
                        xs=xs,
                        ys=ys,
                    )
                    patches_written += len(xs)

                if datasets is None or patches_written == 0:
                    Path(tmp_h5_file.name).unlink(missing_ok=True)
                    _logger.info(f"no tiles found in {slide_path}, skipping")
                    continue

                Path(tmp_h5_file.name).rename(feature_output_path)
        except Exception:
            _logger.exception(
                f"error while extracting multiplex features from {slide_path}"
            )
            if tmp_h5_file is not None:
                Path(tmp_h5_file.name).unlink(missing_ok=True)
            continue


class _Datasets(dict[str, h5py.Dataset]):
    pass


def _create_datasets(h5_fp: h5py.File, outputs: MultiplexFeatures) -> _Datasets:
    feat_dim = int(outputs.feats.shape[1])

    feats_ds = h5_fp.create_dataset(
        "feats",
        shape=(0, feat_dim),
        maxshape=(None, feat_dim),
        dtype="f4",
    )
    h5_fp["patch_embeddings"] = feats_ds

    datasets = _Datasets(
        feats=feats_ds,
        coord_x=h5_fp.create_dataset(
            "coord_x",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
        ),
        coord_y=h5_fp.create_dataset(
            "coord_y",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
        ),
    )
    if outputs.marker_embeddings is not None:
        n_markers = int(outputs.marker_embeddings.shape[1])
        marker_feat_dim = int(outputs.marker_embeddings.shape[2])
        datasets["marker_embeddings"] = h5_fp.create_dataset(
            "marker_embeddings",
            shape=(0, n_markers, marker_feat_dim),
            maxshape=(None, n_markers, marker_feat_dim),
            dtype="f4",
        )
    if outputs.token_embeddings is not None:
        n_markers = int(outputs.token_embeddings.shape[1])
        token_y = int(outputs.token_embeddings.shape[2])
        token_x = int(outputs.token_embeddings.shape[3])
        token_feat_dim = int(outputs.token_embeddings.shape[4])
        datasets["token_embeddings"] = h5_fp.create_dataset(
            "token_embeddings",
            shape=(0, n_markers, token_y, token_x, token_feat_dim),
            maxshape=(None, n_markers, token_y, token_x, token_feat_dim),
            dtype="f4",
        )
    return datasets


def _append_batch(
    *,
    datasets: _Datasets,
    outputs: MultiplexFeatures,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    batch_size = int(outputs.feats.shape[0])
    start = int(datasets["feats"].shape[0])
    stop = start + batch_size

    for dataset in datasets.values():
        dataset.resize(stop, axis=0)

    datasets["feats"][start:stop] = (
        outputs.feats.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )
    if outputs.marker_embeddings is not None:
        datasets["marker_embeddings"][start:stop] = (
            outputs.marker_embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    if outputs.token_embeddings is not None:
        datasets["token_embeddings"][start:stop] = (
            outputs.token_embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    datasets["coord_x"][start:stop] = xs.astype(np.int32, copy=False)
    datasets["coord_y"][start:stop] = ys.astype(np.int32, copy=False)


def _normalize_outputs(outputs: MultiplexFeatures) -> MultiplexFeatures:
    feats = outputs.feats
    marker_embeddings = outputs.marker_embeddings
    token_embeddings = outputs.token_embeddings

    if feats.ndim != 2:
        raise ValueError(
            f"expected feats to have shape (batch, feature), got {tuple(feats.shape)}"
        )
    if marker_embeddings is not None and marker_embeddings.ndim != 3:
        raise ValueError(
            "expected marker_embeddings to have shape "
            f"(batch, marker, feature), got {tuple(marker_embeddings.shape)}"
        )
    if token_embeddings is not None and token_embeddings.ndim == 3:
        token_embeddings = token_embeddings.unsqueeze(2).unsqueeze(3)
    if token_embeddings is not None and token_embeddings.ndim != 5:
        raise ValueError(
            "expected token_embeddings to have shape "
            f"(batch, marker, token_y, token_x, feature), got "
            f"{tuple(token_embeddings.shape)}"
        )

    return MultiplexFeatures(
        feats=feats,
        marker_embeddings=marker_embeddings,
        token_embeddings=token_embeddings,
    )


def _encode_batch(
    *,
    extractor: Extractor | MultiplexExtractor,
    model: torch.nn.Module,
    patch_batch: Tensor,
    marker_normalization: tuple[Tensor, Tensor] | None,
    marker_names: Sequence[str],
) -> MultiplexFeatures:
    model_device = _model_device(model)

    with torch.inference_mode():
        if isinstance(extractor, MultiplexExtractor):
            if extractor.preprocess is not None:
                transformed = extractor.preprocess(model, patch_batch, marker_names)
            elif marker_normalization is not None:
                means, stds = marker_normalization
                patch_batch = (patch_batch.float() - means) / stds
                transformed = patch_batch
            else:
                patch_batch = patch_batch.float()
                transformed = patch_batch
            if extractor.preprocess is None and extractor.transform is not None:
                transformed = torch.stack(
                    [extractor.transform(patch) for patch in transformed]
                )
            if extractor.forward_with_markers is not None:
                return extractor.forward_with_markers(
                    model, transformed.to(model_device), marker_names
                )
            return extractor.forward(model, transformed.to(model_device))

        marker_embeddings = []
        for marker_index in range(patch_batch.shape[1]):
            rgb_batch = torch.stack(
                [
                    extractor.transform(_channel_to_rgb_image(patch[marker_index]))
                    for patch in patch_batch
                ]
            )
            marker_embeddings.append(model(rgb_batch.to(model_device)).detach())

        stacked = torch.stack(marker_embeddings, dim=1)
        pooled = stacked.mean(dim=1)
        return MultiplexFeatures(
            feats=pooled,
            marker_embeddings=stacked,
            token_embeddings=stacked.unsqueeze(2).unsqueeze(3),
        )


def _channel_to_rgb_image(channel: Tensor) -> Image.Image:
    return Image.fromarray(_to_uint8(channel.detach().cpu().numpy()), mode="L").convert(
        "RGB"
    )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        max_value = max(1, np.iinfo(array.dtype).max)
        if max_value == 255:
            return array.astype(np.uint8, copy=False)
        return (
            np.rint(array.astype(np.float32) / max_value * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )

    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        return array.astype(np.uint8)
    if 0.0 <= float(array.min()) and float(array.max()) <= 1.0:
        return np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
    return np.rint(array).clip(0, 255).astype(np.uint8)


def _iter_patch_batches(
    slide: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
) -> Iterator[tuple[Tensor, np.ndarray, np.ndarray]]:
    _, height, width = slide.shape

    patches: list[Tensor] = []
    xs: list[int] = []
    ys: list[int] = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = torch.from_numpy(
                np.ascontiguousarray(slide[:, y : y + patch_size, x : x + patch_size])
            )
            patches.append(patch)
            xs.append(x)
            ys.append(y)

            if len(patches) == batch_size:
                yield torch.stack(patches), np.asarray(xs), np.asarray(ys)
                patches, xs, ys = [], [], []

    if patches:
        yield torch.stack(patches), np.asarray(xs), np.asarray(ys)


def _iter_array_patch_batches(
    patches: np.ndarray,
    coords: np.ndarray,
    *,
    batch_size: int,
) -> Iterator[tuple[Tensor, np.ndarray, np.ndarray]]:
    """Yield precomputed SPImage patches in STAMP's batched interface."""

    for start in range(0, len(patches), batch_size):
        stop = start + batch_size
        batch_coords = coords[start:stop]
        yield (
            torch.from_numpy(np.ascontiguousarray(patches[start:stop])),
            batch_coords[:, 0],
            batch_coords[:, 1],
        )


def _read_slide(slide_path: Path, *, required_channels: int) -> np.ndarray:
    try:
        with TiffFile(slide_path) as tif:
            slide = tif.series[0].asarray()
    except ValueError as exc:
        if "requires the 'imagecodecs' package" in str(exc):
            raise ModuleNotFoundError(
                "Reading LZW-compressed multiplex TIFF/QPTIFF files requires the "
                "'imagecodecs' package. Reinstall STAMP with imagecodecs available "
                "or run `pip install imagecodecs` in the active environment."
            ) from exc
        raise

    slide = np.asarray(slide)
    if slide.ndim == 2:
        slide = slide[np.newaxis, :, :]
    elif slide.ndim != 3:
        raise ValueError(
            f"{slide_path}: expected 2D or 3D array for multiplex slide, got {slide.shape}"
        )

    if (
        slide.shape[0] <= 64
        and slide.shape[1] > 64
        and slide.shape[2] > 64
        and slide.shape[0] >= required_channels
    ):
        return _trim_channels(
            slide,
            slide_path=slide_path,
            required_channels=required_channels,
        )

    if (
        slide.shape[-1] <= 64
        and slide.shape[0] > 64
        and slide.shape[1] > 64
        and slide.shape[-1] >= required_channels
    ):
        return _trim_channels(
            np.moveaxis(slide, -1, 0),
            slide_path=slide_path,
            required_channels=required_channels,
        )

    if slide.shape[0] >= required_channels:
        return _trim_channels(
            slide,
            slide_path=slide_path,
            required_channels=required_channels,
        )

    raise ValueError(
        f"{slide_path}: found {slide.shape} but expected at least {required_channels} channels"
    )


def _trim_channels(
    slide: np.ndarray,
    *,
    slide_path: Path,
    required_channels: int,
) -> np.ndarray:
    if slide.shape[0] == required_channels:
        return slide

    _logger.warning(
        "%s: found %d channels but preprocessing.markers defines %d; "
        "using the first %d channels in file order",
        slide_path,
        slide.shape[0],
        required_channels,
        required_channels,
    )
    return slide[:required_channels]


def _autofill_marker_statistics(
    *,
    marker_configs: Sequence[MultiplexMarkerConfig],
    marker_metadata_csv: Path | None,
) -> tuple[list[MultiplexMarkerConfig], Path | None]:
    marker_stats, metadata_source = _load_marker_metadata_stats(marker_metadata_csv)
    resolved_marker_configs: list[MultiplexMarkerConfig] = []

    for marker in marker_configs:
        mean = marker.mean
        std = marker.std

        if (mean is None or std is None) and marker_stats:
            if csv_stats := marker_stats.get(marker.name.strip().lower()):
                csv_mean, csv_std = csv_stats
                mean = mean if mean is not None else csv_mean
                std = std if std is not None else csv_std

        if mean is None or std is None:
            _logger.info(
                "No complete mean/std metadata found for multiplex marker %s; "
                "leaving missing values unset.",
                marker.name,
            )

        resolved_marker_configs.append(
            marker.model_copy(update={"mean": mean, "std": std})
        )

    return resolved_marker_configs, metadata_source


def _load_marker_metadata_stats(
    marker_metadata_csv: Path | None,
) -> tuple[dict[str, tuple[float, float]], Path | None]:
    for csv_path in _candidate_marker_metadata_csv_paths(marker_metadata_csv):
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            _logger.exception("Failed to read marker metadata CSV at %s", csv_path)
            continue

        required_columns = {"marker_name", "marker_mean", "marker_std"}
        if not required_columns.issubset(df.columns):
            _logger.warning(
                "Skipping marker metadata CSV %s because it is missing required "
                "columns %s",
                csv_path,
                sorted(required_columns),
            )
            continue

        stats: dict[str, tuple[float, float]] = {}
        for row in df.itertuples(index=False):
            marker_name = str(getattr(row, "marker_name", "")).strip().lower()
            if not marker_name:
                continue
            try:
                marker_mean = float(getattr(row, "marker_mean"))
                marker_std = float(getattr(row, "marker_std"))
            except (TypeError, ValueError):
                continue
            if marker_std <= 0.0:
                continue
            stats[marker_name] = (marker_mean, marker_std)

        if stats:
            _logger.info(
                "Loaded multiplex marker metadata for %d markers from %s",
                len(stats),
                csv_path,
            )
            return stats, csv_path

    return {}, None


def _candidate_marker_metadata_csv_paths(
    marker_metadata_csv: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if marker_metadata_csv is not None:
        candidates.append(marker_metadata_csv.expanduser())
    candidates.append(_PACKAGED_MARKER_METADATA_CSV)

    unique_candidates: list[Path] = []
    seen_candidates: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_candidates:
            continue
        seen_candidates.add(resolved_candidate)
        unique_candidates.append(resolved_candidate)

    return unique_candidates


def _marker_normalization(
    marker_configs: Sequence[MultiplexMarkerConfig],
) -> tuple[Tensor, Tensor] | None:
    if not any(
        marker.mean is not None or marker.std is not None for marker in marker_configs
    ):
        return None

    means = torch.tensor(
        [0.0 if marker.mean is None else marker.mean for marker in marker_configs],
        dtype=torch.float32,
    )[:, None, None]
    stds = torch.tensor(
        [1.0 if marker.std is None else marker.std for marker in marker_configs],
        dtype=torch.float32,
    )[:, None, None]
    return means, stds


def _stem(path: Path) -> str:
    name = path.name
    name_lower = name.lower()
    for ext in _SUPPORTED_MULTIPLEX_EXTENSIONS:
        if name_lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _discover_slide_files(root: Path) -> list[Path]:
    ext_priority = {".qptiff": 0, ".tiff": 1, ".tif": 2}
    chosen: dict[str, tuple[int, Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue

        suffix = path.suffix.lower()
        if suffix not in ext_priority:
            continue

        stem = _stem(path)
        priority = ext_priority[suffix]
        previous = chosen.get(stem)
        if previous is None or priority < previous[0]:
            chosen[stem] = (priority, path)

    return sorted(item[1] for item in chosen.values())


def _get_slide_paths(wsi_list: Path) -> Iterable[str]:
    suffix = wsi_list.suffix.lower()
    if suffix == ".txt":
        with open(wsi_list) as file_handle:
            return [line.strip() for line in file_handle if line.strip()]
    if suffix == ".csv":
        df = pd.read_csv(wsi_list, header=None)
        return df.iloc[:, 0].astype(str).tolist()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(wsi_list, header=None)
        return df.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"Unsupported file type: {suffix}")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")
