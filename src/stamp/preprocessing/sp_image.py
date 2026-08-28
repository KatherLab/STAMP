"""Standalone SPImage loader vendored from the official KRONOS2 repository.

SPImage loads channel-first multiplex arrays and returns edge-inclusive patches
with the official input scaling. Marker-aware z-score normalization remains the
responsibility of ``KRONOS2Model.preprocess``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tifffile import imread


def _clean_marker_name(name: str) -> str:
    """Apply KRONOS2's official marker-name formatting."""

    translation_table = str.maketrans(
        {
            "-": "_",
            " ": "_",
            ":": "_",
            "α": "a",
            "(": "_",
            ")": "_",
            "/": "",
        }
    )
    return name.lower().translate(translation_table).strip()


def _scaling_factor(dtype: np.dtype) -> float:
    """Return the official KRONOS2 SPImage dtype scaling factor."""

    if dtype == np.uint8:
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 400.0
    raise ValueError(f"no scaling_factor for dtype {dtype!r}")


def _grid_starts(extent: int, patch_size: int, step: int) -> list[int]:
    """Return edge-inclusive patch origins along one image axis."""

    starts = list(range(0, extent - patch_size + 1, step))
    if not starts or starts[-1] != extent - patch_size:
        starts.append(extent - patch_size)
    return starts


class SPImage:
    """Load a multiplex image and construct official KRONOS2 input patches."""

    def __init__(
        self,
        image: str | os.PathLike[str] | np.ndarray,
        *,
        markers: Sequence[str],
        mpp: float,
    ) -> None:
        array = (
            imread(image)
            if isinstance(image, (str, os.PathLike, Path))
            else np.asarray(image)
        )
        if array.ndim == 4:
            array = array.reshape(-1, array.shape[-2], array.shape[-1])
        if array.ndim != 3 or array.shape[0] != len(markers):
            channel_count = array.shape[0] if array.ndim == 3 else "?"
            raise ValueError(
                f"image {array.shape} has {channel_count} channels but "
                f"{len(markers)} marker names were given"
            )
        self.img = array
        self.mpp = mpp
        self.markers = [_clean_marker_name(marker) for marker in markers]

    def to_patches(
        self,
        *,
        patch_size: int = 256,
        marker_subset: Sequence[str] | None = None,
        overlap: int = 0,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return ``(patches, marker_names, coords)`` in KRONOS2 input format."""

        if overlap < 0 or overlap >= patch_size:
            raise ValueError("overlap must be non-negative and smaller than patch_size")
        if marker_subset is None:
            selected = np.arange(len(self.markers))
        else:
            wanted = {_clean_marker_name(marker) for marker in marker_subset}
            selected = np.asarray(
                [index for index, marker in enumerate(self.markers) if marker in wanted]
            )
        marker_names = [self.markers[index] for index in selected]
        scaling = _scaling_factor(self.img.dtype)

        _, height, width = self.img.shape
        x_starts = _grid_starts(width, patch_size, patch_size - overlap)
        y_starts = _grid_starts(height, patch_size, patch_size - overlap)
        coords = [(x, y) for y in y_starts for x in x_starts]
        patches = np.empty(
            (len(coords), len(selected), patch_size, patch_size), dtype=np.float32
        )
        for index, (x, y) in enumerate(coords):
            patches[index] = (
                self.img[selected, y : y + patch_size, x : x + patch_size].astype(
                    np.float32
                )
                / scaling
            )
        return patches, marker_names, np.asarray(coords, dtype=int)
