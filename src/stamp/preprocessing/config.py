from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch._prims_common import DeviceLikeType

from stamp.preprocessing.tiling import Microns, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    output_dir: Path
    wsi_dir: Path
    cache_dir: Path | None = None
    tile_size_um: Microns = Microns(256.0)
    tile_size_px: TilePixels = TilePixels(224)
    extractor: Literal[
        "ctranspath", "mahmood-uni", "mahmood-conch", "dino-bloom", "virchow2", "empty"
    ]
    max_workers: int = 8
    accelerator: DeviceLikeType = "cuda" if torch.cuda.is_available() else "cpu"

    # Background rejection
    brightness_cutoff: int | None = Field(240, ge=0, le=255)
    """Any tile brighter than this will be discarded as probable background.
    If set to `None`, the brightness-based background rejection is disabled.
    """

    canny_cutoff: float | None = Field(0.02, ge=0, le=1)
    """Any tile with a lower ratio of pixels classified as "edges" than this
    will be rejected.
    If set to `None`, brightness-based rejection is disabled.
    """
