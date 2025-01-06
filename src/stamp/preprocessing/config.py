from pathlib import Path
from typing import Literal

import torch
from pydantic import AliasChoices, BaseModel, Field
from torch._prims_common import DeviceLikeType

from stamp.preprocessing.tiling import Microns, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    output_dir: Path
    wsi_dir: Path
    cache_dir: Path | None = None
    tile_size_um: Microns = Field(
        Microns(256.0), validation_alias=AliasChoices("tile_size_um", "microns")
    )
    tile_size_px: TilePixels = TilePixels(224)
    extractor: Literal[
        "ctranspath", "mahmood-uni", "mahmood-conch", "dino-bloom", "virchow2", "empty"
    ]
    max_workers: int = Field(8, validation_alias=AliasChoices("max_workers", "cores"))
    accelerator: DeviceLikeType = "cuda" if torch.cuda.is_available() else "cpu"
    brightness_cutoff: int | None = Field(240, ge=0, le=255)
