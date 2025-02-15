from enum import StrEnum
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.preprocessing.tiling import Microns, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class ExtractorName(StrEnum):
    CTRANSPATH = "ctranspath"
    CONCH = "mahmood-conch"
    UNI = "mahmood-uni"
    DINO_BLOOM = "dino-bloom"
    VIRCHOW2 = "virchow2"
    EMPTY = "empty"
    CONCHV1_5 = "conchv1_5"
    GIGAPATH = "gigapath"
    H_OPTIMUS_0 = "h_optimus_0"
    UNI2 = "uni2"

class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    wsi_dir: Path
    cache_dir: Path | None = None
    tile_size_um: Microns = Microns(256.0)
    tile_size_px: TilePixels = TilePixels(224)
    extractor: ExtractorName
    max_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Background rejection
    brightness_cutoff: int | None = Field(240, gt=0, lt=255)
    """Any tile brighter than this will be discarded as probable background.
    If set to `None`, the brightness-based background rejection is disabled.
    """

    canny_cutoff: float | None = Field(0.02, gt=0.0, lt=1.0)
    """Any tile with a lower ratio of pixels classified as "edges" than this
    will be rejected.
    If set to `None`, brightness-based rejection is disabled.
    """
