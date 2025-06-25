from enum import StrEnum
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.types import ImageExtension, Microns, SlideMPP, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class ExtractorName(StrEnum):
    CTRANSPATH = "ctranspath"
    CHIEF_CTRANSPATH = "chief-ctranspath"
    CONCH = "mahmood-conch"
    CONCH1_5 = "mahmood-conch1_5"
    UNI = "mahmood-uni"
    UNI2 = "mahmood-uni2"
    DINO_BLOOM = "dino-bloom"
    GIGAPATH = "gigapath"
    H_OPTIMUS_0 = "h-optimus-0"
    H_OPTIMUS_1 = "h-optimus-1"
    VIRCHOW = "virchow"
    VIRCHOW_FULL = "virchow_full"
    VIRCHOW2 = "virchow2"
    MUSK = "musk"
    MSTAR = "mstar"
    PLIP = "plip"
    EMPTY = "empty"


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    wsi_dir: Path
    cache_dir: Path | None = None
    cache_tiles_ext: ImageExtension = "jpg"
    tile_size_um: Microns = Microns(256.0)
    tile_size_px: TilePixels = TilePixels(224)
    extractor: ExtractorName
    max_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    generate_hash: bool = True

    default_slide_mpp: SlideMPP | None = None
    """MPP of the slide to use if none can be inferred from the WSI"""

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
