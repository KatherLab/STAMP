from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from torch._prims_common import DeviceLikeType

from stamp.preprocessing.tiling import Microns, TilePixels


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    output_dir: Path
    wsi_dir: Path
    cache_dir: Path | None = None
    tile_size_um: Microns = Field(
        Microns(256.0), validation_alias=AliasChoices("tile_size_um", "microns")
    )
    tile_size_px: TilePixels = TilePixels(224)
    extractor: Literal["ctranspath", "mahmood-uni", "mahmood-conch"] = Field(
        validation_alias=AliasChoices("extractor", "feat_extractor")
    )
    max_workers: int = Field(8, validation_alias=AliasChoices("max_workers", "cores"))
    device: DeviceLikeType = "cpu"
