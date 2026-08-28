from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stamp.types import ImageExtension, Microns, SlideMPP, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class ExtractorName(StrEnum):
    KRONOS = "kronos"
    KRONOS2 = "kronos2"
    CTRANSPATH = "ctranspath"
    CHIEF_CTRANSPATH = "chief-ctranspath"
    CONCH = "conch"
    CONCH1_5 = "conch1_5"
    UNI = "uni"
    UNI2 = "uni2"
    DINO_BLOOM = "dino-bloom"
    GIGAPATH = "gigapath"
    H_OPTIMUS_0 = "h-optimus-0"
    H_OPTIMUS_1 = "h-optimus-1"
    VIRCHOW = "virchow"
    VIRCHOW_FULL = "virchow-full"
    VIRCHOW2 = "virchow2"
    MUSK = "musk"
    MSTAR = "mstar"
    PLIP = "plip"
    KEEP = "keep"
    TICON = "ticon"
    EMPTY = "empty"
    RED_DINO = "red-dino"


class PreprocessingMode(StrEnum):
    WSI = "wsi"
    MULTIPLEX = "multiplex"


class MultiplexMarkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mean: float | None = None
    std: float | None = Field(default=None, gt=0.0)


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    wsi_dir: Path
    mode: PreprocessingMode = PreprocessingMode.WSI
    wsi_list: Path | None = Field(
        default=None, description="Txt, Excel or CSV to read data filename from"
    )
    cache_dir: Path | None = None
    cache_tiles_ext: ImageExtension = "jpg"
    tile_size_um: Microns = Microns(256.0)
    tile_size_px: TilePixels = TilePixels(224)
    extractor: ExtractorName
    max_workers: int = 8
    device: str = Field(
        default_factory=lambda: (
            "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
    )
    generate_hash: bool = True
    parallel: bool = False
    process_step: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional explicit name for the feature-output subdirectory. "
            "When set, feature files are written to output_dir/process_step "
            "instead of STAMP's extractor-name directory."
        ),
    )

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

    markers: list[MultiplexMarkerConfig] | None = Field(
        default=None,
        description=(
            "Channel-ordered marker metadata for multiplex preprocessing. "
            "Required when preprocessing.mode='multiplex'."
        ),
    )
    marker_metadata_csv: Path | None = Field(
        default=None,
        description=(
            "Optional extractor-specific marker metadata. For KRONOS2, this is "
            "the additional-marker CSV registered with the model and must contain "
            "only markers absent from its shipped vocabulary. For other multiplex "
            "extractors, it contains marker_name, marker_mean, marker_std and is "
            "used to auto-fill STAMP's marker-normalization statistics."
        ),
    )

    @model_validator(mode="after")
    def _validate_multiplex_fields(self) -> "PreprocessingConfig":
        if self.mode != PreprocessingMode.MULTIPLEX:
            return self

        if not self.markers:
            raise ValueError(
                "preprocessing.markers must be provided when preprocessing.mode='multiplex'"
            )

        return self
