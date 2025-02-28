from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path
from torch._prims_common import DeviceLikeType

class EncoderName(StrEnum):
    COBRA = "katherlab-cobra"

class SlideEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    slide_table: Path
    device: DeviceLikeType
    dtype: torch.dtype,
    # TODO: Add this to the config yaml