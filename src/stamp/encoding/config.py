from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch._prims_common import DeviceLikeType


class EncoderName(StrEnum):
    COBRA = "katherlab-cobra"
    CHIEF = "chief"
    TITAN = "mahmood-titan"


class SlideEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    device: DeviceLikeType


class PatientEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    slide_table: Path
    device: DeviceLikeType
