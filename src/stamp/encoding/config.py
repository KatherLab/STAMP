from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch._prims_common import DeviceLikeType  # type: ignore


class EncoderName(StrEnum):
    COBRA = "katherlab-cobra"
    EAGLE = "katherlab-eagle"
    CHIEF = "chief"
    TITAN = "mahmood-titan"
    GIGAPATH = "gigapath"
    PRISM = "paigeai-prism"
    MADELEINE = "mahmood-madeleine"
    THREADS = "mahmood-threads"


class SlideEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    device: DeviceLikeType
    agg_feat_dir: Path | None = None


class PatientEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    slide_table: Path
    device: DeviceLikeType
    agg_feat_dir: Path | None = None
