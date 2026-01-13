from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from stamp.types import DeviceLikeType, PandasLabel


class EncoderName(StrEnum):
    COBRA = "cobra"
    EAGLE = "eagle"
    CHIEF_CTRANSPATH = "chief"
    TITAN = "titan"
    GIGAPATH = "gigapath"
    MADELEINE = "madeleine"
    PRISM = "prism"
    TICON = "ticon"


class SlideEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    device: DeviceLikeType
    agg_feat_dir: Path | None = None
    generate_hash: bool = True


class PatientEncodingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    encoder: EncoderName
    output_dir: Path
    feat_dir: Path
    slide_table: Path
    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"
    device: DeviceLikeType
    agg_feat_dir: Path | None = None
    generate_hash: bool = True
