from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from stamp.preprocessing.tiling import SlideMPP


class HeatmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path

    slide_paths: list[Path] | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    topk: int = 0
    bottomk: int = 0

    default_slide_mpp: SlideMPP | None = None
    """MPP of the slide to use if none can be inferred from the WSI"""


class AttentionUIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path

    slide_paths: list[Path] | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    topk: int = 0
    bottomk: int = 0

    default_slide_mpp: SlideMPP | None = None
    """MPP of the slide to use if none can be inferred from the WSI"""
