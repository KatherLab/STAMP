from pathlib import Path

from pydantic import BaseModel, ConfigDict


class HeatmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path

    slide_paths: list[Path] | None = None

    topk: int = 0
    bottomk: int = 0
