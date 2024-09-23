from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field


class HeatmapConfig(BaseModel):
    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path = Field(alias=AliasChoices("checkpoint_path", "model_path"))
    output_dir: Path
    n_toptiles: int = 8
    overview: bool = True
