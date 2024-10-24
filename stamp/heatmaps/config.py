from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field


class HeatmapConfig(BaseModel):
    output_dir: Path

    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path = Field(
        validation_alias=AliasChoices("checkpoint_path", "model_path")
    )

    slides: list[Path] | None = None
