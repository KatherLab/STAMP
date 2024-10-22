from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field


class TrainConfig(BaseModel):
    output_dir: Path

    clini_table: Path
    slide_table: Path
    feature_dir: Path
    target_label: str = Field(pattern="^[a-zA-Z0-9_]+$")
    categories: list[str] | None = None
    cat_labels: list[str] | None = Field(default_factory=list)
    cont_labels: list[str] | None = Field(default_factory=list)


class CrossvalConfig(TrainConfig):
    n_splits: int = Field(5, ge=2)


class DeploymentConfig(BaseModel):
    output_dir: Path

    clini_table: Path
    slide_table: Path
    feature_dir: Path = Field(
        validation_alias=AliasChoices("feature_dir", "default_feature_dir")
    )
    target_label: str = Field(pattern="^[a-zA-Z0-9_]+$")
    cat_labels: list[str] | None = Field(default_factory=list)
    cont_labels: list[str] | None = Field(default_factory=list)
    # We can't have things called `model_` in pydantic, so let's call it `checkpoint_path` instead
    checkpoint_path: Path = Field(
        validation_alias=AliasChoices("model_path", "checkpoint_path")
    )
