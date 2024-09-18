# %%
from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    clini_table: Path
    slide_table: Path
    feature_dir: Path
    output_dir: Path
    target_label: str = Field(pattern="^[a-zA-Z]+$")
    categories: list[str] | None = None
    cat_labels: list[str] | None = None
    cont_labels: list[str] | None = None


class CrossvalConfig(TrainConfig):
    n_splits: int = Field(ge=2)


class DeploymentConfig(BaseModel):
    clini_table: Path
    slide_table: Path
    output_dir: Path
    deploy_feature_dir: Path
    target_label: str = Field(pattern="^[a-zA-Z]+$")
    cat_labels: list[str] | None = None
    cont_labels: list[str] | None = None
    checkpoint_path: Path = Field(alias="model_path")
