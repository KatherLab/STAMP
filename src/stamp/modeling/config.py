import os
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.modeling.registry import ModelName
from stamp.types import Category, PandasLabel


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="The directory to save the results to")

    clini_table: Path = Field(description="Excel or CSV to read clinical data from")
    slide_table: Path | None = Field(
        default=None, description="Excel or CSV to read patient-slide associations from"
    )
    feature_dir: Path = Field(description="Directory containing feature files")

    ground_truth_label: PandasLabel = Field(
        description="Name of categorical column in clinical table to train on"
    )
    categories: Sequence[Category] | None = None

    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    params_path: Path | None = Field(
        default=None,
        description="Optional: Path to a YAML file with advanced training parameters.",
    )

    # Experimental features
    use_vary_precision_transform: bool = False


class CrossvalConfig(TrainConfig):
    n_splits: int = Field(5, ge=2)


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    checkpoint_paths: list[Path]
    clini_table: Path | None = None
    slide_table: Path
    feature_dir: Path

    ground_truth_label: PandasLabel | None = None
    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    num_workers: int = min(os.cpu_count() or 1, 16)
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"


class VitModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_model: int = 512
    dim_feedforward: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.0
    # Experimental feature: Use ALiBi positional embedding
    use_alibi: bool = False


class MlpModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_hidden: int = 512
    num_layers: int = 2
    dropout: float = 0.25


class TransMILModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim_hidden: int = 512


class LinearModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Tile level models
    vit: VitModelParams
    trans_mil: TransMILModelParams | None = None
    # Patient level models
    mlp: MlpModelParams
    linear: LinearModelParams | None = None


class AdvancedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bag_size: int = 512
    num_workers: int = min(os.cpu_count() or 1, 16)
    batch_size: int = 64
    max_epochs: int = 32
    patience: int = 16
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"
    max_lr: float = 1e-4
    div_factor: float = 25.0
    model_name: ModelName | None = Field(
        default=None,
        description=(
            "Optional model selection of: vit, mlp, tranmil, linear. "
            "vit or mlp are defaults based on feature type."
        ),
    )
    model_params: ModelParams
