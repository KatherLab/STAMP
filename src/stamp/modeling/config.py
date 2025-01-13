import os
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from stamp.modeling.data import PandasLabel


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(description="The directory to save the results to")

    clini_table: Path = Field(description="Excel or CSV to read clinical data from")
    slide_table: Path = Field(
        description="Excel or CSV to read patient-slide associations from"
    )
    feature_dir: Path = Field(description="Directory containing feature files")

    ground_truth_label: PandasLabel = Field(
        description="Name of categorical column in clinical table to train on"
    )
    categories: list[str] | None = None

    patient_label: PandasLabel = "PATIENT"
    filename_label: PandasLabel = "FILENAME"

    # Dataset and -loader parameters
    bag_size: int = 512
    num_workers: int = min(os.cpu_count() or 1, 16)

    # Training paramenters
    batch_size: int = 64
    max_epochs: int = 64
    patience: int = 16
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"

    # Experimental features
    use_vary_precision_transform: bool = False
    use_alibi: bool = False


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
