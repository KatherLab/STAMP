from typing import Any

from pydantic import BaseModel

from stamp.heatmaps.config import HeatmapConfig
from stamp.modeling.config import CrossvalConfig, DeploymentConfig, TrainConfig
from stamp.modeling.statistics import StatsConfig
from stamp.preprocessing.config import PreprocessingConfig


class StampConfig(BaseModel):
    preprocessing: PreprocessingConfig | None = None

    training: TrainConfig | None = None
    crossval: CrossvalConfig | None | Any = None
    deployment: DeploymentConfig | None = None

    statistics: StatsConfig | None = None

    heatmaps: HeatmapConfig | None = None
