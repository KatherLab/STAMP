from pydantic import BaseModel, ConfigDict

from stamp.heatmaps.config import HeatmapConfig, AttentionUIConfig
from stamp.modeling.config import CrossvalConfig, DeploymentConfig, TrainConfig
from stamp.preprocessing.config import PreprocessingConfig
from stamp.statistics import StatsConfig


class StampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: PreprocessingConfig | None = None

    training: TrainConfig | None = None
    crossval: CrossvalConfig | None = None
    deployment: DeploymentConfig | None = None

    statistics: StatsConfig | None = None

    heatmaps: HeatmapConfig | None = None
    attentionui: AttentionUIConfig | None = None