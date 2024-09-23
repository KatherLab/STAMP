from typing import Any

from pydantic import AliasChoices, AliasPath, BaseModel, Field

from stamp.heatmaps.config import HeatmapConfig
from stamp.modeling.config import CrossvalConfig, DeploymentConfig, TrainConfig
from stamp.modeling.statistics import StatsConfig
from stamp.preprocessing.config import PreprocessingConfig


class StampConfig(BaseModel):
    preprocessing: PreprocessingConfig | None

    # All three are read from the "modeling" field
    # TODO all these `Any`s and `union_mode`s are only necessary to catch half-defined aliases
    training: TrainConfig | None | Any = Field(
        None,
        validation_alias=AliasChoices("training", "modeling"),
        union_mode="left_to_right",
    )
    crossval: CrossvalConfig | None | Any = Field(
        None,
        validation_alias=AliasChoices("crossval", "modeling"),
        union_mode="left_to_right",
    )
    deployment: DeploymentConfig | None | Any = Field(
        None,
        validation_alias=AliasChoices("deployment", "modeling"),
        union_mode="left_to_right",
    )

    statistics: StatsConfig | None = Field(
        None,
        validation_alias=AliasChoices(
            "statistics", AliasPath("modeling", "statistics")
        ),
    )

    heatmaps: HeatmapConfig | None = Field(None)
