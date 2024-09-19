from pydantic import AliasChoices, BaseModel, Field

from stamp.modeling.config import CrossvalConfig, DeploymentConfig, TrainConfig
from stamp.preprocessing.config import PreprocessingConfig


class StampConfig(BaseModel):
    preprocessing: PreprocessingConfig | None

    # All three are read from the "modeling" field
    training: TrainConfig | None = Field(
        None, validation_alias=AliasChoices("training", "modeling")
    )
    crossval: CrossvalConfig | None = Field(
        None, validation_alias=AliasChoices("crossval", "modeling")
    )
    deployment: DeploymentConfig | None = Field(
        None, validation_alias=AliasChoices("deployment", "modeling")
    )
