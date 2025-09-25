from enum import StrEnum

from stamp.modeling.models import (
    LitPatientClassifier,
    LitTileClassifier,
    LitTileRegressor,
    LitTileSurvival,
)
from stamp.types import Task


class ModelName(StrEnum):
    """Enum for available model names."""

    VIT = "vit"
    MLP = "mlp"
    TRANS_MIL = "trans_mil"
    LINEAR = "linear"


# Map (feature_type, task) → correct Lightning wrapper class
MODEL_REGISTRY = {
    ("tile", "classification"): LitTileClassifier,
    ("tile", "regression"): LitTileRegressor,
    ("tile", "survival"): LitTileSurvival,
    ("patient", "classification"): LitPatientClassifier,
}


def load_model_class(task: Task, feature_type: str, model_name: ModelName):
    LitModelClass = MODEL_REGISTRY[(feature_type, task)]

    match model_name:
        case ModelName.VIT:
            from stamp.modeling.models.vision_tranformer import (
                VisionTransformer as ModelClass,
            )

        case ModelName.TRANS_MIL:
            from stamp.modeling.models.trans_mil import (
                TransMIL as ModelClass,
            )

        case ModelName.MLP:
            from stamp.modeling.models.mlp import MLP as ModelClass

        case ModelName.LINEAR:
            from stamp.modeling.models.mlp import (
                Linear as ModelClass,
            )

        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return LitModelClass, ModelClass
