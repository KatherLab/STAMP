from enum import StrEnum
from typing import Sequence, Type, TypedDict

import lightning

from stamp.modeling.classifier import LitPatientlassifier, LitTileClassifier


class ModelName(StrEnum):
    """Enum for available model names."""

    VIT = "vit"
    MLP = "mlp"
    TRANS_MIL = "trans_mil"
    LINEAR = "linear"


class ModelInfo(TypedDict):
    """A dictionary to map a model to supported feature types. For example,
    a linear classifier is not compatible with tile-evel feats."""

    model_class: Type[lightning.LightningModule]
    supported_features: Sequence[str]


MODEL_REGISTRY: dict[ModelName, ModelInfo] = {
    ModelName.VIT: {
        "model_class": LitTileClassifier,
        "supported_features": LitTileClassifier.supported_features,
    },
    ModelName.MLP: {
        "model_class": LitPatientlassifier,
        "supported_features": LitPatientlassifier.supported_features,
    },
    ModelName.TRANS_MIL: {
        "model_class": LitTileClassifier,
        "supported_features": LitTileClassifier.supported_features,
    },
    ModelName.LINEAR: {
        "model_class": LitPatientlassifier,
        "supported_features": LitPatientlassifier.supported_features,
    },
}