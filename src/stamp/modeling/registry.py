from enum import StrEnum
from typing import Sequence, Type, TypedDict

import lightning

from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.mlp_classifier import LitMLPClassifier


class ModelName(StrEnum):
    """Enum for available model names."""

    VIT = "vit"
    MLP = "mlp"


class ModelInfo(TypedDict):
    """A dictionary to map a model to supported feature types. For example,
    a linear classifier is not compatible with tile-evel feats."""

    model_class: Type[lightning.LightningModule]
    supported_features: Sequence[str]


MODEL_REGISTRY: dict[ModelName, ModelInfo] = {
    ModelName.VIT: {
        "model_class": LitVisionTransformer,
        "supported_features": LitVisionTransformer.supported_features,
    },
    ModelName.MLP: {
        "model_class": LitMLPClassifier,
        "supported_features": LitMLPClassifier.supported_features,
    },
}
