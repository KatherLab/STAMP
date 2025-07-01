from pathlib import Path
from typing import (
    Final,
    Literal,
    NewType,
    TypeAlias,
    TypeVar,
)

import torch
from beartype.typing import Mapping
from jaxtyping import Bool, Float, Integer
from torch import Tensor

# tiling

ImageExtension: TypeAlias = Literal["png", "jpg"]
EXTENSION_TO_FORMAT: Final[Mapping[ImageExtension, str]] = {
    "png": "png",
    "jpg": "jpeg",
}

Microns = NewType("Microns", float)
"""Micrometers, usually referring to the tissue on the slide"""

SlidePixels = NewType("SlidePixels", int)
"""Pixels of the WSI scan at largest magnification (i.e. coordinates used by OpenSlide)"""

TilePixels = NewType("TilePixels", int)
"""Pixels after resizing, i.e. how they appear on the final tile"""

SlideMPP = NewType("SlideMPP", float)

# modeling

DeviceLikeType: TypeAlias = str | torch.device | int

PatientId: TypeAlias = str
GroundTruth: TypeAlias = str
FeaturePath = NewType("FeaturePath", Path)

Category: TypeAlias = str

BagSize: TypeAlias = int

# A batch of the above
Bags: TypeAlias = Float[Tensor, "batch tile feature"]
BagSizes: TypeAlias = Integer[Tensor, "batch"]  # noqa: F821
EncodedTargets: TypeAlias = Bool[Tensor, "batch category_is_hot"]
"""The ground truth, encoded numerically (currently: one-hot)"""
CoordinatesBatch: TypeAlias = Float[Tensor, "batch tile 2"]

PandasLabel: TypeAlias = str

GroundTruthType = TypeVar("GroundTruthType", covariant=True)
