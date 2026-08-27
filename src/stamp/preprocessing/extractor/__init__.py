from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


ExtractorModel = TypeVar("ExtractorModel", bound=nn.Module)


@dataclass(frozen=True)
class Extractor(Generic[ExtractorModel]):
    _: KW_ONLY
    model: ExtractorModel
    transform: Callable[[Image.Image], Float[torch.Tensor, "batch ..."]]
    identifier: str
    """An ID _uniquely_ identifying the model and extractor.
    
    If possible, it should include the digest of the model weights etc.
    so that any change in the model also changes its ID.
    """


@dataclass(frozen=True)
class MultiplexFeatures:
    feats: Float[Tensor, "batch feature"]
    marker_embeddings: Float[Tensor, "batch marker feature"] | None = None
    """Optional per-marker embeddings, when exposed by the extractor."""

    token_embeddings: Float[Tensor, "batch marker token_y token_x feature"] | None = (
        None
    )
    """Optional per-marker spatial embeddings, when exposed by the extractor."""


@dataclass(frozen=True)
class MultiplexExtractor(Generic[ExtractorModel]):
    _: KW_ONLY
    model: ExtractorModel
    identifier: str
    forward: Callable[
        [ExtractorModel, Float[Tensor, "batch marker height width"]],
        MultiplexFeatures,
    ]
    transform: (
        Callable[
            [Float[Tensor, "marker height width"]],
            Float[Tensor, "..."],
        ]
        | None
    ) = None
    preprocess: (
        Callable[
            [
                ExtractorModel,
                Float[Tensor, "batch marker height width"],
                Sequence[str],
            ],
            Float[Tensor, "batch marker height width"],
        ]
        | None
    ) = None
    """Optional marker-aware preprocessing applied before ``transform``.

    Extractors such as KRONOS2 carry their own marker vocabulary and
    normalisation statistics, so they need both the raw channel data and the
    channel-ordered marker names.
    """
    forward_with_markers: (
        Callable[
            [
                ExtractorModel,
                Float[Tensor, "batch marker height width"],
                Sequence[str],
            ],
            MultiplexFeatures,
        ]
        | None
    ) = None
