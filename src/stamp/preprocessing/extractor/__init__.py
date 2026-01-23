from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from PIL import Image
from torch import nn

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
