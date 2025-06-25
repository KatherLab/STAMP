"""A feature extractor not extracting any features

Useful for only generating cache files.
"""

import torch
import torchvision
import torchvision.transforms.functional
from jaxtyping import Float

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2025 Marko van Treeck"
__license__ = "MIT"


class _EmptyModel(torch.nn.Module):
    """A model that returns an empty tensor per input"""

    def forward(
        self, batch: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch feature"]:
        return torch.zeros(batch.size(0), 0).type_as(batch)


def empty() -> Extractor[_EmptyModel]:
    return Extractor(
        model=_EmptyModel(),
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(torch.Tensor.float),
            ]
        ),
        identifier=ExtractorName.EMPTY,
    )
