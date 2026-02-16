"""
Port from https://github.com/Snarci/RedDino
RedDino: A Foundation Model for Red Blood Cell Analysis
"""

from typing import Callable, cast

try:
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "red_dino dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[red_dino]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

__author__ = ""
__copyright__ = ""
__license__ = "MIT"


class RedDinoClsOnly(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch)
        if isinstance(out, tuple):
            out = out[0]
        # if model returns tokens, return class token
        if getattr(out, "ndim", 0) >= 2 and out.shape[1] > 1:
            return out[:, 0]
        return out


def red_dino() -> Extractor[RedDinoClsOnly]:
    """Extracts features from single image using RedDino encoder."""

    model = timm.create_model(
        "hf-hub:Snarcy/RedDino-large",
        pretrained=True,
    )

    transform = cast(
        Callable[[Image.Image], torch.Tensor],
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    return Extractor(
        model=RedDinoClsOnly(model),
        transform=transform,
        identifier=ExtractorName.RED_DINO,
    )
