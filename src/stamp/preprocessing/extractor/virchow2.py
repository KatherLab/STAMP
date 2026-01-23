from typing import Callable, cast

try:
    import timm
    import torch
    from PIL import Image
    from timm.data.config import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers.mlp import SwiGLUPacked
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "virchow2 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[virchow2]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

__author__ = "Tim Lenz"
__copyright__ = "Copyright (C) 2025 Tim Lenz"
__license__ = "MIT"


class Virchow2ClsOnly(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)[:, 0]


def virchow2() -> Extractor[Virchow2ClsOnly]:
    """Extracts features from slide tiles using Virchow2 tile encoder."""

    # Load the model structure
    model = timm.create_model(  # pyright: ignore[reportPrivateImportUsage]
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )

    # Define the transform
    transform = cast(
        Callable[[Image.Image], torch.Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )

    return Extractor(
        model=Virchow2ClsOnly(model),
        transform=transform,
        identifier=ExtractorName.VIRCHOW2,
    )
