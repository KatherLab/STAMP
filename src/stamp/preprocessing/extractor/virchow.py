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
        "virchow dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[virchow2]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor

__author__ = "Tim Lenz"
__copyright__ = "Copyright (C) 2025 Tim Lenz"
__license__ = "MIT"


class VirchowClsOnly(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)[:, 0]


def virchow() -> Extractor[VirchowClsOnly]:
    """Extracts features from slide tiles using Virchow tile encoder.
    The ouput consists of the class token only, resulting in a embedding
    of dimension 1280"""

    # Load the model structure
    model = timm.create_model(  # pyright: ignore[reportPrivateImportUsage]
        "hf-hub:paige-ai/Virchow",
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
        model=VirchowClsOnly(model),
        transform=transform,
        identifier="virchow",
    )
