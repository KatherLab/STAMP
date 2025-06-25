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

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
# nah just kidding


class VirchowConcatenated(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model(batch)

        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def virchow_full() -> Extractor[VirchowConcatenated]:
    """Extracts features from slide tiles using Virchow tile encoder
    concatenating the class token with the mean patch token to create
    the final tile embedding of dimension 2560.
    """

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
        model=VirchowConcatenated(model),
        transform=transform,
        identifier=ExtractorName.VIRCHOW_FULL,
    )
