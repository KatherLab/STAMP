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
        " Please reinstall stamp using `pip install 'stamp[virchow]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor

__author__ = "Tim Lenz"
__copyright__ = "Copyright (C) 2025 Tim Lenz"
__license__ = "MIT"


class VirchowFull(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model(batch)
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        breakpoint()
        return embedding
    
def virchow() -> Extractor[VirchowFull]:
    """Extracts features from slide tiles using Virchow2 tile encoder."""

    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.eval()
    
    # Define the transform
    transform = cast(
        Callable[[Image.Image], torch.Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )

    return Extractor(
        model=VirchowFull(model),
        transform=transform,
        identifier="virchow_full",
    )