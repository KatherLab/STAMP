from typing import cast

import torch
from torchvision.transforms import Compose

from stamp.preprocessing.extractor import Extractor

try:
    from conch.open_clip_custom import create_model_from_pretrained
    from conch.open_clip_custom.coca_model import CoCa
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conch dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[conch]'`"
    ) from e

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class _StampConchModel(torch.nn.Module):
    def __init__(self, model: CoCa) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch, proj_contrast=False, normalize=False)


def conch() -> Extractor[_StampConchModel]:
    model, preprocess = cast(
        tuple[CoCa, Compose],
        create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch"),
    )

    return Extractor(
        model=_StampConchModel(model),
        transform=preprocess,
        identifier="mahmood-conch",
    )
