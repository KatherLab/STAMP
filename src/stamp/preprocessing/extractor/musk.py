from stamp.preprocessing.extractor import Extractor

try:
    import torch
    import torchvision
    from musk import modeling, utils
    from PIL import Image
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from timm.models import create_model
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "musk dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[musk]'`"
    ) from e

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2022-2025 Juan Pablo Ricapito"
__license__ = "MIT"


class Musk(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(
            image=batch,
            with_head=False,
            out_norm=False,
            ms_aug=True,
            return_global=True,
        )[0]  # return (vision_cls, text_cls)


def musk() -> Extractor:
    model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate(
        "hf_hub:xiangjx/musk", model, "model|module", ""
    )

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(384, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )

    return Extractor(model=Musk(model), transform=transform, identifier="musk")
