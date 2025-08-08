import torch
import torchvision

try:
    from transformers import CLIPModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PLIP dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[plip]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


class PLIP(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(batch)


def plip() -> Extractor[PLIP]:
    model = CLIPModel.from_pretrained("vinid/plip")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return Extractor(
        model=PLIP(model), transform=transform, identifier=ExtractorName.PLIP
    )
