"""
Adopted from https://github.com/MAGIC-AI4Med/KEEP
KEEP (KnowledgE-Enhanced Pathology)
"""

try:
    import torch
    from torchvision import transforms
    from transformers import AutoModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "keep dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[keep]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


class KEEPWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch)


def keep() -> Extractor[KEEPWrapper]:
    """Extracts features from slide tiles using the KEEP tile encoder."""
    model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                size=224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return Extractor(
        model=KEEPWrapper(model),
        transform=transform,
        identifier=ExtractorName.KEEP,
    )
