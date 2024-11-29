import torch

from stamp.preprocessing.extractor import Extractor

try:
    from conch.open_clip_custom import create_model_from_pretrained
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conch module not found. You can install it using: `pip install git+https://github.com/Mahmoodlab/CONCH.git`"
    ) from e


class StampConchModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch, proj_contrast=False, normalize=False)


def conch() -> Extractor:
    model, preprocess = create_model_from_pretrained(  # type: ignore
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
    )

    return Extractor(
        model=StampConchModel(model),
        transform=preprocess,
        identifier="mahmood-conch",  # type: ignore
    )
