"""
Adopted from https://github.com/MAGIC-AI4Med/KEEP
KEEP (KnowledgE-Enhanced Pathology)
"""

try:
    import json

    import timm
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "keep dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[keep]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


class KEEPImageModel(nn.Module):
    def __init__(self, vision_config: dict, projection_dim: int) -> None:
        super().__init__()

        self.visual = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=vision_config["img_size"],
            patch_size=vision_config["patch_size"],
            init_values=vision_config["init_values"],
            num_classes=vision_config["num_classes"],
            dynamic_img_size=vision_config.get("dynamic_img_size", True),
        )

        self.visual_head = nn.Sequential(
            nn.Linear(self.visual.num_features, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode_image(self, image_inputs: torch.Tensor) -> torch.Tensor:
        vision_features = self.visual(image_inputs)
        return torch.nn.functional.normalize(self.visual_head(vision_features), dim=-1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encode_image(batch)


def _remap_layerscale_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for k, v in sd.items():
        if ".ls1.weight" in k or ".ls2.weight" in k:
            k = k.replace(".weight", ".gamma")
        remapped[k] = v
    return remapped


def _load_keep_image_model() -> KEEPImageModel:
    repo_id = "Astaxanthin/KEEP"

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        repo_type="model",
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        repo_type="model",
    )

    with open(config_path) as f:
        cfg = json.load(f)

    model = KEEPImageModel(
        vision_config=cfg["vision_config"],
        projection_dim=cfg["projection_dim"],
    )

    sd = load_file(weights_path)
    sd = {
        k: v
        for k, v in sd.items()
        if k.startswith("visual.") or k.startswith("visual_head.")
    }
    sd = _remap_layerscale_keys(sd)

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def keep() -> Extractor[KEEPImageModel]:
    """Extracts features from slide tiles using the KEEP tile encoder."""
    model = _load_keep_image_model()

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
        model=model,
        transform=transform,
        identifier=ExtractorName.KEEP,
    )
