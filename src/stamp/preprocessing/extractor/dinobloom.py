# %%
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torchvision import transforms

from stamp.preprocessing.extractor import Extractor


def file_digest(file: str | Path) -> str:
    with open(file, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


stamp_resources_dir = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
)

embed_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def get_dino_bloom(model_path: Path, modelname: str = "dinov2_vits14") -> nn.Module:
    # load the original DINOv2 model with the correct architecture and parameters.
    model = cast(nn.Module, torch.hub.load("facebookresearch/dinov2", modelname))
    # load finetuned weights
    pretrained = torch.load(model_path, map_location=torch.device("cpu"))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained["teacher"].items():
        if "dino_head" in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = value

    # corresponds to 224x224 image. patch size=14x14 => 16*16 patches
    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes["dinov2_vits14"]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model


def dino_bloom() -> Extractor:
    model_file = stamp_resources_dir / "dinobloom-s.pth"

    if not model_file.exists():
        tmp_model_file = model_file.with_suffix(".tmp")
        urllib.request.urlretrieve(
            "https://zenodo.org/records/10908163/files/DinoBloom-S.pth",
            tmp_model_file,
        )
        assert (
            file_digest(tmp_model_file)
            == "c2f7990b003e89bcece80e379fb8fe0ba2ec392ce19b286e8a294abd99568e44"
        ), "unexpected model weights"

        # Only rename on successful download
        tmp_model_file.rename(model_file)

    return Extractor(
        model=get_dino_bloom(model_file),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ],
        ),
        identifier="dinobloom-c2f7990b",
    )
