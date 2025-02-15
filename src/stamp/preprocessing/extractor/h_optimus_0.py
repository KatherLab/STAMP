import os
from pathlib import Path

import timm
import torch
from torchvision import transforms
from stamp.preprocessing.extractor import Extractor


def h_optimus_0():
    """Extracts features from slide tiles using H-optimus-0 tile encoder."""
    
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617), 
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    return Extractor(
        model=model,
        transform=transform,
        identifier="h_optimus_0",  # type: ignore
    )