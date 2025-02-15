import os
from pathlib import Path

#import timm
import torch
#from torchvision import transforms
#from timm.data import resolve_data_config
#from timm.data.transforms_factory import create_transform
from transformers import AutoModel 
from stamp.preprocessing.extractor import Extractor
_stamp_cache_dir = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
)


def conchv1_5():
    """Extracts features from slide tiles using GigaPath tile encoder."""

    # Load the model structure
    # model = timm.create_model("hf-hub:MahmoodLab/CONCHv_1_5", pretrained=True)

    
    # model=hf_hub_download("MahmoodLab/CONCHv_1_5", filename="pytorch_model_vision.bin", 
    #                                       local_dir=str(_stamp_cache_dir), 
    #                                       force_download=True)
    
    # Define the transform
    # transform = transforms.Compose([
    #     transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ])
    # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    # model.eval()
    titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model, transform = titan.return_conch()
    return Extractor(
        model=model,
        transform=transform,
        identifier="conchv1_5",  # type: ignore
    )