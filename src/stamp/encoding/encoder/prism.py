import numpy as np
import torch
from transformers import AutoModel

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import ExtractorName

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = ["Shaikovski, et al. (https://huggingface.co/paige-ai/Prism)"]


class Prism(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        super().__init__(
            model=model,
            identifier=EncoderName.PRISM,
            precision=torch.float16,
            required_extractors=[ExtractorName.VIRCHOW_FULL],
        )

    def _generate_slide_embedding(self, feats, device, **kwargs) -> np.ndarray:
        with torch.autocast(device, dtype=self.precision), torch.inference_mode():
            return (
                self.model.slide_representations(feats.unsqueeze(0).to(device))[
                    "image_embedding"
                ]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )

    def _generate_patient_embedding(self, feats_list, device, **kwargs) -> np.ndarray:
        all_feats = torch.cat(feats_list, dim=0).to(device)
        with torch.inference_mode():
            return (
                self.model.slide_representations(all_feats.unsqueeze(0))[
                    "image_embedding"
                ]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
