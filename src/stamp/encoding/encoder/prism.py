import numpy as np
import torch
from transformers import AutoModel

# Patch for transformers compatibility with Prism HuggingFace model
# Prism model code uses transformers.activations.ACT2FN which was removed in newer versions
import sys
# try:
from transformers.activations import ACT2FN  # noqa: F401
# except (ImportError, ModuleNotFoundError):
#     # transformers >= 4.46 removed transformers.activations;
#     # create a shim module so that trust_remote_code model code can still
#     # do "from transformers.activations import ACT2FN"
#     import types
#     from transformers.utils.generic import ACT2FN  # type: ignore[attr-defined]
#     _shim = types.ModuleType("transformers.activations")
#     _shim.ACT2FN = ACT2FN
#     sys.modules["transformers.activations"] = _shim

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
