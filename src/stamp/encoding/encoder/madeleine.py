import os

import torch
from numpy import ndarray

from stamp.cache import STAMP_CACHE_DIR
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import ExtractorName

try:
    from madeleine.models.factory import create_model_from_pretrained
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "madeleine dependencies not installed."
        " Please update your venv using `uv sync --extra madeleine`"
    ) from e

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = ["Jaume, et al. (https://github.com/mahmoodlab/MADELEINE)"]


class Madeleine(Encoder):
    def __init__(self) -> None:
        model, precision = create_model_from_pretrained(
            os.path.join(STAMP_CACHE_DIR, "madeleine")
        )
        # Check if the GPU supports bfloat16; if not, fall back to float16
        if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            precision = torch.float16
        super().__init__(
            model=model,
            identifier=EncoderName.MADELEINE,
            precision=precision,
            required_extractors=[ExtractorName.CONCH],
        )

    def _generate_slide_embedding(
        self, feats: torch.Tensor, device, **kwargs
    ) -> ndarray:
        feats = feats.unsqueeze(0)
        assert feats.ndim == 3, f"Expected 3D tensor, got {feats.ndim}"
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=self.precision):  # type: ignore
                slide_embedding = self.model.encode_he(feats=feats, device=device)
        # Embedding is casted to float32 as bfloat16 is not always supported
        self.precision = torch.float32
        return slide_embedding.detach().squeeze().to(self.precision).cpu().numpy()

    def _generate_patient_embedding(
        self, feats_list: list, device, **kwargs
    ) -> ndarray:
        all_feats = torch.cat(feats_list, dim=0).unsqueeze(0)
        assert all_feats.ndim == 3, f"Expected 3D tensor, got {all_feats.ndim}"
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=self.precision):  # type: ignore
                patient_embedding = self.model.encode_he(feats=all_feats, device=device)
        self.precision = torch.float32
        return patient_embedding.detach().squeeze().to(self.precision).cpu().numpy()
