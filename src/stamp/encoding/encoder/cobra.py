from numpy import ndarray

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import ExtractorName

try:
    import torch
    from cobra.utils.load_cobra import get_cobra
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "cobra dependencies not installed."
        " Please update your venv using `uv sync --extra cobra`"
    ) from e

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = ["Lenz, Neidlinger, et al. (https://github.com/KatherLab/COBRA)"]


class Cobra(Encoder):
    def __init__(self) -> None:
        model = get_cobra(download_weights=True)
        if torch.cuda.get_device_capability()[0] < 8:
            print(
                f"\033[93mCOBRA (Mamba2) is designed to run on GPUs with compute capability 8.0 or higher!! "
                f"Your GPU has compute capability {torch.cuda.get_device_capability()[0]}. "
                f"We are forced to switch to mixed FP16 precision. This may lead to numerical instability and reduced performance!!\033[0m"
            )
            precision = torch.float16
        else:
            precision = torch.float32
        super().__init__(
            model=model,
            identifier=EncoderName.COBRA,
            precision=precision,
            required_extractors=[
                ExtractorName.CTRANSPATH,
                ExtractorName.UNI,
                ExtractorName.VIRCHOW2,
                ExtractorName.H_OPTIMUS_0,
            ],
        )

    def _generate_slide_embedding(
        self, feats: torch.Tensor, device, **kwargs
    ) -> ndarray:
        feats = feats.unsqueeze(0).to(device)
        assert feats.ndim == 3, f"Expected 3D tensor, got {feats.ndim}"
        with torch.inference_mode():
            slide_embedding = self.model(feats)
        return slide_embedding.detach().squeeze().cpu().numpy()

    def _generate_patient_embedding(
        self, feats_list: list, device, **kwargs
    ) -> ndarray:
        all_feats = torch.cat(feats_list, dim=0).unsqueeze(0)
        assert all_feats.ndim == 3, f"Expected 3D tensor, got {all_feats.ndim}"
        with torch.inference_mode():
            slide_embedding = self.model(all_feats.to(device))
        return slide_embedding.detach().squeeze().cpu().numpy()
