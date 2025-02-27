try:
    from transformers import AutoModel 
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conchv1_5 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[conch1_5]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def conch1_5():
    """Extracts features from slide tiles using GigaPath tile encoder."""
    titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model, transform = titan.return_conch()
    return Extractor(
        model=model,
        transform=transform,
        identifier="conch1_5",  # type: ignore
    )