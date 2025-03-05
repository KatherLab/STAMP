from collections.abc import Callable
from typing import cast

try:
    from transformers import AutoModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conchv1_5 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[conch1_5]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def conch1_5() -> Extractor:
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    model, eval_transform = titan.return_conch()
    return Extractor(
        model=model, transform=eval_transform, identifier="mahmood-conch1_5"
    )
