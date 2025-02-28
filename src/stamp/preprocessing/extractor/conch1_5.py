from collections.abc import Callable
from typing import cast

try:
    import timm
    from PIL.Image import Image
    from timm.data.config import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch import Tensor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "conchv1_5 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[conch1_5]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def conch1_5() -> Extractor:
    model = timm.create_model("hf_hub:MahmoodLab/conchv1_5", pretrained=True)
    transform = cast(
        Callable[[Image], Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    return Extractor(
        model=model, transform=transform, identifier="mahmood-conch1_5"
    )