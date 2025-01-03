from collections.abc import Callable
from typing import cast

try:
    import timm
    import uni as _  # noqa: F401
    from PIL.Image import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch import Tensor
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "uni dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[uni]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


def uni(revision: str = "77ffbca1ee1cdcee6e87f6deebd2db8a5888c721") -> Extractor:
    model = timm.create_model(
        f"hf-hub:MahmoodLab/uni@{revision}",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    transform = cast(
        Callable[[Image], Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    return Extractor(
        model=model, transform=transform, identifier=f"mahmood-uni-{revision[:8]}"
    )
