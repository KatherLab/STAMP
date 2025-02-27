from stamp.cache import STAMP_CACHE_DIR, file_digest
from pathlib import Path

try:
    import gdown
    import torch
    import torch.nn as nn
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "chief_ctranspath dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[chief_ctranspath]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor, ctranspath


def chief_ctranspath(model_path: Path | None = None) -> Extractor[ctranspath._SwinTransformer]:
    model_path = model_path or STAMP_CACHE_DIR / "chief_ctranspath.pth"
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(
            "https://drive.google.com/u/0/uc?id=1_vgRF1QXa8sPCOpJ1S9BihwZhXQMOVJc&export=download",
            str(model_path),
        )

    digest = file_digest(model_path)
    assert (
        digest == "1646f23001214f74cf432ef0e80b808ee6605143802ae6ed53a87564ddc4924a"
    ), (
        f"The digest of the downloaded checkpoint ({model_path}) did not match the expected value."
    )

    model = ctranspath._swin_tiny_patch4_window7_224(embed_layer=ctranspath._ConvStem, pretrained=False)
    model.head = nn.Identity()
    chief_ctranspath = torch.load(model_path, weights_only=False)
    model.load_state_dict(chief_ctranspath["model"], strict=True)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
    )

    return Extractor(
        model=model,
        transform=transform,
        identifier="chief_ctranspath",
    )
