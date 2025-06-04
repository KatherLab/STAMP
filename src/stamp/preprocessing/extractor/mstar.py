from torchvision import transforms

from stamp.preprocessing.extractor import Extractor

try:
    import timm
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "mSTAR dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[mstar]'`"
    ) from e


def mstar() -> Extractor:
    model = timm.create_model(
        "hf-hub:Wangyh/mSTAR", pretrained=True, init_values=1e-5, dynamic_img_size=True
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return Extractor(model=model, transform=transform, identifier="mstar")
