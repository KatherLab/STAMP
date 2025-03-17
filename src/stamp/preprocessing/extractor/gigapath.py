try:
    import timm
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "gigapath dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[gigapath]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def gigapath():
    """Extracts features from slide tiles using GigaPath tile encoder."""

    # Load the model structure
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    # Define the transform
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return Extractor(
        model=model,
        transform=transform,
        identifier="gigapath",  # type: ignore
    )
