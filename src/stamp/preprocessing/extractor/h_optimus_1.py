try:
    import timm
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "h_optimus_1 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[h_optimus_1]'`"
    ) from e

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


def h_optimus_1() -> Extractor:
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )

    return Extractor(
        model=model,
        transform=transform,
        identifier=ExtractorName.H_OPTIMUS_1,  # type: ignore
    )
