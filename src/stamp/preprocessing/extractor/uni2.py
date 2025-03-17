try:
    import timm
    import torch
    from timm.data import resolve_data_config  # type: ignore
    from timm.data.transforms_factory import create_transform
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "uni2 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[uni2]'`"
    ) from e

from stamp.preprocessing.extractor import Extractor


def uni2() -> Extractor:
    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval()

    return Extractor(
        model=model,
        transform=transform,
        identifier="uni2",  # type: ignore
    )
