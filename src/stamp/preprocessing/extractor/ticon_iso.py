"""
TICON Isolated Mode - Single tile processing compatible with Extractor pipeline.

This module provides TICON in "isolated inference" mode, where each tile is
processed independently through a tile encoder and then through TICON.

While this mode doesn't provide slide-level context, TICON still enhances
individual tile representations.  For full slide-level contextualization,
use TiconEncoder after feature extraction.
"""

from typing import Callable, cast

try:
    import timm
    import torch
    import torch.nn as nn
    from PIL import Image
    from timm.data.config import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers.mlp import SwiGLUPacked
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "TICON dependencies not installed.  "
        "Please reinstall stamp using `pip install 'stamp[ticon]'`"
    ) from e

from stamp.modeling.models.ticon_architecture import (
    TILE_EXTRACTOR_TO_TICON,
    get_ticon_key,
    load_ticon_backbone,
)
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

# =============================================================================
# Tile Encoder Wrappers
# =============================================================================


class _Virchow2ClsOnly(nn.Module):
    """Wrapper for Virchow2 to return only CLS token."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)[:, 0]


# =============================================================================
# Tile Encoder Factory
# =============================================================================


def _create_tile_encoder(
    extractor: ExtractorName,
) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """
    Create tile encoder and transform for a given extractor.

    Args:
        extractor: The tile extractor to create

    Returns:
        Tuple of (model, transform)

    Raises:
        ValueError: If extractor is not supported
        ModuleNotFoundError: If required dependencies are missing
    """
    if extractor == ExtractorName.H_OPTIMUS_1:
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
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return model, transform

    elif extractor == ExtractorName.GIGAPATH:
        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return model, transform

    elif extractor == ExtractorName.UNI2:
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
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **timm_kwargs,
        )
        transform = cast(
            Callable[[Image.Image], torch.Tensor],
            create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
        )
        return model, transform

    elif extractor == ExtractorName.VIRCHOW2:
        base_model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model = _Virchow2ClsOnly(base_model)
        transform = cast(
            Callable[[Image.Image], torch.Tensor],
            create_transform(
                **resolve_data_config(base_model.pretrained_cfg, model=base_model)
            ),
        )
        return model, transform

    elif extractor == ExtractorName.CONCH1_5:
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CONCH v1.5 dependencies not installed. "
                "Please reinstall stamp using `pip install 'stamp[conch1_5]'`"
            ) from e

        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        model, transform = titan.return_conch()
        return model, transform

    else:
        raise ValueError(
            f"Unsupported tile extractor for TICON: {extractor}. "
            f"Supported:  {list(TILE_EXTRACTOR_TO_TICON.keys())}"
        )


# =============================================================================
# TICON Isolated Model
# =============================================================================


class TICON(nn.Module):
    """
    TICON in Isolated Inference Mode.

    Processes tiles independently:  TileEncoder -> TICON (single tile).
    Compatible with standard Extractor pipeline.

    Supports all tile encoders that TICON was trained on:
    - H-Optimus-1 (1536-dim)
    - GigaPath (1536-dim)
    - UNI2 (1536-dim)
    - Virchow2 (1280-dim)
    - CONCH v1.5 (768-dim)

    Note:
        This mode doesn't use slide-level context.  For full contextualization,
        use TiconEncoder after feature extraction.

    Args:
        tile_extractor: Which tile encoder to use
        device: Device to run on (default: "cuda")

    Example:
        >>> model = TICON(tile_extractor=ExtractorName.GIGAPATH)
        >>> embedding = model(tile_batch)  # [B, 1536]
    """

    def __init__(
        self,
        tile_extractor: ExtractorName = ExtractorName.H_OPTIMUS_1,
        device: str = "cuda",
    ):
        super().__init__()
        self._device = torch.device(device)
        self.tile_extractor = tile_extractor

        # Validate extractor is supported by TICON
        if tile_extractor not in TILE_EXTRACTOR_TO_TICON:
            raise ValueError(
                f"Tile extractor {tile_extractor} is not supported by TICON.  "
                f"Supported:  {list(TILE_EXTRACTOR_TO_TICON.keys())}"
            )

        # Get TICON key and embedding dimension
        self.tile_encoder_key, self.embed_dim = get_ticon_key(tile_extractor)

        # Stage 1: Create tile encoder
        self.tile_encoder, self._transform = _create_tile_encoder(tile_extractor)

        # Stage 2: Load TICON backbone
        self.ticon = load_ticon_backbone(device=device)

        self.to(self._device)
        self.eval()

    def get_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Get image transform for this tile extractor."""
        return self._transform

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process tiles: TileEncoder -> TICON (isolated mode).

        Args:
            x: [B, 3, 224, 224] batch of tile images

        Returns:
            [B, embed_dim] contextualized embeddings
        """
        x = x.to(self._device, non_blocking=True)

        # Stage 1: Extract tile features with autocast
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(self._device.type == "cuda"),
        ):
            emb = self.tile_encoder(x)

        # Handle different output shapes (some models return [B, N, D])
        if emb.dim() == 3:
            emb = emb[:, 0]  # Take CLS token

        # Add sequence dimension for TICON:  [B, D] -> [B, 1, D]
        emb = emb.unsqueeze(1)

        # Stage 2: TICON (single tile = no spatial context, use zero coords)
        coords = torch.zeros(
            emb.size(0),
            1,
            2,
            device=self._device,
            dtype=torch.float32,
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(self._device.type == "cuda"),
        ):
            out = self.ticon(
                x=emb.float(),  # TICON expects float32 input
                relative_coords=coords,
                tile_encoder_key=self.tile_encoder_key,
            )

        # Remove sequence dimension: [B, 1, D] -> [B, D]
        return out.squeeze(1)


# =============================================================================
# Factory Function (für extract_ in __init__.py)
# =============================================================================


def ticon_iso(
    tile_extractor: ExtractorName = ExtractorName.H_OPTIMUS_1,
    device: str = "cuda",
) -> Extractor[TICON]:
    """
    Create TICON in Isolated Mode (Extractor-compatible).

    This mode processes each tile independently through both the tile encoder
    and TICON. While it doesn't provide slide-level context, TICON still
    enhances individual tile representations.

    Args:
        tile_extractor:  Which tile encoder to use.  Supported:
            - ExtractorName.H_OPTIMUS_1 (default)
            - ExtractorName.GIGAPATH
            - ExtractorName.UNI2
            - ExtractorName.VIRCHOW2
            - ExtractorName.CONCH1_5
        device: CUDA device

    Returns:
        Extractor compatible with standard pipeline
    """
    model = TICON(tile_extractor=tile_extractor, device=device)

    return Extractor(
        model=model,
        transform=model.get_transform(),
        identifier=ExtractorName.TICON,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "TICON",
    "ticon_iso",
]
