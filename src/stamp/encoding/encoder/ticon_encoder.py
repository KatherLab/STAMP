"""
TICON Model Architecture and Configuration.

Shared between "Isolated" and "Contextualized" modes.
Contains all model components, configuration, and utility functions.
Adapted from:

@misc{belagali2025ticonslideleveltilecontextualizer,
      title={TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning},
      author={Varun Belagali and Saarthak Kapse and Pierre Marza and Srijan Das and Zilinghan Li and Sofiène Boutaj and Pushpak Pati and Srikar Yellapragada and Tarak Nath Nandi and Ravi K Madduri and Joel Saltz and Prateek Prasanna and Stergios Christodoulidis and Maria Vakalopoulou and Dimitris Samaras},
      year={2025},
      eprint={2512.21331},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21331},
}
"""
import logging
import math
import os
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

# try:
#     from torch.amp.autocast_mode import autocast
# except (ImportError, AttributeError):
#     try:
#         from torch.cuda.amp import autocast
#     except ImportError:
#         from torch.amp import autocast  # type: ignore
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder, EncoderName
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType

_logger = logging.getLogger("stamp")

# Mapping:  ExtractorName -> (ticon_key, embedding_dim)
TILE_EXTRACTOR_TO_TICON: dict[ExtractorName, tuple[ExtractorName, int]] = {
    ExtractorName.CONCH1_5: (ExtractorName.CONCH1_5, 768),
    ExtractorName.H_OPTIMUS_1: (ExtractorName.H_OPTIMUS_1, 1536),
    ExtractorName.UNI2: (ExtractorName.UNI2, 1536),
    ExtractorName.GIGAPATH: (ExtractorName.GIGAPATH, 1536),
    ExtractorName.VIRCHOW2: (ExtractorName.VIRCHOW2, 1280),
}

# TICON model configuration
TICON_MODEL_CFG: dict[str, Any] = {
    "transformers_kwargs": {
        "embed_dim": 1536,
        "drop_path_rate": 0.0,
        "block_kwargs": {
            "attn_kwargs": {"num_heads": 24},
        },
    },
    "encoder_kwargs": {"depth": 6},
    "decoder_kwargs": {"depth": 1},
    "in_dims": [768, 1536, 1536, 1536, 1280],
    "tile_encoder_keys": [
        ExtractorName.CONCH1_5,
        ExtractorName.H_OPTIMUS_1,
        ExtractorName.UNI2,
        ExtractorName.GIGAPATH,
        ExtractorName.VIRCHOW2,
    ],
    "num_decoders": 1,
    "decoder_out_dims": [768, 1536, 1536, 1536, 1280],
}


def get_ticon_key(extractor: ExtractorName) -> tuple[ExtractorName, int]:
    """Get TICON key and embedding dimension for a given tile extractor."""
    if extractor not in TILE_EXTRACTOR_TO_TICON:
        raise ValueError(
            f"No TICON mapping for extractor {extractor}. "
            f"Supported: {list(TILE_EXTRACTOR_TO_TICON.keys())}"
        )
    return TILE_EXTRACTOR_TO_TICON[extractor]


def get_slopes(n: int) -> list[float]:
    """Get ALiBi slopes for n attention heads."""

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def scaled_dot_product_attention_alibi(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    # try Flash Attention with ALiBi first
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_bias,
                dropout_p=dropout_p if training else 0.0,
                is_causal=False,
            )
    except Exception:
        pass

    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=training)

    return attn_weight @ value


## TICON BACKBONE COMPONENTS
class Mlp(nn.Module):
    """MLP with SwiGLU activation (used in TICON transformer blocks)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: float = 16 / 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            hidden_features = int(in_features * mlp_ratio)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features // 2, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        return self.fc2(x)


class ProjectionMlp(nn.Module):
    """Projection MLP for input/output transformations with LayerNorm."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.norm(x)


class Attention(nn.Module):
    """Multi-head attention with ALiBi spatial bias for TICON."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # ALiBi slopes (registered as buffer for proper device handling)
        slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes[None, :, None, None])

    def forward(
        self,
        x: Float[Tensor, "b n_q d"],
        coords: Float[Tensor, "b n_q 2"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> Float[Tensor, "b n_q d"]:
        if context is None:
            context = x
            context_coords = coords

        b, n_q, d = x.shape
        n_k = context.shape[1]
        h = self.num_heads

        # Project queries, keys, values
        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)

        # Validate coordinates are available
        if coords is None or context_coords is None:
            raise ValueError(
                "Coordinates must be provided for spatial attention with ALiBi bias"
            )
        # Compute spatial distances for ALiBi
        coords_exp = coords.unsqueeze(2).expand(-1, -1, n_k, -1)
        ctx_coords_exp = context_coords.unsqueeze(1).expand(-1, n_q, -1, -1)
        euclid_dist = torch.sqrt(torch.sum((coords_exp - ctx_coords_exp) ** 2, dim=-1))

        # Apply ALiBi bias
        attn_bias = -self.slopes * euclid_dist[:, None, :, :]

        # Attention with ALiBi
        x = scaled_dot_product_attention_alibi(
            q,
            k,
            v,
            attn_bias=attn_bias,
            training=self.training,
        )

        x = x.transpose(1, 2).reshape(b, n_q, d)
        return self.proj(x)


class ResidualBlock(nn.Module):
    """Residual connection with optional layer scale and stochastic depth."""

    def __init__(
        self,
        drop_prob: float,
        norm: nn.Module,
        fn: nn.Module,
        gamma: nn.Parameter | None,
    ):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob
        self.gamma = gamma

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        fn_out = self.fn(self.norm(x), **kwargs)

        if self.gamma is not None:
            fn_out = self.gamma * fn_out

        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out

        # Stochastic depth
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        drop_path: float,
        norm_layer: Callable[[int], nn.Module],
        context_dim: int | None,
        layer_scale: bool = True,
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()

        gamma1 = nn.Parameter(torch.ones(dim)) if layer_scale else None
        gamma2 = nn.Parameter(torch.ones(dim)) if layer_scale else None

        self.residual1 = ResidualBlock(
            drop_path,
            norm_layer(dim),
            Attention(dim, context_dim=context_dim, **attn_kwargs),
            gamma1,
        )
        self.residual2 = ResidualBlock(
            drop_path,
            norm_layer(dim),
            Mlp(in_features=dim),
            gamma2,
        )

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        context: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> Tensor:
        x = self.residual1(
            x,
            context=context,
            coords=coords,
            context_coords=context_coords,
        )
        x = self.residual2(x)
        return x


class Transformer(nn.Module):
    """Transformer encoder/decoder stack for TICON."""

    def __init__(
        self,
        embed_dim: int,
        norm_layer: Callable[[int], nn.Module],
        depth: int,
        drop_path_rate: float,
        context_dim: int | None = None,
        block_kwargs: Mapping[str, Any] = {},
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    context_dim=context_dim,
                    **block_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        return_layers: set[int],
        contexts: list[Tensor] | None = None,
        context_coords: Tensor | None = None,
    ) -> dict[int, Tensor]:
        outputs = {}
        if 0 in return_layers:
            outputs[0] = x

        for blk_idx, blk in enumerate(self.blocks):
            context = contexts[blk_idx] if contexts is not None else None
            x = blk(
                x,
                coords=coords,
                context=context,
                context_coords=context_coords,
            )
            if blk_idx + 1 in return_layers:
                outputs[blk_idx + 1] = x

        return outputs


class TiconBackbone(nn.Module):
    """
    TICON Encoder-Decoder backbone.

    This is the core TICON model that contextualizes tile embeddings
    using spatial attention with ALiBi positional bias.
    """

    def __init__(
        self,
        in_dims: list[int],
        tile_encoder_keys: list[str],
        transformers_kwargs: Mapping[str, Any],
        encoder_kwargs: Mapping[str, Any],
        decoder_kwargs: Mapping[str, Any] = {},
        norm_layer_type: str = "LayerNorm",
        norm_layer_kwargs: Mapping[str, Any] = {"eps": 1e-5},
        final_norm_kwargs: Mapping[str, Any] = {"elementwise_affine": True},
        out_layer: int = -1,
        num_decoders: int = 0,
        decoder_out_dims: list[int] = [],
        **kwargs,  # Ignore extra kwargs like patch_size
    ):
        super().__init__()

        norm_layer: Callable[[int], nn.Module] = partial(
            getattr(nn, norm_layer_type), **norm_layer_kwargs
        )

        self.encoder = Transformer(
            **transformers_kwargs,
            **encoder_kwargs,
            norm_layer=norm_layer,
        )

        self.tile_encoder_keys = tile_encoder_keys
        self.embed_dim = self.encoder.embed_dim
        self.out_layer = out_layer % (len(self.encoder.blocks) + 1)
        self.enc_norm = norm_layer(self.embed_dim, **final_norm_kwargs)

        # Input projections for each tile encoder
        self.input_proj_dict = nn.ModuleDict(
            {
                f"input_proj_{key}": ProjectionMlp(
                    in_features=in_dims[i],
                    hidden_features=self.embed_dim,
                    out_features=self.embed_dim,
                )
                for i, key in enumerate(tile_encoder_keys)
            }
        )

    def init_weights(self) -> "TiconBackbone":
        """Initialize model weights."""
        self.apply(_init_weights)
        return self

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        relative_coords: Float[Tensor, "b n 2"],
        tile_encoder_key: str,
    ) -> Float[Tensor, "b n d"]:
        """
        Forward pass through TICON encoder.

        Args:
            x:  Tile embeddings [B, N, D]
            relative_coords:  Tile coordinates [B, N, 2]
            tile_encoder_key:  Which input projection to use

        Returns:
            Contextualized embeddings [B, N, embed_dim]
        """
        # Project input to TICON embedding dimension
        x = self.input_proj_dict[f"input_proj_{tile_encoder_key}"](x)

        # Run through transformer encoder
        encoder_outputs = self.encoder(
            x,
            coords=relative_coords,
            return_layers={self.out_layer},
        )

        # Apply final normalization
        return self.enc_norm(encoder_outputs[self.out_layer])


def _init_weights(m: nn.Module) -> None:
    """Initialize model weights following JAX ViT convention."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_ticon_backbone(
    device: DeviceLikeType = "cuda",
    model_cfg: dict | None = None,
) -> TiconBackbone:
    """Load pretrained TICON backbone from HuggingFace."""
    model_cfg = TICON_MODEL_CFG if model_cfg is None else model_cfg

    # Download checkpoint from HuggingFace
    ckpt_path = hf_hub_download(
        repo_id="varunb/TICON",
        filename="backbone/checkpoint.pth",
        repo_type="model",
    )

    # Create model on meta device (no memory allocation)
    with torch.device("meta"):
        model = TiconBackbone(**model_cfg)

    # Move to target device and initialize weights
    model.to_empty(device=device)
    model.init_weights()

    # Load pretrained weights
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.removeprefix("backbone."): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


## TICON BACKBONE END ##


## TICON ENCODER CLASS ##
class TiconEncoder(Encoder):
    def __init__(
        self,
        device: DeviceLikeType = "cuda",
        precision: torch.dtype = torch.float32,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ticon_model = load_ticon_backbone(device=device)

        super().__init__(
            model=ticon_model,
            identifier=EncoderName.TICON,
            precision=precision,
            required_extractors=list(TILE_EXTRACTOR_TO_TICON.keys()),
        )

        self._device = torch.device(device)
        self._current_extractor = None

    def _prepare_coords(self, coords: CoordsInfo, num_tiles: int) -> Tensor:
        """Prepare coordinates tensor for TICON."""
        if coords is None:
            print("No coords provided, using zeros.")
            return torch.zeros(
                1, num_tiles, 2, device=self._device, dtype=torch.float32
            )
        # CoordsInfo: get relative positions
        if isinstance(coords, CoordsInfo):
            coords_data = coords.coords_um
            if coords.tile_size_um and coords.tile_size_um > 0:
                # converting to grid-indices to get relative positions (is optional only, can be left out)
                coords_data = coords.coords_um / coords.tile_size_um
            else:
                coords_data = coords.coords_um
        else:
            coords_data = coords

        # convert CoordsInfo to tensor
        if not isinstance(coords_data, torch.Tensor):
            coords_data = np.array(coords_data)
            coords_tensor = torch.from_numpy(coords_data)
        else:
            coords_tensor = coords_data

        # adapt dimensions (add batch dim)
        if coords_tensor.dim() == 2:
            coords_tensor = coords_tensor.unsqueeze(0)  # [1, N, 2]
        assert (
            coords_tensor.shape[1] == num_tiles
        )  # number of coords-pairs must match number of tiles
        return coords_tensor.to(self._device, dtype=torch.float32)

    def _generate_slide_embedding(
        self,
        feats: torch.Tensor,
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
        """Generate contextualized slide embedding using TICON."""

        # get extractor from kwargs
        extractor = kwargs.get("extractor")
        if extractor is None:
            raise ValueError("extractor must be provided for TICON encoding")

        # Convert extractor-string to ExtractorName to be sure
        if isinstance(extractor, str):
            extractor = ExtractorName(extractor)

        tile_encoder_key, _ = get_ticon_key(extractor)
        print(f"Using tile extractor: {tile_encoder_key} for ticon")
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # add batch dim
        feats = feats.to(self._device, dtype=torch.float32)

        # get coords from kwargs
        coords_tensor = kwargs.get("coords", None)
        print(
            f"Coords tensor shape: {coords_tensor.shape}"
            if coords_tensor is not None
            else "No coords tensor provided"
        )
        # # check pytorch version for autocast compatibility
        # is_legacy_autocast = "torch.cuda.amp" in autocast.__module__

        # ac_kwargs = {
        #     "enabled": (self._device.type == "cuda"),
        #     "dtype": torch.bfloat16,
        # }
        # # if its the new version: add device_type
        # if not is_legacy_autocast:
        #     ac_kwargs["device_type"] = "cuda"

        # Inference mode only/ without autocast
        with torch.no_grad():
            try:
                contextualized = self.model(
                    x=feats,
                    relative_coords=coords_tensor,
                    tile_encoder_key=tile_encoder_key,
                )
            except RuntimeError as e:
                _logger.error(
                    f"RuntimeError during TICON encoding without autocast: {e}. Retrying with autocast."
                )
                raise e

            # try:
            #     with autocast(**ac_kwargs):
            #         contextualized = self.model(
            #             x=feats,
            #             relative_coords=coords_tensor,
            #             tile_encoder_key=tile_encoder_key,
            #         )
            # except RuntimeError as e:
            #     _logger.error(
            #         f"RuntimeError during TICON encoding with autocast {ac_kwargs}: {e}. Retrying without autocast."
            #     )
            #     contextualized = self.model(
            #         x=feats,
            #         relative_coords=coords_tensor,
            #         tile_encoder_key=tile_encoder_key,
            #     )

        return contextualized.detach().squeeze(0).cpu().numpy()

    # only pseudo-code so TiconEncoder can be instantiated
    def _generate_patient_embedding(
        self,
        feats_list: list[torch.Tensor],
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
        contextualized = [
            self._generate_slide_embedding(feats, device, **kwargs)
            for feats in feats_list
        ]
        return np.concatenate(contextualized, axis=0)

    def encode_slides_(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool = True,
        **kwargs,
    ) -> None:
        if generate_hash:
            encode_dir = f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}"
        else:
            encode_dir = f"{self.identifier}-slide"

        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        h5_files = [f for f in os.listdir(feat_dir) if f.endswith(".h5")]

        for filename in (progress := tqdm(h5_files)):
            h5_path = os.path.join(feat_dir, filename)
            slide_name = Path(filename).name
            progress.set_description(slide_name)

            output_path = (encode_dir / slide_name).with_suffix(".h5")
            if output_path.exists():
                _logger.info(f"Skipping {slide_name}: output exists")
                continue
            #
            try:
                feats, coords = self._validate_and_read_features(h5_path)
            except ValueError as e:
                tqdm.write(s=str(e))
                continue
            try:
                feats, coords, extractor = self._read_h5(h5_path)
            except ValueError as e:
                tqdm.write(str(e))
                continue
            try:
                target_extractor = ExtractorName(extractor)  # str → Enum
            except ValueError:
                target_extractor = extractor  # Schon Enum

            # option to save coords because it is not a classical slide, also set feat_type to tile
            coords_um_np = coords.coords_um
            print(
                f"Coords um shape: {coords_um_np.shape}"
                if coords is not None
                else "No coords found"
            )

            # CoordsInfo -> absolute coords in µm
            if isinstance(coords_um_np, torch.Tensor):
                coords_um_np = coords_um_np.detach().cpu().numpy()
                print(f"Converted coords to numpy array, shape: {coords_um_np.shape}")
            else:
                coords_um_np = np.asarray(coords_um_np)
                print(f"Coords as numpy array, shape: {coords_um_np.shape}")

            slide_embedding = self._generate_slide_embedding(
                feats,
                device,
                coords=self._prepare_coords(coords, feats.shape[0]),
                extractor=target_extractor,
            )

            self._save_features_(
                output_path=output_path,
                feats=slide_embedding,
                feat_type="tile",
                coords=coords_um_np,
                tile_size_um=float(coords.tile_size_um)
                if coords.tile_size_um is not None
                else None,
                tile_size_px=int(coords.tile_size_px)
                if coords.tile_size_px is not None
                else None,
                unit="um",
            )

