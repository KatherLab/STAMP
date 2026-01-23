"""
Kronos feature extractor wrapper for STAMP.
Integrates the pretrained Kronos ViT-S16 model into the STAMP Extractor interface.
"""

import json
import logging
import math
import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from torchvision import transforms

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

# ----------------------------------------------------------------------
# Kronos Wrapper
# ----------------------------------------------------------------------


class KRONOS(nn.Module):
    """A wrapper around Kronos to make it compatible with STAMP Extractor.

    This ensures that Kronos returns only patch_features (CLS token embeddings)
    instead of a tuple of three outputs.
    """

    def __init__(self, kronos_model: nn.Module):
        super().__init__()
        self.kronos_model = kronos_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, num_markers, H, W)

        Returns:
            patch_features: Tensor of shape (B, D)
            marker_features: Tensor of shape (B, num_markers, D)
        """
        B, num_markers, _, _ = x.shape
        marker_ids = [torch.arange(num_markers, device=x.device) for _ in range(B)]
        patch_features, marker_features, _ = self.kronos_model(x, marker_ids=marker_ids)
        return marker_features


def kronos() -> Extractor:
    """Return Kronos ViT-S16 extractor compatible with STAMP."""
    # Load Kronos pretrained model
    model, _, _ = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/KRONOS",
        cfg={
            "model_type": "vits16",
            "token_overlap": True,
        },
    )

    # Wrap Kronos to return only patch_features
    model = KRONOS(model)

    # Define the same type of transform as other extractors
    transform = transforms.Compose([])

    return Extractor(
        model=model,
        transform=transform,
        identifier=ExtractorName.KRONOS,  # add to ExtractorName Enum
    )


def get_model_config(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> dict:
    """
    Generate or load configurations for the Kronos model.

    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.
        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        dict: Configuration dictionary containing model settings.
    """
    if cfg is not None:
        # Use provided configuration dictionary
        config = cfg
    elif cfg_path is not None:
        # Load configuration from file
        with open(cfg_path) as f:
            config = json.load(f)
    elif checkpoint_path is not None and checkpoint_path.startswith("hf_hub:"):
        from huggingface_hub import hf_hub_download

        # Download config.json from Hugging Face Hub
        cfg_path = hf_hub_download(
            checkpoint_path[len("hf_hub:") :],
            cache_dir=cache_dir,
            filename="config.json",
            token=hf_auth_token,
        )

        # Load configuration from downloaded file
        with open(cfg_path) as f:
            config = json.load(f)
    else:
        # Default configuration if none provided
        print(
            "No configuration provided. A vit_small model with no token overlap is initialized."
        )
        config = {"model_type": "vits16", "token_overlap": False}

    return config


def create_model(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates a Kronos model defined with configuration details in config.

    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.
        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """

    # Get model configuration
    config = get_model_config(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=cfg,
    )

    # Default arguments for vision transformer
    vit_kwargs = dict(
        img_size=224,
        patch_size=16,
        stride_size=16,
        num_markers=512,
        init_values=1.0e-05,
        ffn_layer="mlp",
        block_chunks=4,
        num_register_tokens=16,
    )
    if config["model_type"] in ["vits16", "vitl16"] and config["token_overlap"]:
        # Adjust stride size if token overlap is enabled
        vit_kwargs["stride_size"] = 8

    model = None
    embedding_dim = None
    if config["model_type"] == "vits16":
        # Create small vision transformer model
        model = vit_small(**vit_kwargs)  # pyright: ignore[reportArgumentType]
        embedding_dim = 384
    elif config["model_type"] == "vitl16":
        # Create large vision transformer model
        model = vit_large(**vit_kwargs)  # pyright: ignore[reportArgumentType]
        embedding_dim = 1024
    else:
        # Raise error for unsupported model type
        raise ValueError(f"Unsupported model type: {config['model_type']}")

    return model, torch.float32, embedding_dim


def create_model_from_pretrained(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates and loads a pretrained Kronos model from the given checkpoint.

    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.

        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """

    # Get model configuration
    config = get_model_config(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=cfg,
    )

    # Create model and retrieve precision and embedding dimension
    model, precision, embedding_dim = create_model(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=config,
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.startswith("hf_hub:"):
        from huggingface_hub import hf_hub_download

        if config["model_type"] == "vits16":
            checkpoint_filename = "kronos_vits16_model.pt"
        elif config["model_type"] == "vitl16":
            checkpoint_filename = "kronos_vitl16_model.pt"
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")

        # Download checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            checkpoint_path[len("hf_hub:") :],
            cache_dir=cache_dir,
            filename=checkpoint_filename,
            token=hf_auth_token,
        )

    # Load the state dictionary, removing specific prefixes and entries
    state_dict = torch.load(checkpoint_path, map_location="cpu")  # pyright: ignore[reportArgumentType]
    state_dict = state_dict["teacher"]
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if "dino_head" not in k}

    # Load the model state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"\033[92mLoaded model weights from {checkpoint_path}\033[0m")

    return model, precision, embedding_dim


# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return attn

        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)  # pyright: ignore[reportPossiblyUnboundVariable]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)  # pyright: ignore[reportPossiblyUnboundVariable]
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, index_select_cat, scaled_index_add

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        # Add this 2 lines
        if return_attention:
            return self.attn(self.norm1(x), return_attn=True)

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
        )
    else:
        x_plus_residual = scaled_index_add(  # pyright: ignore[reportPossiblyUnboundVariable]
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = (
        [b.shape[0] for b in branges]
        if branges is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)  # pyright: ignore[reportPossiblyUnboundVariable]
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(  # pyright: ignore[reportPossiblyUnboundVariable]
            1, -1, x_list[0].shape[-1]
        )  # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [
        get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
        x_list, branges, residual_list, residual_scale_factors
    ):
        outputs.append(
            add_residual(
                x, brange, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs  # pyright: ignore[reportReturnType]


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )  # pyright: ignore[reportAssignmentType]
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )  # pyright: ignore[reportAssignmentType]
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, return_attention=False):  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, return_attention)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)  # pyright: ignore[reportCallIssue]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]  # pyright: ignore[reportArgumentType]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))  # pyright: ignore[reportArgumentType]
        layers.append(nn.GELU())  # pyright: ignore[reportArgumentType]
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))  # pyright: ignore[reportArgumentType]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))  # pyright: ignore[reportArgumentType]
            layers.append(nn.GELU())  # pyright: ignore[reportArgumentType]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))  # pyright: ignore[reportArgumentType]
        return nn.Sequential(*layers)


# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)  # pyright: ignore[reportArgumentType]


# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        stride_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        stride_HW = make_2tuple(stride_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=stride_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, (
            f"Input image height {H} is not a multiple of patch height {patch_H}"
        )
        assert W % patch_W == 0, (
            f"Input image width {W} is not a multiple of patch width: {patch_W}"
        )

        # x = self.proj(x)  # B C H W
        patch_embeddings = []
        for i in range(C):
            embed = self.proj(x[:, i, :, :].unsqueeze(1))
            patch_embeddings.append(embed.flatten(2).transpose(1, 2))

        x = torch.cat(patch_embeddings, dim=1)
        x = self.norm(x)
        if not self.flatten_embedding:
            assert self.flatten_embedding, (
                "flatten_embedding=False not supported. Check the implementation of PatchEmbed."
            )
            H, W = embed.size(2), embed.size(3)  # pyright: ignore[reportPossiblyUnboundVariable]
            x_ = []
            for i in range(C):
                x_.append(
                    x[:, i * H * W : (i + 1) * H * W, :].reshape(
                        -1, H, W, self.embed_dim
                    )
                )  # B H W C for each marker
            x = x_  # pyright: ignore[reportAssignmentType]
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,  # pyright: ignore[reportArgumentType]
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (SwiGLU)")
    else:
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):  # pyright: ignore[reportGeneralTypeIssues]
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,  # pyright: ignore[reportArgumentType]
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


logger = logging.getLogger("dinov2")


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


# --------------------------------------------------------
# 1D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
def get_1d_sincos_marker_embed(embed_dim, max_marker_id, cls_token=False):
    """
    max_marker_id: marker ids length
    return:
    marker_embed: [max_marker_id, embed_dim] or [1+max_marker_id, embed_dim] (w/ or w/o cls_token)
    """
    ids = np.arange(max_marker_id, dtype=float)
    marker_embed = get_1d_sincos_marker_embed_from_grid(embed_dim, ids)
    if cls_token:
        marker_embed = np.concatenate([np.zeros([1, embed_dim]), marker_embed], axis=0)
    return marker_embed


def get_1d_sincos_marker_embed_from_grid(embed_dim, ids):
    """
    embed_dim: output dimension for each marker
    ids: a list of marker ids to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    ids = ids.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", ids, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, return_attention=False):
        for b in self:
            if isinstance(b, Block):
                x = b(x, return_attention)
                break
            else:
                x = b(x)

        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        num_markers=512,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=1,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # -- modality embedding
        self.marker_embed = get_1d_sincos_marker_embed(
            embed_dim, num_markers, cls_token=False
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h, npatch):
        # h0 = h // self.patch_size
        # w0 = w // self.patch_size

        h0_stride = int(np.sqrt(npatch))
        w0_stride = int(np.sqrt(npatch))

        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        # w0 = w // self.patch_size
        # h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0_stride + self.interpolate_offset) / M
            sy = float(h0_stride + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0_stride, h0_stride)

        # patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        # patch_pos_embed = nn.functional.interpolate(patch_pos_embed, mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0_stride, h0_stride) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        # patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None, marker_ids=None):
        B, num_marker, w, h = x.shape
        x = self.patch_embed(x)
        num_patches = int(x.shape[1] / num_marker)

        # selecting marker embeddings based on marker_index
        assert marker_ids is not None, "marker_ids should be provided"
        marker_embed = (
            torch.from_numpy(self.marker_embed)
            .float()
            .unsqueeze(0)
            .to(device=x.device, dtype=x.dtype)
        )
        marker_embed = marker_embed.repeat(B, 1, 1)
        marker_embed = apply_masks(marker_embed, marker_ids)
        marker_embed = torch.repeat_interleave(marker_embed, num_patches, 1)

        # adding selected marker embeddings to patch embeddings
        x = x + marker_embed

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        pos_embed = self.interpolate_pos_encoding(x, w, h, num_patches)
        pos_embed = torch.cat(
            (
                pos_embed[:, 0, :].unsqueeze(0),
                pos_embed[:, 1:, :].repeat(1, num_marker, 1),
            ),
            dim=1,
        )
        x = x + pos_embed

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list, marker_ids_list):
        x = [
            self.prepare_tokens_with_masks(x, masks, marker_ids)
            for x, masks, marker_ids in zip(x_list, masks_list, marker_ids_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, marker_ids=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks, marker_ids)

        x = self.prepare_tokens_with_masks(x, masks, marker_ids)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, masks=None, marker_ids=None, is_training=False):
        if marker_ids is None:
            marker_ids = [
                torch.tensor([i + 4 for i in range(x.shape[1])], device=x.device)
                for _ in range(x.shape[0])
            ]
        ret = self.forward_features(x, masks, marker_ids)
        if is_training:
            return ret
        else:
            B, num_marker, w, h = x.shape
            tokens_per_row = len(
                [i for i in range(0, h - self.patch_size + 1, self.stride_size)]
            )
            tokens_per_col = len(
                [i for i in range(0, w - self.patch_size + 1, self.stride_size)]
            )

            patch_features = ret["x_norm_clstoken"]  # type: ignore
            patch_token_features = ret["x_norm_patchtokens"].reshape(  # type: ignore
                B, num_marker, tokens_per_row, tokens_per_col, self.embed_dim
            )
            patch_marker_features = torch.mean(
                torch.mean(patch_token_features, dim=-2), dim=-2
            )

            return patch_features, patch_marker_features, patch_token_features

    def _get_intermediate_layers_not_chunked(self, x, marker_ids, n=1):
        x = self.prepare_tokens_with_masks(x, masks=None, marker_ids=marker_ids)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def _get_intermediate_layers_chunked(self, x, marker_ids, n=1):
        x = self.prepare_tokens_with_masks(x, masks=None, marker_ids=marker_ids)
        output, i, total_block_len = [], 0, len(self.blocks[-1])  # pyright: ignore[reportArgumentType]
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # pyright: ignore[reportIndexIssue] # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        marker_ids,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, marker_ids, n)  # pyright: ignore[reportArgumentType]
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, marker_ids, n)  # pyright: ignore[reportArgumentType]
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))  # pyright: ignore[reportReturnType]
        return tuple(outputs)

    def get_last_self_attention(self, x, masks=None, marker_ids=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks, marker_ids)

        x = self.prepare_tokens_with_masks(x, masks, marker_ids)

        # Run through model, at the last block just return the attention.
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),  # pyright: ignore[reportArgumentType]
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),  # pyright: ignore[reportArgumentType]
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),  # pyright: ignore[reportArgumentType]
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),  # pyright: ignore[reportArgumentType]
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
