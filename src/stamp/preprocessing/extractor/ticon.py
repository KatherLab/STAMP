"""
This file contains code adapted from:
TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning
https://github.com/cvlab-stonybrook/TICON
"""

import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms

from stamp.preprocessing.extractor import Extractor

try:
    import timm
    from torchvision import transforms
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "h_optimus_1 dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[h_optimus_1]'`"
    ) from e


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: int | float | None = (16 / 3),
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            assert mlp_ratio is not None
            hidden_features = int(in_features * mlp_ratio)
        else:
            assert mlp_ratio is None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features // 2, in_features, bias=bias)

    def forward(self, x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b d"]:
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        return x


class ProjectionMlp(nn.Module):
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

    def forward(self, x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b d"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def scaled_dot_product_attention_custom(
    query,
    key,
    value,
    attn_bias=None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))  # pyright: ignore[reportOptionalMemberAccess]
        attn_bias.to(query.dtype)  # pyright: ignore[reportOptionalMemberAccess]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))  # pyright: ignore[reportOptionalMemberAccess]
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        context_dim: int | None = None,
        # rope_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        # self.rope = Rope(dim=head_dim, **rope_kwargs)
        slopes = torch.Tensor(get_slopes(num_heads))
        self.slopes = slopes[
            None, :, None, None
        ]  # einops.rearrange(slopes, 'b -> 1 b 1 1')

    def forward(
        self,
        x: Float[Tensor, "b n_q d"],
        coords: Float[Tensor, "b n_q 2"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> Float[Tensor, "b n_q d"]:
        if context is None or context_coords is None:
            context = x
            context_coords = coords
        b, n_q, d = x.shape
        b, n_k, _ = context.shape
        h = self.num_heads

        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)

        corrds_expanded = coords.unsqueeze(2).expand(
            -1, -1, n_k, -1
        )  # (b, m, d) -> (b, m, 1, d) -> (b, m, n, d)
        context_coords_expanded = context_coords.unsqueeze(1).expand(-1, n_q, -1, -1)
        euclid_dist = torch.sqrt(
            torch.sum((corrds_expanded - context_coords_expanded) ** 2, dim=-1)
        )
        self.slopes = self.slopes.to(x.device)
        attn_bias = (-1) * self.slopes * euclid_dist[:, None, :, :]

        # x = F.scaled_dot_product_attention(q, k, v)
        x = scaled_dot_product_attention_custom(q, k, v, attn_bias=attn_bias)
        x = x.transpose(1, 2).reshape([b, n_q, d])
        x = self.proj(x)
        return x


class NaiveResidual(nn.Module):
    def __init__(
        self,
        drop_prob: float | int,
        norm: nn.Module,
        fn: nn.Module,
        gamma: nn.Parameter,
    ):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob
        self.gamma = gamma

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n d"]:
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.gamma is not None:
            if self.keep_prob == 1.0 or not self.training:
                return x + self.gamma * fn_out
            mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[
                :, None, None
            ]
            return x + self.gamma * fn_out * mask / self.keep_prob
        else:
            if self.keep_prob == 1.0 or not self.training:
                return x + fn_out
            mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[
                :, None, None
            ]
            return x + fn_out * mask / self.keep_prob


class EfficientResidual(NaiveResidual):
    def forward(
        self,
        x: Float[Tensor, "b n d"],
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n d"]:
        if self.keep_prob == 1.0 or not self.training:
            if self.gamma is not None:
                return x + self.gamma * self.fn(self.norm(x), **kwargs)
            else:
                return x + self.fn(self.norm(x), **kwargs)

        b, _, _ = x.shape
        n_keep = max(int(b * self.keep_prob), 1)
        indices = torch.randperm(b, device=x.device)[:n_keep]
        for k, v in kwargs.items():
            if v is not None:
                kwargs[k] = v[indices]
        if self.gamma is not None:
            return torch.index_add(
                x,
                dim=0,
                source=self.gamma * self.fn(self.norm(x[indices]), **kwargs),
                index=indices,
                alpha=b / n_keep,
            )
        else:
            return torch.index_add(
                x,
                dim=0,
                source=self.fn(self.norm(x[indices]), **kwargs),
                index=indices,
                alpha=b / n_keep,
            )


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float | int,
        norm_layer: Callable[[int], nn.Module],
        context_dim: int | None,
        drop_path_type: str = "efficient",
        layer_scale: int = True,
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        residual_module = {
            "naive": NaiveResidual,
            "efficient": EfficientResidual,
        }[drop_path_type]

        self.layer_scale = layer_scale
        if layer_scale:
            gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        else:
            gamma1 = None
            gamma2 = None

        self.residual1 = residual_module(
            drop_path,
            norm_layer(dim),
            Attention(
                dim,
                context_dim=context_dim,
                **attn_kwargs,
            ),
            gamma1,
        )
        self.residual2 = residual_module(
            drop_path, norm_layer(dim), Mlp(in_features=dim), gamma2
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        coords: Float[Tensor, "b n 2"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> Float[Tensor, "b n d"]:
        x = self.residual1(
            x,
            context=context,
            coords=coords,
            context_coords=context_coords,
        )
        x = self.residual2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        norm_layer: Callable[[int], nn.Module],
        depth: int,
        drop_path_rate: float | int,
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
                for i in range(depth)
            ],
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        return_layers: set[int],
        contexts: list[Float[Tensor, "b n_k d_k"]] | None = None,
        coords: Float[Tensor, "b n 2"] | None = None,
        context_coords: Float[Tensor, "b n_k 2"] | None = None,
    ) -> dict[int, Float[Tensor, "b n d"]]:
        outputs = {}
        if 0 in return_layers:
            outputs[0] = x
        for blk_idx, blk in enumerate(self.blocks):
            context = contexts[blk_idx] if contexts is not None else None
            x = blk(
                x,
                context=context,
                coords=coords,
                context_coords=context_coords,
            )
            if blk_idx + 1 in return_layers:
                outputs[blk_idx + 1] = x
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        in_dims: list = [],
        tile_encoder_keys: list = [],
        norm_layer_type: str = "LayerNorm",
        transformers_kwargs: Mapping[str, Any] = {},
        encoder_kwargs: Mapping[str, Any] = {},
        decoder_kwargs: Mapping[str, Any] = {},
        norm_layer_kwargs: Mapping[str, Any] = {"eps": 1e-5},
        final_norm_kwargs: Mapping[str, Any] = {"elementwise_affine": True},
        out_layer: int = -1,
        num_decoders: int = 1,
        decoder_out_dims: list = [],
    ):
        super().__init__()
        self.patch_size = patch_size

        norm_layer: Callable[[int], nn.Module] = partial(
            getattr(torch.nn, norm_layer_type), **norm_layer_kwargs
        )

        self.encoder = Transformer(
            **transformers_kwargs,
            **encoder_kwargs,
            norm_layer=norm_layer,
        )

        self.tile_encoder_keys = tile_encoder_keys
        self.embed_dim = self.encoder.embed_dim
        self.n_blocks = len(self.encoder.blocks)
        self.out_layer = out_layer % (len(self.encoder.blocks) + 1)
        self.enc_norm = norm_layer(self.embed_dim, **final_norm_kwargs)
        self.num_decoders = num_decoders
        self.decoder_out_dims = decoder_out_dims

        self.decoder_dict = nn.ModuleDict({})
        self.mask_dict = nn.ParameterDict({})
        self.input_proj_dict = nn.ModuleDict({})
        self.output_proj_dict = nn.ModuleDict({})

        for i in range(len(in_dims)):
            self.input_proj_dict[f"input_proj_{self.tile_encoder_keys[i]}"] = (
                ProjectionMlp(
                    in_features=in_dims[i],
                    hidden_features=self.encoder.embed_dim,
                    out_features=self.encoder.embed_dim,
                )
            )

        for i in range(self.num_decoders):
            self.decoder_dict[f"decoder_{i}"] = nn.ModuleDict({})
            self.decoder_dict[f"decoder_{i}"]["transformer"] = Transformer(  # pyright: ignore[reportIndexIssue]
                **transformers_kwargs,
                **decoder_kwargs,
                context_dim=self.encoder.embed_dim,
                norm_layer=norm_layer,
            )

            self.decoder_dict[f"decoder_{i}"]["norm"] = norm_layer(  # pyright: ignore[reportIndexIssue]
                self.decoder_dict[f"decoder_{i}"]["transformer"].embed_dim,  # pyright: ignore[reportIndexIssue]
                **final_norm_kwargs,
            )
            self.mask_dict[f"mask_token_{i}"] = nn.Parameter(
                torch.empty(
                    1,
                    self.decoder_dict[f"decoder_{i}"]["transformer"].embed_dim,  # pyright: ignore[reportIndexIssue]
                )
            )

        for i in range(len(self.decoder_out_dims)):
            self.output_proj_dict[f"output_proj_{self.tile_encoder_keys[i]}"] = (
                ProjectionMlp(
                    in_features=self.encoder.embed_dim,
                    hidden_features=self.encoder.embed_dim,
                    out_features=self.decoder_out_dims[i],
                )
            )

        assert self.num_decoders <= 1

    def init_weights(self):
        for mask_key in self.mask_dict.keys():
            nn.init.normal_(self.mask_dict[mask_key], std=0.02)
        self.apply(_init_weights)
        return self

    def forward_features(
        self,
        x: Float[Tensor, "b n d"],
        relative_coords: Float[Tensor, "b n 2"] | None,
        predict_coords: Float[Tensor, "b n 2"] | None,
        enc_layer: int,
        dec_layer: int | None,
        tile_encoder_key: str | None,
    ) -> tuple[Float[Tensor, "b n d"], dict | None]:
        b, _, _ = x.shape

        # these are the layers we need
        enc_layers = {enc_layer}
        if dec_layer is not None:
            enc_layers.add(len(self.encoder.blocks))

        # encoder fwd
        coords_enc = relative_coords
        coords_dec = predict_coords
        x = self.input_proj_dict[f"input_proj_{tile_encoder_key}"](x)
        encoder_outputs = self.encoder(x, coords=coords_enc, return_layers=enc_layers)
        encoder_outputs = {k: self.enc_norm(v) for k, v in encoder_outputs.items()}

        # decoder fwd
        if dec_layer is not None:
            dec_final_output = {}
            assert self.num_decoders == 1
            for dec_index in range(self.num_decoders):
                decoder_outputs = self.decoder_dict[
                    f"decoder_{dec_index}"
                ][  # pyright: ignore[reportIndexIssue]
                    "transformer"
                ](
                    self.mask_dict[f"mask_token_{dec_index}"][None].expand(
                        *coords_dec.shape[:2],  # pyright: ignore[reportOptionalMemberAccess]
                        -1,  # pyright: ignore[reportOptionalMemberAccess]
                    ),
                    contexts=[encoder_outputs[len(self.encoder.blocks)]]
                    * self.decoder_dict[f"decoder_{dec_index}"]["transformer"].n_blocks,  # pyright: ignore[reportIndexIssue]
                    coords=coords_dec,
                    context_coords=coords_enc,
                    return_layers={dec_layer},
                )
                dec_output = self.decoder_dict[f"decoder_{dec_index}"]["norm"](  # pyright: ignore[reportIndexIssue]
                    decoder_outputs[dec_layer]
                )

                for out_index in range(len(self.decoder_out_dims)):
                    dec_final_output[self.tile_encoder_keys[out_index]] = (
                        self.output_proj_dict[
                            f"output_proj_{self.tile_encoder_keys[out_index]}"
                        ](dec_output)
                    )
        else:
            dec_final_output = None
        enc_output = encoder_outputs[enc_layer]
        return (enc_output, dec_final_output)

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        relative_coords: Float[Tensor, "b n 2"] | None = None,
        tile_encoder_key: str | None = None,
    ) -> Float[Tensor, "b n d"]:
        # print("Input feature range", torch.min(x), torch.max(x))
        # print("Input coords range", torch.min(relative_coords), torch.max(relative_coords))
        enc_output, dec_output = self.forward_features(
            x,
            relative_coords=relative_coords,
            predict_coords=None,
            enc_layer=self.out_layer,
            dec_layer=None,
            tile_encoder_key=tile_encoder_key,
        )

        # print(torch.min(enc_output), torch.max(enc_output))
        return enc_output


# from https://github.com/facebookresearch/mae/blob/main/models_mae.py
def _init_weights(m: nn.Module, xavier_gain=1) -> None:
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm | nn.RMSNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)  # pyright: ignore[reportArgumentType]
    if hasattr(m, "_device_weight_init"):
        m._device_weight_init()  # pyright: ignore[reportCallIssue]


def load_ticon(device: str = "cuda") -> nn.Module:
    model_cfg = {
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
            "conchv15",
            "hoptimus1",
            "uni2h",
            "gigapath",
            "virchow2",
        ],
        "num_decoders": 1,
        "decoder_out_dims": [768, 1536, 1536, 1536, 1280],
    }

    ckpt = hf_hub_download(
        repo_id="varunb/TICON",
        filename="backbone/checkpoint.pth",
        repo_type="model",
    )

    with torch.device("meta"):
        model = EncoderDecoder(**model_cfg)

    model.to_empty(device=device)
    model.init_weights()

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {
        k.removeprefix("backbone."): v
        for k, v in sd.items()
        if k.startswith("backbone.")
    }

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


class HOptimusTICON(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # ----------------------------
        # Stage 1: H-OptimUS
        # ----------------------------
        self.tile_encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )

        # ----------------------------
        # Stage 2: TICON
        # ----------------------------
        ticon_cfg = {
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
                "conchv15",
                "hoptimus1",
                "uni2h",
                "gigapath",
                "virchow2",
            ],
            "num_decoders": 1,
            "decoder_out_dims": [768, 1536, 1536, 1536, 1280],
        }

        with torch.device("meta"):
            self.ticon = EncoderDecoder(**ticon_cfg)

        self.ticon.to_empty(device=device)
        self.ticon.init_weights()

        ckpt = hf_hub_download(
            repo_id="varunb/TICON",
            filename="backbone/checkpoint.pth",
            repo_type="model",
        )

        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = {
            k.removeprefix("backbone."): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }
        self.ticon.load_state_dict(sd, strict=False)

        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224] (CPU or CUDA)
        """
        x = x.to(self.device, non_blocking=True)

        # H-Optimus_1
        emb = self.tile_encoder(x)  # [B, 1536]
        emb = emb.unsqueeze(1)  # [B, 1, 1536]
        # TICON
        # single-tile → zero relative coords
        coords = torch.zeros(
            emb.size(0),
            1,
            2,
            device=self.device,
            dtype=torch.float32,
        )

        out = self.ticon(
            x=emb,
            relative_coords=coords,
            tile_encoder_key="hoptimus1",
        )

        return out.squeeze(1)  # [B, 1536]


def ticon(device: str = "cuda") -> Extractor[nn.Module]:
    model = HOptimusTICON(torch.device(device))

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

    return Extractor(
        model=model,
        transform=transform,
        identifier="ticon",
    )
