import math

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn

__author__ = "Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2025 MMinh Duc Nguyen"
__license__ = "MIT"


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "seq batch dim"]
    ) -> Float[Tensor, "seq batch dim"]:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "seq batch dim"]
    ) -> Float[Tensor, "seq batch dim"]:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )


class CoordAttention(nn.Module):
    def __init__(
        self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # MLP for continuous coordinate-based relative positional bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(inplace=True), nn.Linear(512, num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        coords: Float[Tensor, "batch tokens 2"],
        mask: Bool[Tensor, "batch tokens tokens"] | None = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        """
        Args:
            x: (B, N, C) - input features
            coords: (B, N, 2) - real coordinates (e.g., WSI patch centers)
            mask: Optional attention mask (B, N, N)
        Returns:
            Output: (B, N, C)
        """
        B, N, C = x.shape
        # Compute QKV
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)

        # Coordinate difference and bias computation
        rel_coords = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 2)
        rel_coords = rel_coords / (
            rel_coords.norm(dim=-1, keepdim=True) + 1e-6
        )  # normalize direction
        bias = self.cpb_mlp(rel_coords)  # (B, N, N, num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)

        attn = attn + bias

        # Optional attention mask
        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # (B, 1, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        query: Float[Tensor, "batch tokens dim"],
        key: Float[Tensor, "batch tokens dim"],
        value: Float[Tensor, "batch tokens dim"],
    ) -> Float[Tensor, "batch tokens dim"]:
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.layernorm1(query + attn_output)

        # Feed-forward
        ffn_output = self.ffn(query)
        query = self.layernorm2(query + ffn_output)
        return query


### Coordinates bias Attention approach
class CTransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = CoordAttention(dim=512, num_heads=8)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        coords: Float[Tensor, "batch tokens 2"],
    ) -> Float[Tensor, "batch tokens dim"]:
        x = x + self.attn(self.norm(x), coords)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class CTransformer(nn.Module):
    def __init__(self, dim_output: int, dim_input: int, dim_hidden: int):
        super(CTransformer, self).__init__()
        self.pos_layer = PPEG(dim=dim_hidden)
        self._fc1 = nn.Sequential(nn.Linear(dim_input, dim_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.n_classes = dim_output
        self.layer1 = CTransLayer(dim=dim_hidden)
        self.layer2 = CTransLayer(dim=dim_hidden)
        self.norm = nn.LayerNorm(dim_hidden)
        self._fc2 = nn.Linear(dim_hidden, self.n_classes)

    def forward(self, h, coords, *args, **kwargs) -> Tensor:
        h = self._fc1(h)  # [B, n, dim_hidden]

        # pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, dim_hidden]

        # Pad coords similarly?
        coords = torch.cat([coords, coords[:, :add_length, :]], dim=1)

        # cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # Add the [CLS] token coordinates (zero predefined)
        cls_coords = torch.zeros(B, 1, 2).cuda()
        coords = torch.cat((cls_coords, coords), dim=1)

        # Translayer x1
        h = self.layer1(h, coords)  # [B, N, dim_hidden]

        # # PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, dim_hidden]

        # Translayer x2
        h = self.layer2(h, coords)  # [B, N, dim_hidden]

        # cls_token
        h = self.norm(h)[:, 0]

        # predict
        logits = self._fc2(h)  # [B, n_classes]
        return logits
