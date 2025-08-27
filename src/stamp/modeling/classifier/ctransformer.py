import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

__author__ = "Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2025 MMinh Duc Nguyen"
__license__ = "MIT"


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
            nn.Linear(2, 32), nn.ReLU(inplace=True), nn.Linear(32, num_heads)
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


### Coordinates bias Attention approach
class CTransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = CoordAttention(dim=dim, num_heads=8)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        coords: Float[Tensor, "batch tokens 2"],
    ) -> Float[Tensor, "batch tokens dim"]:
        x = x + self.attn(self.norm(x), coords)
        return x


class CTransformer(nn.Module):
    def __init__(self, dim_output: int, dim_input: int, dim_hidden: int):
        super(CTransformer, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(dim_input, dim_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.n_classes = dim_output
        self.layer1 = CTransLayer(dim=dim_hidden)
        self.layer2 = CTransLayer(dim=dim_hidden)
        self.norm = nn.LayerNorm(dim_hidden)
        self._fc2 = nn.Linear(dim_hidden, self.n_classes)

    def forward(
        self,
        h: Float[Tensor, "batch tiles dim_input"],
        coords: Float[Tensor, "batch tile 2"],
        **kwargs,
    ) -> Tensor:
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

        # Translayer x2
        h = self.layer2(h, coords)  # [B, N, dim_hidden]

        # cls_token
        h = self.norm(h)[:, 0]

        # predict
        logits = self._fc2(h)  # [B, n_classes]
        return logits
