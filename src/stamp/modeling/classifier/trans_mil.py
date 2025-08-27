"""
Code adapted from:
https://github.com/szc19990412/TransMIL/blob/main/models/TransMIL.py
"""

from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, einsum, nn

# --- Helpers ---


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x: Tensor, iters: int = 6) -> Tensor:
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I_mat = torch.eye(x.shape[-1], device=device)
    I_mat = rearrange(I_mat, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I_mat - (xz @ (15 * I_mat - (xz @ (7 * I_mat - xz)))))

    return z


# --- Nystrom Attention ---


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head**-0.5

        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch n dim"],
        mask: Bool[Tensor, "batch n"] | None = None,
        return_attn: bool = False,
        return_attn_matrices: bool = False,
    ) -> Float[Tensor, "batch n dim"]:
        b, n, _ = x.shape
        h, m, iters, eps = (
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )

        # Pad sequence to be divisible by landmarks
        remainder = n % m
        if remainder > 0:
            pad_len = m - remainder
            x = F.pad(x, (0, 0, pad_len, 0), value=0)
            if mask is not None:
                mask = F.pad(mask, (pad_len, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if mask is not None:
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        len = ceil(n / m)
        q_landmarks = reduce(q, "... (n l) d -> ... n d", "sum", l=len)
        k_landmarks = reduce(k, "... (n l) d -> ... n d", "sum", l=len)

        divisor = len
        if mask is not None:
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=len)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        sim1 = einsum("... i d, ... j d -> ... i j", q, k_landmarks)
        sim2 = einsum("... i d, ... j d -> ... i j", q_landmarks, k_landmarks)
        sim3 = einsum("... i d, ... j d -> ... i j", q_landmarks, k)

        if mask is not None:
            mask_val = -torch.finfo(q.dtype).max
            sim1.masked_fill_(
                ~(mask[..., None] * mask_landmarks[..., None, :]),  # type: ignore
                mask_val,
            )
            sim2.masked_fill_(
                ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]),  # type: ignore
                mask_val,
            )
            sim3.masked_fill_(
                ~(mask_landmarks[..., None] * mask[..., None, :]),  # type: ignore
                mask_val,
            )

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn_matrices:
            return out, (attn1, attn2_inv, attn3)  # type: ignore
        elif return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn  # type: ignore

        return out


# --- Transformer blocks ---


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        attn_values_residual: bool = True,
        attn_values_residual_conv_kernel: int = 33,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NystromAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                num_landmarks=num_landmarks,
                                pinv_iterations=pinv_iterations,
                                residual=attn_values_residual,
                                residual_conv_kernel=attn_values_residual_conv_kernel,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
        mask: Bool[Tensor, "batch sequence"] | None = None,
    ) -> Float[Tensor, "batch sequence dim"]:
        for attn, ff in self.layers:  # type: ignore
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch tokens dim"]
    ) -> Float[Tensor, "batch tokens dim"]:
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch tokens dim"], H: int, W: int
    ) -> Float[Tensor, "batch tokens dim"]:
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim_output: int, dim_input: int, dim_hidden: int):
        super().__init__()
        self.pos_layer = PPEG(dim=dim_hidden)
        self._fc1 = nn.Sequential(nn.Linear(dim_input, dim_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        self.n_classes = dim_output
        self.layer1 = TransLayer(dim=dim_hidden)
        self.layer2 = TransLayer(dim=dim_hidden)
        self.norm = nn.LayerNorm(dim_hidden)
        self._fc2 = nn.Linear(dim_hidden, self.n_classes)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, h: Float[Tensor, "batch tiles dim_input"], **kwargs
    ) -> Float[Tensor, "batch n_classes"]:
        # Project to lower dim
        h = self._fc1(h)  # [B, n, C]

        # Pad to square for reshaping
        H = h.shape[1]
        _H = _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, C]

        # Add class token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer → Positional Encoding → Transformer
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        # Class token output
        h = self.norm(h)[:, 0]

        # Classifier
        logits = self._fc2(h)  # [B, n_classes]
        return logits
