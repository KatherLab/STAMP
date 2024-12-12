"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

# TODO implement masking

from typing import Iterable, cast

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor, nn


def feed_forward(
    dim: int,
    hidden_dim: int,
    dropout: float = 0.5,
) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

    def forward(
        self, x: Float[Tensor, "batch sequence feature"]
    ) -> Float[Tensor, "batch sequence feature"]:
        x = self.norm(x)
        attn_output, _ = self.mhsa(x, x, x, need_weights=False)
        return attn_output


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim,
                            heads=heads,
                            dropout=dropout,
                        ),
                        feed_forward(
                            dim,
                            mlp_dim,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x: Float[Tensor, "batch sequence feature"]
    ) -> Float[Tensor, "batch sequence feature"]:
        for attn, ff in cast(Iterable[tuple[nn.Module, nn.Module]], self.layers):
            x_attn = attn(x)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim_output: int,
        dim_input: int,
        dim_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(dim_model))

        self.project_features = nn.Sequential(
            nn.Linear(dim_input, dim_model, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.transformer = Transformer(
            dim=dim_model,
            depth=n_layers,
            heads=n_heads,
            mlp_dim=dim_feedforward,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim_model, dim_output))

    def forward(
        self, bags: Float[Tensor, "batch tile feature"]
    ) -> Float[Tensor, "batch logit"]:
        batch_size, _n_tiles, _n_features = bags.shape

        # map input sequence to latent space of TransMIL
        bags = self.project_features(bags)

        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat((cls_tokens, bags), dim=1)

        bags = self.transformer(bags)

        bags = bags[:, 0]  # only take class token

        return self.mlp_head(bags)
