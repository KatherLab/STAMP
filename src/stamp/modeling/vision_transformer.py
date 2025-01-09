"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

from collections.abc import Iterable
from typing import cast

import torch
from beartype import beartype
from einops import repeat
from jaxtyping import Bool, Float, jaxtyped
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

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
    ) -> Float[Tensor, "batch sequence proj_feature"]:
        """
        Args:
            attn_mask:
                Which of the features to ignore during self-attention.
                `attn_mask[b,q,k] == False` means that
                query `q` of batch `b` can attend to key `k`.
                If `attn_mask` is `None`, all tokens can attend to all others.
        """
        x = self.norm(x)
        attn_output, _ = self.mhsa(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=(
                attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                if attn_mask is not None
                else None
            ),
        )
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

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
    ) -> Float[Tensor, "batch sequence proj_feature"]:
        for attn, ff in cast(Iterable[tuple[nn.Module, nn.Module]], self.layers):
            x_attn = attn(x, attn_mask=attn_mask)
            x = x_attn + x
            x = ff(x) + x

        x = self.norm(x)
        return x


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

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bags: Float[Tensor, "batch tile feature"],
        *,
        mask: Bool[Tensor, "batch tile"] | None,
    ) -> Float[Tensor, "batch logit"]:
        batch_size, _n_tiles, _n_features = bags.shape

        # Map input sequence to latent space of TransMIL
        bags = self.project_features(bags)

        # Prepend a class token to every bag,
        # include it in the mask.
        # TODO should the tiles be able to refer to the class token? Test!
        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat([cls_tokens, bags], dim=1)
        if mask is not None:
            mask_with_class_token = torch.cat(
                [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
            )
            square_attn_mask = torch.einsum(
                "bq,bk->bqk", mask_with_class_token, mask_with_class_token
            )
            # Don't allow other tiles to reference the class token
            square_attn_mask[:, 1:, 0] = True

            bags = self.transformer(bags, attn_mask=square_attn_mask)
        else:
            bags = self.transformer(bags, attn_mask=None)

        # Only take class token
        bags = bags[:, 0]

        return self.mlp_head(bags)
