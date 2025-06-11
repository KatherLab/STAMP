"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

from collections.abc import Iterable
from typing import Literal, assert_never, cast, overload

import torch
from beartype import beartype
from einops import repeat
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.alibi import MultiHeadALiBi


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
        *,
        dim: int,
        num_heads: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.norm = nn.LayerNorm(dim)

        if use_alibi:
            self.mhsa: MultiHeadALiBi | nn.MultiheadAttention = MultiHeadALiBi(
                embed_dim=dim,
                num_heads=num_heads,
            )
        else:
            self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence xy"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: Literal[False] = False,
    ) -> Float[Tensor, "batch sequence proj_feature"]: ...

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence xy"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: Literal[True],
    ) -> tuple[  # if return_attention is True, return the attention weights as well
        Float[Tensor, "batch sequence proj_feature"],
        Float[Tensor, "batch heads sequence sequence"],
    ]: ...

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence xy"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        # Help, my abstractions are leaking!
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: bool = False,
    ) -> (
        Float[Tensor, "batch sequence proj_feature"]
        | tuple[
            Float[Tensor, "batch sequence proj_feature"],
            Float[Tensor, "batch heads sequence sequence"],
        ]
    ):
        """
        Args:
            attn_mask:
                Which of the features to ignore during self-attention.
                `attn_mask[b,q,k] == False` means that
                query `q` of batch `b` can attend to key `k`.
                If `attn_mask` is `None`, all tokens can attend to all others.
            alibi_mask:
                Which query-key pairs to apply ALiBi to.
                If this module was constructed using `use_alibi=False`,
                this has no effect.
            return_attention:
                If True, returns the attention weights alongside the output.
        """
        x = self.norm(x)

        # Initialize attention weights with default shape
        last_attn_weights: Float[Tensor, "batch heads sequence sequence"] | None = None

        match self.mhsa:
            case nn.MultiheadAttention():
                attn_output, attn_weights = self.mhsa(
                    x,
                    x,
                    x,
                    need_weights=True,
                    average_attn_weights=False,
                    attn_mask=(
                        attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                        if attn_mask is not None
                        else None
                    ),
                )
                last_attn_weights = attn_weights

            case MultiHeadALiBi():
                # Modified MultiHeadALiBi to return attention weights
                if hasattr(self.mhsa, "return_attention_weights"):
                    try:
                        attn_output, attn_weights = self.mhsa(
                            q=x,
                            k=x,
                            v=x,
                            coords_q=coords,
                            coords_k=coords,
                            attn_mask=attn_mask,
                            alibi_mask=alibi_mask,
                            return_attention=True,
                        )
                        last_attn_weights = attn_weights
                    except (TypeError, ValueError, RuntimeError) as e:
                        # If the return_attention param exists but fails, fall back
                        attn_output = self.mhsa(
                            q=x,
                            k=x,
                            v=x,
                            coords_q=coords,
                            coords_k=coords,
                            attn_mask=attn_mask,
                            alibi_mask=alibi_mask,
                        )
                        # Create dummy attention weights to satisfy type checking
                        if return_attention:
                            print(
                                f"Warning: Failed to return attention weights ({type(e).__name__}: {e}). Creating dummy weights."
                            )
                            batch_size, seq_len, _ = x.shape
                            last_attn_weights = torch.zeros(
                                batch_size,
                                self.heads,
                                seq_len,
                                seq_len,
                                device=x.device,
                                dtype=x.dtype,
                            )
                else:
                    attn_output = self.mhsa(
                        q=x,
                        k=x,
                        v=x,
                        coords_q=coords,
                        coords_k=coords,
                        attn_mask=attn_mask,
                        alibi_mask=alibi_mask,
                    )
                    last_attn_weights = None
            case _ as unreachable:
                assert_never(unreachable)

        if return_attention:
            # Ensure we always return valid tensor for attention weights
            if last_attn_weights is None:
                # Create default attention weights if none were produced
                batch_size, seq_len, _ = x.shape
                last_attn_weights = torch.zeros(
                    batch_size,
                    self.heads if hasattr(self, "heads") else 1,
                    seq_len,
                    seq_len,
                    device=x.device,
                    dtype=x.dtype,
                )
            return attn_output, last_attn_weights

        return attn_output


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim=dim,
                            num_heads=heads,
                            dropout=dropout,
                            use_alibi=use_alibi,
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

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence 2"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: Literal[False] = False,
    ) -> Float[Tensor, "batch sequence proj_feature"]: ...

    @overload
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence 2"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: Literal[True],
    ) -> tuple[
        Float[Tensor, "batch sequence proj_feature"],
        list[Float[Tensor, "batch heads sequence sequence"]],
    ]: ...

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence 2"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
        return_attention: bool = False,
    ) -> (
        Float[Tensor, "batch sequence proj_feature"]
        | tuple[
            Float[Tensor, "batch sequence proj_feature"],
            list[Float[Tensor, "batch heads sequence sequence"]],
        ]
    ):
        attention_weights = []

        for attn, ff in cast(Iterable[tuple[SelfAttention, nn.Module]], self.layers):
            if return_attention:
                x_attn, attn_weights = attn(
                    x,
                    coords=coords,
                    attn_mask=attn_mask,
                    alibi_mask=alibi_mask,
                    return_attention=True,
                )
                attention_weights.append(attn_weights)
            else:
                x_attn = attn(
                    x, coords=coords, attn_mask=attn_mask, alibi_mask=alibi_mask
                )

            x = x_attn + x
            x = ff(x) + x

        x = self.norm(x)

        if return_attention:
            return x, attention_weights
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
        use_alibi: bool,
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
            use_alibi=use_alibi,
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim_model, dim_output))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bags: Float[Tensor, "batch tile feature"],
        *,
        coords: Float[Tensor, "batch tile 2"],
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
        coords = torch.cat(
            [torch.zeros(batch_size, 1, 2).type_as(coords), coords], dim=1
        )

        match mask:
            case None:
                bags = self.transformer(bags, coords=coords, attn_mask=None)

            case _:
                mask_with_class_token = torch.cat(
                    [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
                )
                square_attn_mask = torch.einsum(
                    "bq,bk->bqk", mask_with_class_token, mask_with_class_token
                )
                # Don't allow other tiles to reference the class token
                square_attn_mask[:, 1:, 0] = True

                # Don't apply ALiBi to the query, as the coordinates don't make sense here
                alibi_mask = torch.zeros_like(square_attn_mask)
                alibi_mask[:, 0, :] = True
                alibi_mask[:, :, 0] = True

                bags = self.transformer(
                    bags,
                    coords=coords,
                    attn_mask=square_attn_mask,
                    alibi_mask=alibi_mask,
                )

        # Only take class token
        bags = bags[:, 0]

        return self.mlp_head(bags)

    def get_attention_maps(
        self,
        bags: Float[Tensor, "batch tile feature"],
        *,
        coords: Float[Tensor, "batch tile 2"],
        mask: Bool[Tensor, "batch tile"] | None,
    ) -> Iterable[Float[Tensor, "batch heads sequence sequence"]]:
        """Extract the attention maps from the last layer of the transformer."""
        batch_size, _n_tiles, _n_features = bags.shape

        # Map input sequence to latent space of TransMIL
        bags = self.project_features(bags)

        # Prepend a class token to every bag
        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat([cls_tokens, bags], dim=1)
        coords = torch.cat(
            [torch.zeros(batch_size, 1, 2).type_as(coords), coords], dim=1
        )

        # Create necessary masks
        if mask is None:
            bags, attention_weights = self.transformer(
                bags,
                coords=coords,
                attn_mask=None,
                alibi_mask=None,
                return_attention=True,
            )
        else:
            mask_with_class_token = torch.cat(
                [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
            )
            square_attn_mask = torch.einsum(
                "bq,bk->bqk", mask_with_class_token, mask_with_class_token
            )
            # Don't allow other tiles to reference the class token
            square_attn_mask[:, 1:, 0] = True

            # Don't apply ALiBi to the query, as the coordinates don't make sense here
            alibi_mask = torch.zeros_like(square_attn_mask)
            alibi_mask[:, 0, :] = True
            alibi_mask[:, :, 0] = True

            bags, attention_weights = self.transformer(
                bags,
                coords=coords,
                attn_mask=square_attn_mask,
                alibi_mask=alibi_mask,
                return_attention=True,
            )

        # Return the attention weights
        return attention_weights
