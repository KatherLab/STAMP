# %%
import torch
from test_crossval import test_crossval_integration

from stamp.modeling.alibi import MultiHeadALiBi


def test_alibi_shapes(embed_dim: int = 32, num_heads: int = 8) -> None:
    att = MultiHeadALiBi(
        num_heads=num_heads,
        embed_dim=embed_dim,
    )

    q = torch.rand(2, 23, embed_dim)
    k = torch.rand(2, 34, embed_dim)
    v = torch.rand(2, 8, embed_dim)
    coords_q = torch.rand(2, 23, 2)
    coords_k = torch.rand(2, 34, 2)
    attn_mask = torch.rand(2, 23, 34) > 0.5

    att(
        q=q,
        k=k,
        v=v,
        coords_q=coords_q,
        coords_k=coords_k,
        attn_mask=attn_mask,
        alibi_mask=torch.zeros((2, 23, 34), dtype=torch.bool),
    )

def test_alibi_integration() -> None:
    test_crossval_integration(
        use_alibi=True,
    )
