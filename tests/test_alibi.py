import torch

from stamp.modeling.alibi import MultiHeadALiBi


def test_alibi_shapes() -> None:
    att = MultiHeadALiBi(
        num_heads=8, query_dim=4, key_dim=7, value_dim=9, inner_dim=13, out_dim=56
    )

    q = torch.rand(2, 23, 4)
    k = torch.rand(2, 34, 7)
    v = torch.rand(2, 8, 9)
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
