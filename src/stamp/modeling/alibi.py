import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class _RunningMeanScaler(nn.Module):
    """Scales values by the inverse of the mean of values seen before."""

    def __init__(self, dtype=torch.float32) -> None:
        super().__init__()
        self.running_mean = nn.Buffer(torch.ones(1, dtype=dtype))
        self.items_so_far = nn.Buffer(torch.ones(1, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Welford's algorithm
            self.running_mean.copy_(
                (self.running_mean + (x - self.running_mean) / self.items_so_far).mean()
            )
            self.items_so_far += 1

        return x / self.running_mean


class _ALiBi(nn.Module):
    # See MultiHeadAliBi
    def __init__(self) -> None:
        super().__init__()

        self.scale_distance = _RunningMeanScaler()
        self.bias_scale = nn.Parameter(torch.rand(1))

    def forward(
        self,
        *,
        q: Float[Tensor, "batch query qk_feature"],
        k: Float[Tensor, "batch key qk_feature"],
        v: Float[Tensor, "batch key v_feature"],
        coords_q: Float[Tensor, "batch query coord"],
        coords_k: Float[Tensor, "batch key coord"],
        attn_mask: Bool[Tensor, "batch query key"] | None,
        alibi_mask: Bool[Tensor, "batch query key"] | None,
    ) -> Float[Tensor, "batch query v_feature"]:
        """
        Args:
            alibi_mask:
                Which query-key pairs to mask from ALiBi (i.e. don't apply ALiBi to).
        """
        weight_logits = torch.einsum("bqf,bkf->bqk", q, k) * (k.size(-1) ** -0.5)
        distances = torch.linalg.norm(
            coords_q.unsqueeze(2) - coords_k.unsqueeze(1), dim=-1
        )
        scaled_distances = self.scale_distance(distances) * self.bias_scale

        if alibi_mask is not None:
            scaled_distances = scaled_distances.where(~alibi_mask, 0.0)

        weights = torch.softmax(weight_logits, dim=-1)

        if attn_mask is not None:
            weights = (weights - scaled_distances).where(~attn_mask, 0.0)
        else:
            weights = weights - scaled_distances

        attention = torch.einsum("bqk,bkf->bqf", weights, v)

        return attention


class MultiHeadALiBi(nn.Module):
    """Attention with Linear Biases

    Based on
    > PRESS, Ofir; SMITH, Noah A.; LEWIS, Mike.
    > Train short, test long: Attention with linear biases enables input length extrapolation.
    > arXiv preprint arXiv:2108.12409, 2021.

    Since the distances between in WSIs may be quite large,
    we scale the distances by the mean distance seen during training.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"{embed_dim=} has to be divisible by {num_heads=}")

        self.query_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )
        self.key_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )
        self.value_encoders = nn.ModuleList(
            [
                nn.Linear(in_features=embed_dim, out_features=embed_dim // num_heads)
                for _ in range(num_heads)
            ]
        )

        self.attentions = nn.ModuleList([_ALiBi() for _ in range(num_heads)])

        self.fc = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(
        self,
        *,
        q: Float[Tensor, "batch query mh_qk_feature"],
        k: Float[Tensor, "batch key mh_qk_feature"],
        v: Float[Tensor, "batch key hm_v_feature"],
        coords_q: Float[Tensor, "batch query coord"],
        coords_k: Float[Tensor, "batch key coord"],
        attn_mask: Bool[Tensor, "batch query key"] | None,
        alibi_mask: Bool[Tensor, "batch query key"] | None,
    ) -> Float[Tensor, "batch query mh_v_feature"]:
        stacked_attentions = torch.stack(
            [
                att(
                    q=q_enc(q),
                    k=k_enc(k),
                    v=v_enc(v),
                    coords_q=coords_q,
                    coords_k=coords_k,
                    attn_mask=attn_mask,
                    alibi_mask=alibi_mask,
                )
                for q_enc, k_enc, v_enc, att in zip(
                    self.query_encoders,
                    self.key_encoders,
                    self.value_encoders,
                    self.attentions,
                    strict=True,
                )
            ]
        )
        return self.fc(stacked_attentions.permute(1, 2, 0, 3).flatten(-2, -1))
