"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_layer=nn.LayerNorm, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.0
    ):
        super().__init__()
        self.heads = heads
        self.norm = norm_layer(dim)
        self.mhsa = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        x = self.norm(x)
        attn_output, _ = self.mhsa(x, x, x, need_weights=False, attn_mask=mask)
        return attn_output


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.0
    ):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            norm_layer=norm_layer,
                            dropout=dropout,
                        ),
                        FeedForward(
                            dim, mlp_dim, norm_layer=norm_layer, dropout=dropout
                        ),
                    ]
                )
            )
        self.norm = norm_layer(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x_attn = attn(x, mask=mask)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x)


class TransMIL(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_dim: int = 768,
        dim: int = 512,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: int = 0.0,
        emb_dropout: int = 0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x, lens):
        # remove unnecessary padding
        # (deactivated for now, since the memory usage fluctuates more and is overall bigger)
        # x = x[:, :torch.max(lens)].contiguous()
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        cls_tokens = repeat(self.cls_token, "d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        lens = lens + 1  # account for cls token

        x = self.transformer(x, mask=None)

        x = x[:, 0]  # only take class token

        return self.mlp_head(x)
