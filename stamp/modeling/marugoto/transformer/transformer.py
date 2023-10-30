"""
In parts from https://github.com/lucidrains
"""

import torch
from einops import rearrange
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8, use_ff=True, use_norm=True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim // heads)
        self.use_ff = use_ff
        self.use_norm = use_norm
        if self.use_ff:
            self.ff = FeedForward()

    def forward(self, x):
        if self.use_norm:
            x = x + self.attn(self.norm(x))
        else:
            x = x + self.attn(x)
        if self.use_ff:
            x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.n_classes = num_classes

        self._fc1 = nn.Sequential(nn.Linear(2048, 512, bias=True), nn.ReLU())
        self.layer1 = TransformerLayer(dim=512, heads=8, use_ff=False, use_norm=True)
        self.layer2 = TransformerLayer(dim=512, heads=8, use_ff=False, use_norm=True)
        self._fc2 = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x,_):

        h = x
        h = self._fc1(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = h.mean(dim=1)
        logits = self._fc2(h)

        return logits
