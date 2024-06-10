"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat



class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = heads != 1 or dim_head != dim

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = norm_layer(dim)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, mask=None):
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         dots = (q @ k.mT) * self.scale

#         if mask is not None:
#             mask_value = torch.finfo(dots.dtype).min
#             dots.masked_fill_(mask, mask_value)

#         # improve numerical stability of softmax
#         dots = dots - torch.amax(dots, dim=-1, keepdim=True)
#         attn = F.softmax(dots, dim=-1)

#         out = attn @ v
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out), attn


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=512 // 8, norm_layer=nn.LayerNorm, dropout=0.):
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, norm_layer=norm_layer, dropout=dropout),
                FeedForward(dim, mlp_dim, norm_layer=norm_layer, dropout=dropout)
            ]))
        self.norm = norm_layer(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x_attn = attn(x, mask=mask)
            x = x_attn + x
            x = ff(x) + x
        return self.norm(x)


class TransMIL(nn.Module):
    def __init__(self, *, 
        num_classes: int, input_dim: int = 768, dim: int = 512,
        depth: int = 2, heads: int = 8, dim_head: int = 64, mlp_dim: int = 2048,
        pool: str ='cls', dropout: int = 0., emb_dropout: int = 0.
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.fc = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.GELU())
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, nn.LayerNorm, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, lens):
        # remove unnecessary padding
        # (deactivated for now, since the memory usage fluctuates more and is overall bigger)
        # x = x[:, :torch.max(lens)].contiguous()
        b, n, d = x.shape

        # map input sequence to latent space of TransMIL
        x = self.dropout(self.fc(x))

        add_cls = self.pool == 'cls'
        if add_cls:
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            lens = lens + 1 # account for cls token

        # mask indicating zero padded feature vectors
        # (deactivated for now, since it seems to use more memory than without)
        mask = None
        if torch.amin(lens) != torch.amax(lens) and False:
            mask = torch.arange(0, n + add_cls, dtype=torch.int32, device=x.device).repeat(b, 1) < lens[..., None]
            mask = (~mask[:, None, :]).repeat(1, (n + add_cls), 1) # shape: (B, L, L)
            # mask = (~mask[:, None, :]).expand(-1, (n + add_cls), -1)

        x = self.transformer(x, mask)

        if mask is not None and self.pool == 'mean':
            x = torch.cumsum(x, dim=1)[torch.arange(b), lens - 1]
            x = x / lens[..., None]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        return self.mlp_head(x)
