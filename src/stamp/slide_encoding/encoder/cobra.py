import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from mamba_ssm import Mamba2  # TODO: add to requirements and try catch

from stamp.cache import STAMP_CACHE_DIR, file_digest  # TODO: assert hash
from stamp.slide_encoding.encoder import Encoder

warnings.simplefilter(action="ignore", category=FutureWarning)


def cobra() -> Encoder:
    checkpoint_path = os.path.join(STAMP_CACHE_DIR, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = hf_hub_download(
            "KatherLab/COBRA",
            filename="pytorch_model.bin",
            local_dir=STAMP_CACHE_DIR,
            force_download=True,
        )
        print(f"Saving model to {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")  # TODO: Ask why is cpu
    model = Cobra(
        input_dims=[768, 1024, 1280, 1536],
    )
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return Encoder(
        model=model,
        identifier="cobra",
    )


# Copy from https://github.com/KatherLab/COBRA


class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024, dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, embed_dim),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x)


class Cobra(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        input_dims=[384, 512, 1024, 1280, 1536],
        num_heads=8,
        layers=2,
        dropout=0.25,
        att_dim=96,
        d_state=128,
    ):
        super().__init__()

        self.embed = nn.ModuleDict({str(d): Embed(d, embed_dim) for d in input_dims})

        self.norm = nn.LayerNorm(embed_dim)

        self.mamba_enc = Mamba2Enc(
            embed_dim,
            embed_dim,
            n_classes=embed_dim,
            layer=layers,
            dropout=dropout,
            d_state=d_state,
        )

        self.num_heads = num_heads
        self.attn = nn.ModuleList(
            [
                BatchedABMIL(
                    input_dim=int(embed_dim / num_heads),
                    hidden_dim=att_dim,
                    dropout=dropout,
                    n_classes=1,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, x, multi_fm_mode=False, fm_idx=None, get_attention=False):
        if multi_fm_mode:
            fm_embs = torch.concat(
                [self.embed[str(xi.shape[-1])](xi) for xi in x], dim=0
            )
            assert fm_embs.shape[-1] == self.embed_dim, fm_embs.shape
            assert len(fm_embs.shape) == 3, fm_embs.shape
            assert fm_embs.shape[0] == len(x), fm_embs.shape
            logits = torch.mean(fm_embs, dim=0)
        else:
            logits = self.embed[str(x.shape[-1])](x)

        h = self.norm(self.mamba_enc(logits))

        if self.num_heads > 1:
            h_ = rearrange(h, "b t (e c) -> b t e c", c=self.num_heads)

            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(
                    h_[:, :, :, i], return_raw_attention=True
                )
                attention.append(processed_attention)

            A = torch.stack(attention, dim=-1)

            A = (
                rearrange(A, "b t e c -> b t (e c)", c=self.num_heads)
                .mean(-1)
                .unsqueeze(-1)
            )
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=-1)
        else:
            A = self.attn[0](h)

        if multi_fm_mode:
            if fm_idx:
                feats = torch.bmm(A, x[fm_idx]).squeeze(0).squeeze(0)
            else:
                feats = []
                for i, xi in enumerate(x):
                    feats.append(torch.bmm(A, xi).squeeze(0).squeeze(0))
                    assert (
                        len(feats[i].shape) == 1 and feats[i].shape[0] == xi.shape[-1]
                    ), feats[i].shape
        else:
            feats = torch.bmm(A, x).squeeze(1)

        if get_attention:
            return feats, A
        return feats


"""
Adapted from: https://github.com/isyangshu/MambaMIL/blob/main/models/MambaMIL.py
Shu Yang, Yihui Wang, and Hao Chen. MambaMIL: En-
hancing Long Sequence Modeling with Sequence Reorder-
ing in Computational Pathology. In proceedings of Medi-
cal Image Computing and Computer Assisted Intervention –
MICCAI 2024. Springer Nature Switzerland, 2024
"""


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Mamba2Enc(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        n_classes,
        dropout=0.25,
        act="gelu",
        layer=2,
        rate=10,
        d_state=64,
    ):
        super(Mamba2Enc, self).__init__()
        self._fc1_layers: list[nn.Module] = [nn.Linear(in_dim, dim)]
        if act.lower() == "relu":
            self._fc1_layers += [nn.ReLU()]
        elif act.lower() == "gelu":
            self._fc1_layers += [nn.GELU()]
        if dropout:
            self._fc1_layers += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1_layers)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    Mamba2(
                        d_model=dim,
                        d_state=d_state,
                        d_conv=4,
                        expand=2,
                    ),
                )
            )

        self.n_classes = n_classes
        self.classifier = nn.Linear(dim, self.n_classes)
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x  # .float()

        h = self._fc1(h)

        for layer in self.layers:
            h_ = h
            h = layer[0](h)  # type: ignore
            h = layer[1](h)
            h = h + h_

        logits = self.classifier(h)
        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


""" adapted from: https://github.com/mahmoodlab/MADELEINE/blob/main/core/models/abmil.py
    Guillaume Jaume, Anurag Jayant Vaidya, Andrew Zhang,
    Andrew H Song, Richard J. Chen, Sharifa Sahai, Dandan
    Mo, Emilio Madrigal, Long Phi Le, and Mahmood Faisal.
    Multistain pretraining for slide representation learning in
    pathology. In European Conference on Computer Vision.
    Springer, 2024.
"""


class BatchedABMIL(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=256,
        dropout=False,
        n_classes=1,
        n_heads=1,
        activation="softmax",
    ):
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attention_a = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Tanh()])

        self.attention_b = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        )

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, return_raw_attention=False):
        assert len(x.shape) == 3, x.shape
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        if self.activation == "softmax":
            activated_A = F.softmax(A, dim=1)
        elif self.activation == "leaky_relu":
            activated_A = F.leaky_relu(A)
        elif self.activation == "relu":
            activated_A = F.relu(A)
        elif self.activation == "sigmoid":
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError("Activation not implemented.")

        if return_raw_attention:
            return activated_A, A

        return activated_A