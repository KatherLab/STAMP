from typing import Optional, Tuple

import torch
import torch.nn as nn

from stamp.modeling.models.ft_transformer import FTTransformer
from stamp.modeling.models.mlp import MLP
from stamp.modeling.models.vision_tranformer import VisionTransformer


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MILTabularModel(nn.Module):
    def __init__(
        self,
        # -------- Tabular (FT-Transformer) --------
        tab_categories: tuple[int, ...],
        tab_num_continuous: int,
        tab_embed_dim: int,
        tab_depth: int,
        tab_num_heads: int,
        tab_head_dim: int,
        tab_out_dim: int,
        tab_num_special_tokens: int,
        tab_attn_dropout: float,
        tab_ff_dropout: float,
        tab_num_residual_streams: int,
        # -------- MIL (Vision Transformer) --------
        dim_output: int,
        dim_input: int,
        mil_model_dim: int,
        mil_num_layers: int,
        mil_num_heads: int,
        mil_ff_dim: int,
        mil_dropout: float,
        mil_use_alibi: bool,
        # -------- Fusion / control --------
        use_tabular: bool = True,
        use_image: bool = True,
        output_dim: int = 1,  # numerical output (regression / survival)
        fusion: str = "concat",  # "gate" (default) or "concat"
    ):
        super().__init__()
        self.use_tabular = use_tabular
        self.use_image = use_image
        features = []
        self.fusion = fusion.lower()
        if self.fusion not in ("gate", "concat"):
            raise ValueError("fusion must be 'gate' or 'concat'")
            # if self.use_tabular:
            #     self.tabular_model = FTTransformer(
            #         categories=tab_categories,
            #         num_continuous=tab_num_continuous,
            #         dim=tab_embed_dim,
            #         depth=tab_depth,
            #         heads=tab_num_heads,
            #         dim_head=tab_head_dim,
            #         dim_out=tab_embed_dim,
            #         num_special_tokens=tab_num_special_tokens,
            #         attn_dropout=tab_attn_dropout,
            #         ff_dropout=tab_ff_dropout,
            #         num_residual_streams=tab_num_residual_streams,
            #     )
            # self.w_tab = nn.Parameter(torch.tensor(0.01))
        if self.use_tabular:
            # correct: number of categorical *columns* is len(tab_categories)
            tab_in_dim = tab_num_continuous + len(tab_categories)

            # TabularMLP should accept in_dim and produce tab_embed_dim
            self.tabular_model = TabularMLP(
                in_dim=tab_in_dim,
                out_dim=tab_embed_dim,
                hidden=4,
            )

        # keep tab influence small by default
        self.w_tab = nn.Parameter(torch.tensor(0.1))
        if self.use_image:
            self.image_model = MLP(
                dim_input=dim_input,
                dim_hidden=mil_model_dim,
                dim_output=mil_model_dim,
                num_layers=mil_num_layers,
                dropout=mil_dropout,
            )
            self.w_img = nn.Parameter(torch.tensor(1.0))

        # if use_image and use_tabular:
        #     self.fc = nn.Linear(mil_model_dim + tab_embed_dim, output_dim)
        # elif use_image:
        #     self.fc = nn.Linear(mil_model_dim, output_dim)
        # elif use_tabular:
        #     self.fc = nn.Linear(tab_embed_dim, output_dim)
        # else:
        #     raise ValueError("At least one of use_image or use_tabular must be True")

        fusion_dim = 0
        if self.use_image:
            fusion_dim += mil_model_dim
        if self.use_tabular:
            fusion_dim += tab_embed_dim
        if fusion_dim == 0:
            raise ValueError("At least one of use_image or use_tabular must be True")

        self.fc = nn.Linear(fusion_dim, output_dim)

    def forward(
        self,
        bags: torch.Tensor,
        *,
        coords: torch.Tensor,
        tabular: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = []

        # ---- MIL branch ----
        if self.use_image:
            img_features = self.image_model(
                bags,
                coords=coords,
                mask=mask,
            )
            img_features = self.w_img * img_features

        # ---- Tabular branch ----
        # if self.use_tabular:
        #     assert tabular is not None
        #     x_categ, x_numer = tabular
        #     tab_features = self.tabular_model(x_categ, x_numer)
        #     tab_features = self.w_tab * tab_features

        # # ---- Fusion ----
        # # if self.use_image and self.use_tabular:
        # #     fused = torch.cat([img_features, tab_features], dim=-1)  # pyright: ignore[reportPossiblyUnboundVariable]
        # # elif self.use_image:
        # #     fused = img_features  # pyright: ignore[reportPossiblyUnboundVariable]
        # # elif self.use_tabular:
        # #     fused = tab_features  # pyright: ignore[reportPossiblyUnboundVariable]
        # # else:
        # #     raise RuntimeError("No active modalities")
        # alpha = torch.sigmoid(tab_features.mean(dim=-1, keepdim=True))  # pyright: ignore[reportPossiblyUnboundVariable]
        # img_features = img_features * alpha  # pyright: ignore[reportPossiblyUnboundVariable]
        # fused = img_features

        # logits = self.fc(fused)
        # return logits

        if self.use_tabular:
            assert tabular is not None, "use_tabular=True but tabular is None"
            x_categ, x_numer = tabular
            # build tab_input: categorical columns (as indices) are expected to be provided appropriately.
            # Here we assume x_categ already contains one-hot or numeric per categorical column;
            # you previously used x_categ.float(), so keep that behavior.
            tab_input = torch.cat([x_categ.float(), x_numer], dim=-1)
            tab_features = self.tabular_model(tab_input)  # [B, tab_embed_dim]
            # tab_features = self.w_tab * tab_features
        else:
            tab_features = None

        # ---- Fusion ----
        if self.use_image and self.use_tabular:
            if self.fusion == "concat":
                # explicit concatenation
                fused = torch.cat([img_features, tab_features], dim=-1)
            else:  # gate
                # gating: tabular provides a scalar gate per sample
                # compute gate from tabular embedding (reduce to scalar)
                alpha = torch.sigmoid(tab_features.mean(dim=-1, keepdim=True))  # [B, 1]
                fused = img_features * alpha  # broadcasting over feature dim
        elif self.use_image:
            fused = img_features
        elif self.use_tabular:
            fused = tab_features
        else:
            raise RuntimeError("No active modalities")

        # safety check: ensure fc matches fused dim
        if fused.shape[-1] != self.fc.in_features:
            raise RuntimeError(
                f"Fusion dimension mismatch: fused.shape[-1]={fused.shape[-1]} != "
                f"fc.in_features={self.fc.in_features}"
            )

        logits = self.fc(fused)
        return logits