from typing import Optional, Tuple

import torch
import torch.nn as nn

from stamp.modeling.models.ft_transformer import FTTransformer
from stamp.modeling.models.vision_tranformer import VisionTransformer


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
    ):
        super().__init__()
        if use_tabular:
            self.tabular_model = FTTransformer(
                categories=tab_categories,
                num_continuous=tab_num_continuous,
                dim=tab_embed_dim,
                depth=tab_depth,
                heads=tab_num_heads,
                dim_head=tab_head_dim,
                dim_out=tab_out_dim,
                num_special_tokens=tab_num_special_tokens,
                attn_dropout=tab_attn_dropout,
                ff_dropout=tab_ff_dropout,
                num_residual_streams=tab_num_residual_streams,
            )
        if use_image:
            self.image_model = VisionTransformer(
                dim_output=dim_output,
                dim_input=dim_input,
                dim_model=mil_model_dim,
                n_layers=mil_num_layers,
                n_heads=mil_num_heads,
                dim_feedforward=mil_ff_dim,
                dropout=mil_dropout,
                use_alibi=mil_use_alibi,
            )

        fusion_dim = dim_output + (tab_out_dim if use_tabular else 0)
        self.fc = nn.Linear(fusion_dim, output_dim)

    def forward(
        self,
        *,
        bags: torch.Tensor,
        coords: torch.Tensor,
        bag_sizes: torch.Tensor,
        tabular: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        features = []

        # ---- MIL branch ----
        if self.use_image:
            img_features = self.image_model(
                bags,
                coords=coords,
                mask=None,
            )  # [B, mil_out_dim]
            features.append(img_features)

        # ---- Tabular branch ----
        if self.use_tabular:
            assert tabular is not None, "use_tabular=True but no tabular data provided"
            x_categ, x_numer = tabular
            tab_features = self.tabular_model(x_categ, x_numer)  # [B, tab_out_dim]
            features.append(tab_features)

        # ---- Safety check ----
        if not features:
            raise RuntimeError("Both use_image and use_tabular are False")

        # ---- Fusion ----
        fused = torch.cat(features, dim=-1)

        # ---- Head ----
        logits = self.fc(fused)  # [B, output_dim]
        return logits
