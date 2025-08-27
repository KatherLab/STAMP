import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        dim_input: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dim_output: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Linear(dim_input, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, dim_output)
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch num_patches dim_input"],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        """
        Args:
            x: Input tensor of shape [batch, num_patches, dim_input]
            **kwargs: Additional unused inputs like 'coords', 'mask'
        Returns:
            Class logits for each sample: [batch, dim_output]
        """
        B, N, D = x.shape
        x = self.embedding(x)

        # Add [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, embed_dim]

        x = self.transformer(x)
        cls_output = x[:, 0]  # [CLS] token output

        return self.classifier(cls_output)
