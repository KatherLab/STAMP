from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class MLP(nn.Module):
    """
    Simple MLP for regression/classification from a feature vector.

    Accepts:
      - (B, F) single feature vector per sample
      - (B, T, F) bag of feature vectors per sample (mean pooled to (B, F))
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        in_dim = dim_input
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, dim_output))
        self.mlp = nn.Sequential(*layers)  # type: ignore

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "..."],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        if x.ndim == 3:  # (B, T, F)
            x = x.mean(dim=1)  # → (B, F)
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        return self.mlp(x.float())


class Linear(nn.Module):
    def __init__(self, dim_input: int, dim_output: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_output)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "..."],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        if x.ndim == 3:
            x = x.mean(dim=1)  # (B, F)
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        return self.fc(x)
