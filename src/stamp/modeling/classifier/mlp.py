from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """
    Simple MLP for classification from a single feature vector.
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
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class LinearClassifier(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    @jaxtyped
    @beartype
    def forward(
        self,
        x: Float[Tensor, "batch dim_in"],  # batch of feature vectors
    ) -> Float[Tensor, "batch dim_out"]:
        return self.fc(x)
