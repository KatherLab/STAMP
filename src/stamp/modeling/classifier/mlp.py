from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.classifier import LitPatientlassifier


class MLP(nn.Module):
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


class MLPClassifier(LitPatientlassifier):
    model_name: str = "mlp"

    def build_backbone(
        self, dim_input: int, dim_output: int, metadata: dict
    ) -> nn.Module:
        params = self.get_model_params(MLP, metadata)
        return MLP(
            dim_input=dim_input,
            dim_output=dim_output,
            **params,
        )


class Linear(nn.Module):
    def __init__(self, dim_input: int, dim_output: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_output)

    @jaxtyped
    @beartype
    def forward(
        self,
        x: Float[Tensor, "batch dim_in"],  # batch of feature vectors
    ) -> Float[Tensor, "batch dim_output"]:
        return self.fc(x)


class LinearClassifier(LitPatientlassifier):
    model_name: str = "linear"

    def build_backbone(
        self, dim_input: int, dim_output: int, metadata: dict
    ) -> nn.Module:
        params = self.get_model_params(Linear, metadata)
        return Linear(
            dim_input=dim_input,
            dim_output=dim_output,
            **params,
        )
