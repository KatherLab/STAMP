from torch import nn

from stamp.modeling.models import LitTileRegressor
from stamp.modeling.models.classifier.mlp import MLP, Linear


class LinearRegressor(LitTileRegressor):
    model_name: str = "linear_regressor"

    def build_backbone(self, dim_input: int, metadata: dict) -> nn.Module:
        return Linear(dim_input, 1)


class MLPRegressor(LitTileRegressor):
    model_name: str = "mlp_regressor"

    def build_backbone(self, dim_input: int, metadata: dict) -> nn.Module:
        params = self.get_model_params(MLP, metadata)
        return MLP(
            dim_input=dim_input,
            dim_output=1,
            **params,
        )
