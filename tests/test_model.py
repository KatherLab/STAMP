import torch

from stamp.modeling.models.mlp import MLP
from stamp.modeling.models.trans_mil import TransMIL
from stamp.modeling.models.vision_tranformer import VisionTransformer


def test_vision_transformer_dims(
    # arbitrarily chosen constants
    num_classes: int = 3,
    batch_size: int = 6,
    n_tiles: int = 75,
    input_dim: int = 456,
    n_heads: int = 4,
) -> None:
    model = VisionTransformer(
        dim_output=num_classes,
        dim_input=input_dim,
        dim_model=n_heads * 33,
        n_layers=3,
        n_heads=n_heads,
        dim_feedforward=135,
        dropout=0.12,
        use_alibi=False,
    )

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    mask = torch.rand((batch_size, n_tiles)) > 0.5
    logits = model.forward(bags, coords=coords, mask=mask)
    assert logits.shape == (batch_size, num_classes)


def test_inference_reproducibility(
    # arbitrarily chosen constants
    num_classes: int = 4,
    batch_size: int = 7,
    n_tiles: int = 76,
    input_dim: int = 457,
    n_heads: int = 5,
) -> None:
    model = VisionTransformer(
        dim_output=num_classes,
        dim_input=input_dim,
        dim_model=n_heads * 34,
        n_layers=3,
        n_heads=n_heads,
        dim_feedforward=135,
        dropout=0.12,
        use_alibi=False,
    )

    model = model.eval()

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    mask = (
        torch.arange(n_tiles).to(device=bags.device).unsqueeze(0).repeat(batch_size, 1)
    ) >= torch.randint(1, n_tiles, (batch_size, 1))

    with torch.inference_mode():
        logits1 = model.forward(
            bags,
            coords=coords,
            mask=mask,
        )
        logits2 = model.forward(
            bags,
            coords=coords,
            mask=mask,
        )

    assert logits1.allclose(logits2)


def test_mlp_classifier_dims(
    num_classes: int = 3,
    batch_size: int = 6,
    input_dim: int = 32,
    dim_hidden: int = 64,
    num_layers: int = 2,
) -> None:
    model = MLP(
        dim_output=num_classes,
        dim_input=input_dim,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        dropout=0.1,
    )
    feats = torch.rand((batch_size, input_dim))
    logits = model.forward(feats)
    assert logits.shape == (batch_size, num_classes)


def test_mlp_inference_reproducibility(
    num_classes: int = 4,
    batch_size: int = 7,
    input_dim: int = 33,
    dim_hidden: int = 64,
    num_layers: int = 3,
) -> None:
    model = MLP(
        dim_output=num_classes,
        dim_input=input_dim,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        dropout=0.1,
    )
    model = model.eval()
    feats = torch.rand((batch_size, input_dim))
    with torch.inference_mode():
        logits1 = model.forward(feats)
        logits2 = model.forward(feats)
    assert torch.allclose(logits1, logits2)


def test_trans_mil_dims(
    # arbitrarily chosen constants
    num_classes: int = 3,
    batch_size: int = 6,
    n_tiles: int = 75,
    input_dim: int = 456,
    dim_hidden: int = 512,
) -> None:
    model = TransMIL(dim_output=num_classes, dim_input=input_dim, dim_hidden=dim_hidden)

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    mask = torch.rand((batch_size, n_tiles)) > 0.5
    logits = model.forward(bags, coords=coords, mask=mask)
    assert logits.shape == (batch_size, num_classes)


def test_trans_mil_inference_reproducibility(
    # arbitrarily chosen constants
    num_classes: int = 4,
    batch_size: int = 7,
    n_tiles: int = 76,
    input_dim: int = 457,
    dim_hidden: int = 512,
) -> None:
    model = TransMIL(dim_output=num_classes, dim_input=input_dim, dim_hidden=dim_hidden)

    model = model.eval()

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    mask = (
        torch.arange(n_tiles).to(device=bags.device).unsqueeze(0).repeat(batch_size, 1)
    ) >= torch.randint(1, n_tiles, (batch_size, 1))

    with torch.inference_mode():
        logits1 = model.forward(
            bags,
            coords=coords,
            mask=mask,
        )
        logits2 = model.forward(
            bags,
            coords=coords,
            mask=mask,
        )

    assert logits1.allclose(logits2)
