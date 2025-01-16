import torch

from stamp.modeling.vision_transformer import VisionTransformer


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
