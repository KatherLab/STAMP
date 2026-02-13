import torch

from stamp.modeling.models.barspoon import EncDecTransformer
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


def test_enc_dec_transformer_dims(
    batch_size: int = 6,
    n_tiles: int = 75,
    input_dim: int = 456,
    d_model: int = 128,
) -> None:
    target_n_outs = {"subtype": 3, "grade": 4}
    model = EncDecTransformer(
        d_features=input_dim,
        target_n_outs=target_n_outs,
        d_model=d_model,
        num_encoder_heads=4,
        num_decoder_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        positional_encoding=True,
    )

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    logits = model.forward(bags, coords)

    assert set(logits.keys()) == set(target_n_outs.keys())
    for target_label, n_out in target_n_outs.items():
        assert logits[target_label].shape == (batch_size, n_out)


def test_enc_dec_transformer_single_target(
    batch_size: int = 4,
    n_tiles: int = 50,
    input_dim: int = 256,
    d_model: int = 64,
) -> None:
    target_n_outs = {"label": 5}
    model = EncDecTransformer(
        d_features=input_dim,
        target_n_outs=target_n_outs,
        d_model=d_model,
        num_encoder_heads=4,
        num_decoder_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
    )

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    logits = model.forward(bags, coords)

    assert list(logits.keys()) == ["label"]
    assert logits["label"].shape == (batch_size, 5)


def test_enc_dec_transformer_no_positional_encoding(
    batch_size: int = 4,
    n_tiles: int = 30,
    input_dim: int = 128,
    d_model: int = 64,
) -> None:
    target_n_outs = {"a": 2, "b": 3}
    model = EncDecTransformer(
        d_features=input_dim,
        target_n_outs=target_n_outs,
        d_model=d_model,
        num_encoder_heads=4,
        num_decoder_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        positional_encoding=False,
    )

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))
    logits = model.forward(bags, coords)

    for target_label, n_out in target_n_outs.items():
        assert logits[target_label].shape == (batch_size, n_out)


def test_enc_dec_transformer_inference_reproducibility(
    batch_size: int = 5,
    n_tiles: int = 40,
    input_dim: int = 200,
    d_model: int = 64,
) -> None:
    target_n_outs = {"subtype": 3, "grade": 4}
    model = EncDecTransformer(
        d_features=input_dim,
        target_n_outs=target_n_outs,
        d_model=d_model,
        num_encoder_heads=4,
        num_decoder_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
    )
    model = model.eval()

    bags = torch.rand((batch_size, n_tiles, input_dim))
    coords = torch.rand((batch_size, n_tiles, 2))

    with torch.inference_mode():
        logits1 = model.forward(bags, coords)
        logits2 = model.forward(bags, coords)

    for target_label in target_n_outs:
        assert logits1[target_label].allclose(logits2[target_label])
