import torch

from tableseq.models import TableSeqEncoder


def test_encoder_forward_shapes():
    model = TableSeqEncoder(
        {
            "dropout": 0.0,
            "structure_dropout": 0.0,
            "hidden_dim": 128,
            "transformer_ff_dim": 256,
            "structure_mid_channels": 64,
        }
    )
    model.eval()

    x = torch.randn(2, 3, 32, 64)
    with torch.no_grad():
        tokens, structure_logits, key_bias = model(x, return_struct=True)

    assert tokens.shape == (2, 16, 128)
    assert structure_logits.shape == (2, 3, 4, 8)
    assert key_bias.shape == (2, 16)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(structure_logits).all()
    assert torch.isfinite(key_bias).all()


def test_encoder_rejects_wrong_channels():
    model = TableSeqEncoder()
    x = torch.randn(1, 1, 64, 128)
    try:
        model(x)
    except ValueError as exc:
        assert "input channels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong number of channels")
