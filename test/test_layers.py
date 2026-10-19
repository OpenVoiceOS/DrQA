import torch

from drqa import layers


def test_stacked_brnn_unpadded_keeps_features():
    rnn = layers.StackedBRNN(input_size=4, hidden_size=5, num_layers=2,
                             concat_layers=True)
    x = torch.randn(3, 6, 4)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    out = rnn(x, mask)
    assert out.shape == (3, 6, 2 * 5 * 2)


def test_stacked_brnn_padded_path_shape_and_finite():
    torch.manual_seed(0)
    rnn = layers.StackedBRNN(input_size=4, hidden_size=5, num_layers=2,
                             concat_layers=True)
    rnn.eval()
    x = torch.randn(3, 6, 4)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[0, 4:] = True
    mask[2, 5:] = True
    out = rnn._forward_padded(x, mask)
    assert out.shape == (3, 6, 2 * 5 * 2)
    assert torch.isfinite(out).all()


def test_stacked_brnn_padded_matches_unpadded_without_padding():
    torch.manual_seed(0)
    rnn = layers.StackedBRNN(input_size=4, hidden_size=5, num_layers=1)
    rnn.eval()
    x = torch.randn(2, 6, 4)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    padded = rnn._forward_padded(x, mask)
    unpadded = rnn._forward_unpadded(x, mask)
    assert torch.allclose(padded, unpadded, atol=1e-5)


def test_seq_attn_match_shape_and_finite():
    attn = layers.SeqAttnMatch(4)
    x = torch.randn(2, 5, 4)
    y = torch.randn(2, 3, 4)
    y_mask = torch.zeros(2, 3, dtype=torch.bool)
    y_mask[1, 2] = True
    out = attn(x, y, y_mask)
    assert out.shape == (2, 5, 4)
    assert torch.isfinite(out).all()


def test_bilinear_attn_masks_padding():
    attn = layers.BilinearSeqAttn(4, 6)
    attn.eval()
    x = torch.randn(2, 5, 4)
    y = torch.randn(2, 6)
    x_mask = torch.zeros(2, 5, dtype=torch.bool)
    x_mask[1, 3:] = True
    alpha = attn(x, y, x_mask)
    assert alpha.shape == (2, 5)
    assert (alpha[1, 3:] == 0).all()


def test_linear_self_attn_sums_to_one():
    attn = layers.LinearSeqAttn(4)
    x = torch.randn(2, 5, 4)
    x_mask = torch.zeros(2, 5, dtype=torch.bool)
    alpha = attn(x, x_mask)
    assert torch.allclose(alpha.sum(1), torch.ones(2), atol=1e-5)


def test_uniform_weights_ignore_padding():
    x = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 4] = True
    alpha = layers.uniform_weights(x, mask)
    assert torch.allclose(alpha.sum(1), torch.ones(2), atol=1e-5)
    assert alpha[0, 4] == 0
    assert torch.allclose(alpha[0, :4], torch.full((4,), 0.25))
