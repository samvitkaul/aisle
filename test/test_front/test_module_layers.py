"""Module layer — neural layers (Linear, Embedding, LayerNorm)."""
import pytest

import src.front.module as nn
from src import make_tensor

_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestLinear:

    @pytest.mark.unit
    def test_basic(self):
        lin = nn.Linear(uid("lin"), in_features=32, out_features=64)
        x = make_tensor(name=uid("x"), shape=[2, 5, 32], dtype='float32')
        y = lin(x)
        assert y.shape == [2, 5, 64]

    @pytest.mark.unit
    def test_with_bias(self):
        lin = nn.Linear(uid("lin"), in_features=16, out_features=8, bias=True)
        x = make_tensor(name=uid("x"), shape=[4, 16], dtype='float32')
        y = lin(x)
        assert y.shape == [4, 8]

    @pytest.mark.unit
    def test_param_tensor_exists(self):
        lin = nn.Linear(uid("lin"), in_features=10, out_features=20)
        assert lin.param.shape == [10, 20]
        assert lin.param.is_param is True

    @pytest.mark.unit
    def test_bias_tensor_exists(self):
        lin = nn.Linear(uid("lin"), in_features=10, out_features=20, bias=True)
        assert lin.bias is not None
        assert lin.bias.shape == [20]
        assert lin.bias.is_param is True

    @pytest.mark.unit
    def test_no_bias_tensor(self):
        lin = nn.Linear(uid("lin"), in_features=10, out_features=20, bias=False)
        assert lin.bias is None

    @pytest.mark.unit
    def test_batch(self):
        """Linear should handle batched inputs [B, S, in] → [B, S, out]."""
        lin = nn.Linear(uid("lin"), in_features=48, out_features=128)
        x = make_tensor(name=uid("x"), shape=[1, 9, 48], dtype='float32')
        y = lin(x)
        assert y.shape == [1, 9, 128]


class TestEmbedding:

    @pytest.mark.unit
    def test_basic(self):
        emb = nn.Embedding(uid("emb"), tbl_size=50257, emb_dim=768)
        idx = make_tensor(name=uid("idx"), shape=[1, 9], dtype='int64')
        y = emb(idx)
        assert y.shape is not None

    @pytest.mark.unit
    def test_param_shape(self):
        emb = nn.Embedding(uid("emb"), tbl_size=1000, emb_dim=64)
        assert emb.emb_wt.shape == [1000, 64]
        assert emb.emb_wt.is_param is True


class TestLayerNorm:

    @pytest.mark.unit
    def test_basic(self):
        ln = nn.LayerNorm(uid("ln"), 64)
        x = make_tensor(name=uid("x"), shape=[2, 5, 64], dtype='float32')
        y = ln(x)
        assert y.shape == [2, 5, 64]

    @pytest.mark.unit
    def test_2d(self):
        ln = nn.LayerNorm(uid("ln"), 32)
        x = make_tensor(name=uid("x"), shape=[4, 32], dtype='float32')
        y = ln(x)
        assert y.shape == [4, 32]

    @pytest.mark.unit
    def test_param_shapes(self):
        ln = nn.LayerNorm(uid("ln"), 128)
        assert ln.scale.shape == [128]
        assert ln.bias.shape == [128]
        assert ln.scale.is_param is True
        assert ln.bias.is_param is True
