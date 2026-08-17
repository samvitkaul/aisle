"""Functional layer — unary ops (Softmax, Gelu, Sigmoid, Transpose, Split)."""
import pytest

from src import make_tensor
import src.front.functional as F


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestUnaryOps:
    """Unary ops should preserve input shape and propagate dtype."""

    @pytest.mark.unit
    @pytest.mark.parametrize("OpFactory", [F.Softmax, F.Gelu, F.Sigmoid])
    @pytest.mark.parametrize("shape", [
        [4],
        [3, 5],
        [2, 3, 4],
    ])
    def test_shape_preserved(self, OpFactory, shape):
        name = uid("unary")
        op = OpFactory(name)
        x = make_tensor(name=uid("x"), shape=shape, dtype='float32')
        y = op(x)
        assert y.shape == shape
        assert y.dtype == x.dtype

    @pytest.mark.unit
    def test_softmax_scalar(self):
        """ONNX Softmax requires rank >= 1 (no axis exists for scalar input)."""
        op = F.Softmax(uid("softmax"))
        x = make_tensor(name=uid("x"), shape=[], dtype='float32')
        with pytest.raises(ValueError):
            op(x)

    @pytest.mark.unit
    @pytest.mark.parametrize("perm, expected", [
        ([1, 0],    [5, 3]),
        ([0, 1],    [3, 5]),
    ])
    def test_transpose_2d(self, perm, expected):
        op = F.Transpose(uid("transpose"), perm=perm)
        x = make_tensor(name=uid("x"), shape=[3, 5], dtype='float32')
        y = op(x)
        assert y.shape == expected

    @pytest.mark.unit
    def test_transpose_3d(self):
        op = F.Transpose(uid("transpose"), perm=[0, 2, 1])
        x = make_tensor(name=uid("x"), shape=[2, 3, 4], dtype='float32')
        y = op(x)
        assert y.shape == [2, 4, 3]

    @pytest.mark.unit
    def test_transpose_4d_attention(self):
        """Transpose as used in attention: [B, S, nH, dH] → [B, nH, S, dH]."""
        op = F.Transpose(uid("transpose"), perm=[0, 2, 1, 3])
        x = make_tensor(name=uid("x"), shape=[1, 9, 3, 16], dtype='float32')
        y = op(x)
        assert y.shape == [1, 3, 9, 16]

    @pytest.mark.unit
    def test_split_even(self):
        """Split a tensor evenly along axis 2 into 3 parts."""
        op = F.Split(uid("split"), num_outputs=3, axis=2)
        x = make_tensor(name=uid("x"), shape=[1, 9, 144], dtype='float32')
        q, k, v = op(x)
        assert q.shape == [1, 9, 48]
        assert k.shape == [1, 9, 48]
        assert v.shape == [1, 9, 48]

    @pytest.mark.unit
    def test_split_axis0(self):
        op = F.Split(uid("split"), num_outputs=2, axis=0)
        x = make_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        a, b = op(x)
        assert a.shape == [2, 8]
        assert b.shape == [2, 8]


class TestSoftmaxDimAlias:
    """Task 014 Phase 1 — PyTorch `dim=` alias for the Softmax `axis` attr.

    The alias must:
      * write the value under the canonical ONNX key on op.attrs;
      * normalize to a non-negative axis via softmax_sinf;
      * match the legacy `axis=` path bit-for-bit (modulo the deprecation
        warning the legacy path now emits).
    """

    @pytest.mark.unit
    def test_softmax_dim_alias_neg_normalized(self):
        """`dim=-1` on rank-3 input -> op.attrs['axis'] == 2 (normalized)."""
        op_h = F.Softmax(uid("softmax_dim_neg"), dim=-1)
        x = make_tensor(name=uid("x"), shape=[2, 4, 8], dtype='float32')
        y = op_h(x)
        assert y.shape == [2, 4, 8]
        assert op_h.op.attrs['axis'] == 2

    @pytest.mark.unit
    def test_softmax_dim_alias_rank4(self):
        """`dim=1` on rank-4 input -> op.attrs['axis'] == 1."""
        op_h = F.Softmax(uid("softmax_dim_pos"), dim=1)
        x = make_tensor(name=uid("x"), shape=[2, 4, 6, 8], dtype='float32')
        y = op_h(x)
        assert y.shape == [2, 4, 6, 8]
        assert op_h.op.attrs['axis'] == 1

    @pytest.mark.unit
    def test_softmax_axis_emits_deprecation_warning(self):
        """Backward compat: `axis=` still works but emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning) as recw:
            op_h = F.Softmax(uid("softmax_axis_legacy"), axis=-1)
        x = make_tensor(name=uid("x"), shape=[2, 4, 8], dtype='float32')
        y = op_h(x)
        assert y.shape == [2, 4, 8]
        assert op_h.op.attrs['axis'] == 2
        assert any("'dim'" in str(w.message) and "'axis'" in str(w.message)
                   for w in recw), f"deprecation msg missing names: {[str(w.message) for w in recw]}"

    @pytest.mark.unit
    def test_softmax_dim_matches_axis(self):
        """`dim=1` and `axis=1` must produce identical op.attrs['axis']."""
        op_dim = F.Softmax(uid("sm_dim"), dim=1)
        with pytest.warns(DeprecationWarning):
            op_axis = F.Softmax(uid("sm_axis"), axis=1)
        x1 = make_tensor(name=uid("x"), shape=[2, 4, 6, 8], dtype='float32')
        x2 = make_tensor(name=uid("x"), shape=[2, 4, 6, 8], dtype='float32')
        op_dim(x1)
        op_axis(x2)
        assert op_dim.op.attrs['axis'] == op_axis.op.attrs['axis'] == 1

    @pytest.mark.unit
    def test_softmax_dim_and_axis_raises(self):
        """Passing both `dim=` and `axis=` raises TypeError naming both."""
        with pytest.raises(TypeError) as exc:
            F.Softmax(uid("sm_collide"), dim=-1, axis=-1)
        msg = str(exc.value)
        assert "'dim'" in msg and "'axis'" in msg
        assert "Softmax" in msg
