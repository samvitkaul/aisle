"""Functional layer — binary ops (Add/Sub/Mul/Div/Pow, MatMul, Gather, Reshape, Unsqueeze, Squeeze)."""
import pytest
import numpy as np

from src import make_tensor
import src.front.functional as F


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestBinaryOps:
    """Binary ops: elementwise (broadcast-aware), MatMul, Gather, Reshape."""

    @pytest.mark.unit
    @pytest.mark.parametrize("OpFactory", [F.Add, F.Sub, F.Mul, F.Div, F.Pow])
    def test_elementwise_same_shape(self, OpFactory):
        op = OpFactory(uid("binop"))
        a = make_tensor(name=uid("a"), shape=[2, 3], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[2, 3], dtype='float32')
        y = op(a, b)
        assert y.shape == [2, 3]

    @pytest.mark.unit
    @pytest.mark.parametrize("OpFactory", [F.Add, F.Sub, F.Mul, F.Div])
    def test_elementwise_broadcast(self, OpFactory):
        op = OpFactory(uid("bcast"))
        a = make_tensor(name=uid("a"), shape=[2, 3, 4], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[4], dtype='float32')
        y = op(a, b)
        assert y.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_add_broadcast_scalar(self):
        op = F.Add(uid("add_sc"))
        a = make_tensor(name=uid("a"), shape=[3, 5], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[], dtype='float32')
        y = op(a, b)
        assert y.shape == [3, 5]

    @pytest.mark.unit
    @pytest.mark.parametrize("shA, shB, expected", [
        ([3, 4],       [4, 5],       [3, 5]),
        ([4],          [4, 5],       [5]),
        ([3, 4],       [4],          [3]),
        ([2, 3, 4],    [2, 4, 5],    [2, 3, 5]),
        ([2, 3, 4, 5], [2, 3, 5, 6], [2, 3, 4, 6]),
    ])
    def test_matmul_shapes(self, shA, shB, expected):
        op = F.MatMul(uid("mm"))
        a = make_tensor(name=uid("a"), shape=shA, dtype='float32')
        b = make_tensor(name=uid("b"), shape=shB, dtype='float32')
        y = op(a, b)
        assert y.shape == expected

    @pytest.mark.unit
    def test_matmul_incompatible(self):
        op = F.MatMul(uid("mm_neg"))
        a = make_tensor(name=uid("a"), shape=[3, 4], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[5, 6], dtype='float32')
        with pytest.raises((ValueError, AssertionError)):
            op(a, b)

    @pytest.mark.unit
    def test_gather_basic(self):
        """Gather rows from a 2D table (embedding-like)."""
        op = F.Gather(uid("gather"))
        table = make_tensor(name=uid("tbl"), shape=[100, 64], dtype='float32')
        idx_data = np.array([0, 5, 10], dtype=np.int64)
        indices = make_tensor(name=uid("idx"), shape=[3], data=idx_data, dtype='int64')
        y = op(table, indices)
        assert y.shape is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("input_shape, target, expected", [
        ([2, 3, 4], [24],         [24]),
        ([2, 3, 4], [4, 3, 2],    [4, 3, 2]),
        ([2, 3, 4], [2, -1],      [2, 12]),
        ([2, 3, 4], [2, -1, 2],   [2, 6, 2]),
        ([2, 3, 4], [2, 12],      [2, 12]),
    ])
    def test_reshape(self, input_shape, target, expected):
        op = F.Reshape(uid("reshape"))
        s_data = np.array(target, dtype=np.int64)
        x = make_tensor(name=uid("x"), shape=input_shape, dtype='float32')
        s = make_tensor(name=uid("s"), shape=list(s_data.shape), data=s_data, dtype='int64', is_const=True)
        y = op(x, s)
        assert y.shape == expected

    @pytest.mark.unit
    def test_unsqueeze(self):
        op = F.Unsqueeze(uid("unsq"))
        x = make_tensor(name=uid("x"), shape=[3, 4], dtype='float32')
        axes_data = np.array([0, 3], dtype=np.int64)
        axes = make_tensor(name=uid("axes"), shape=[2], data=axes_data, dtype='int64', is_const=True)
        y = op(x, axes)
        assert y.shape == [1, 3, 4, 1]

    @pytest.mark.unit
    def test_squeeze(self):
        op = F.Squeeze(uid("sq"))
        x = make_tensor(name=uid("x"), shape=[1, 3, 1, 4], dtype='float32')
        axes_data = np.array([0, 2], dtype=np.int64)
        axes = make_tensor(name=uid("axes"), shape=[2], data=axes_data, dtype='int64', is_const=True)
        y = op(x, axes)
        assert y.shape == [3, 4]
