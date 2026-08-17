"""Functional layer — variadic ops (Concat, Slice, Trilu, TopK)."""
import pytest
import numpy as np

from src import make_tensor
import src.front.functional as F


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestVariadicOps:
    """Variadic-input ops: Concat, Slice, Trilu, TopK."""

    @pytest.mark.unit
    def test_concat_axis1(self):
        """Concat requires 'axis' in op.attrs; Concat registration lacks attrs
        so we construct TensorOpHandle manually to pass axis correctly."""
        name = uid("concat")
        op = F.UniversalOperator(
            name, optype="Concat", params=[], ipos=[(2, float("inf"))]
        )
        # Manually inject axis into the op's attrs since registration omits it
        op.op.attrs['axis'] = 1
        a = make_tensor(name=uid("a"), shape=[2, 3, 4], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[2, 5, 4], dtype='float32')
        y = op(a, b)
        assert y.shape == [2, 8, 4]

    @pytest.mark.unit
    def test_concat_axis0_two_tensors(self):
        """Concat two tensors along axis 0.
        NOTE: variadic input_range uses tuple membership (not range), so only
        counts that appear literally in the ipos tuple are accepted."""
        name = uid("concat2")
        op = F.UniversalOperator(
            name, optype="Concat", params=[], ipos=[(2, float("inf"))]
        )
        op.op.attrs['axis'] = 0
        a = make_tensor(name=uid("a"), shape=[2, 4], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[3, 4], dtype='float32')
        y = op(a, b)
        assert y.shape == [5, 4]

    @pytest.mark.unit
    def test_slice_3input(self):
        """Slice with only 3 inputs (data, starts, ends). Per ONNX semantics,
        default axes is [0, 1, ..., len(starts)-1], so starts/ends of shape [1]
        slices only axis 0."""
        op = F.Slice(uid("slice"))
        x = make_tensor(name=uid("x"), shape=[4, 6], dtype='float32')
        starts = make_tensor(name=uid("starts"), shape=[1], data=np.array([0], dtype=np.int64), dtype='int64', is_const=True)
        ends   = make_tensor(name=uid("ends"),   shape=[1], data=np.array([2], dtype=np.int64), dtype='int64', is_const=True)
        y = op(x, starts, ends)
        assert y.shape == [2, 6]

    @pytest.mark.unit
    def test_slice_single_axis(self):
        """Slice a 1D tensor with 3 inputs (data, starts, ends).
        For 3-input Slice, default axes cover all dims, so the input must be 1D
        for starts/ends of shape [1] to match."""
        op = F.Slice(uid("slice"))
        x = make_tensor(name=uid("x"), shape=[10], dtype='float32')
        starts = make_tensor(name=uid("starts"), shape=[1], data=np.array([2], dtype=np.int64), dtype='int64', is_const=True)
        ends   = make_tensor(name=uid("ends"),   shape=[1], data=np.array([7], dtype=np.int64), dtype='int64', is_const=True)
        y = op(x, starts, ends)
        assert y.shape == [5]

    @pytest.mark.unit
    def test_slice_with_steps(self):
        """Slice with step=2 on axis 0. Requires 5 inputs but the variadic
        check `len(xargs) in (3, 6)` rejects 5."""
        op = F.Slice(uid("slice_step"))
        x      = make_tensor(name=uid("x"),      shape=[10, 5], dtype='float32')
        starts = make_tensor(name=uid("starts"), shape=[1], data=np.array([0], dtype=np.int64), dtype='int64', is_const=True)
        ends   = make_tensor(name=uid("ends"),   shape=[1], data=np.array([10], dtype=np.int64), dtype='int64', is_const=True)
        axes   = make_tensor(name=uid("axes"),   shape=[1], data=np.array([0], dtype=np.int64), dtype='int64', is_const=True)
        steps  = make_tensor(name=uid("steps"),  shape=[1], data=np.array([2], dtype=np.int64), dtype='int64', is_const=True)
        y = op(x, starts, ends, axes, steps)
        assert y.shape == [5, 5]

    @pytest.mark.unit
    def test_trilu_upper(self):
        """Trilu with one input produces same-shape output."""
        op = F.Trilu(uid("trilu"))
        x = make_tensor(name=uid("x"), shape=[3, 3], dtype='float32')
        y = op(x)
        assert y.shape == [3, 3]

    @pytest.mark.unit
    @pytest.mark.parametrize("x_shape, k, axis, expected", [
        ([10],          3,  -1, [3]),
        ([4, 8],        3,  -1, [4, 3]),
        ([4, 8],        2,   0, [2, 8]),
        ([2, 3, 16],    4,  -1, [2, 3, 4]),
        ([2, 3, 16],    1,   1, [2, 1, 16]),
        ([2, 4, 6, 8],  3,   2, [2, 4, 3, 8]),
    ])
    def test_topk_shape(self, x_shape, k, axis, expected):
        """F.TopK returns (values, indices); both have axis dim replaced by k."""
        op = F.TopK(uid("topk"), num_outputs=2, dim=axis)
        x = make_tensor(name=uid("x"), shape=x_shape, dtype='float32')
        k_tensor = make_tensor(
                name=uid("k"),
                shape=[1],
                data=np.array([k], dtype=np.int64),
                dtype='int64',
                is_const=True,
                )
        values, indices = op(x, k_tensor)
        assert values.shape  == expected
        assert indices.shape == expected

    @pytest.mark.unit
    def test_topk_dtype(self):
        """Values inherit X.dtype; Indices are INT64."""
        from src.utils.data_types import DataType
        op = F.TopK(uid("topk_dtype"), num_outputs=2, dim=-1)
        x = make_tensor(name=uid("x"), shape=[4, 8], dtype='bfloat16')
        k_tensor = make_tensor(
                name=uid("k"),
                shape=[1],
                data=np.array([2], dtype=np.int64),
                dtype='int64',
                is_const=True,
                )
        values, indices = op(x, k_tensor)
        assert values.dtype  == DataType.BFLOAT16
        assert indices.dtype == DataType.INT64

    @pytest.mark.unit
    def test_topk_default_axis(self):
        """When 'axis' is omitted, default is -1 (last dim)."""
        op = F.TopK(uid("topk_def"), num_outputs=2)
        x = make_tensor(name=uid("x"), shape=[2, 4, 8], dtype='float32')
        k_tensor = make_tensor(
                name=uid("k"),
                shape=[1],
                data=np.array([3], dtype=np.int64),
                dtype='int64',
                is_const=True,
                )
        values, indices = op(x, k_tensor)
        assert values.shape  == [2, 4, 3]
        assert indices.shape == [2, 4, 3]


class TestTopKDimAlias:
    """Task 014 Phase 1 — PyTorch `dim=` alias for the TopK `axis` attr."""

    @pytest.mark.unit
    def test_topk_dim_alias(self):
        """`F.TopK(..., dim=2)` on rank-3 input -> op.attrs['axis'] == 2
        and output shape matches the legacy `axis=` path."""
        op_dim = F.TopK(uid("topk_dim"), num_outputs=2, dim=2)
        with pytest.warns(DeprecationWarning):
            op_axis = F.TopK(uid("topk_axis"), num_outputs=2, axis=2)
        x1 = make_tensor(name=uid("x"), shape=[2, 3, 16], dtype='float32')
        x2 = make_tensor(name=uid("x"), shape=[2, 3, 16], dtype='float32')
        k1 = make_tensor(name=uid("k"), shape=[1],
                         data=np.array([4], dtype=np.int64),
                         dtype='int64', is_const=True)
        k2 = make_tensor(name=uid("k"), shape=[1],
                         data=np.array([4], dtype=np.int64),
                         dtype='int64', is_const=True)
        v1, i1 = op_dim(x1, k1)
        v2, i2 = op_axis(x2, k2)
        assert v1.shape == v2.shape == [2, 3, 4]
        assert i1.shape == i2.shape == [2, 3, 4]
        assert op_dim.op.attrs['axis'] == op_axis.op.attrs['axis'] == 2

    @pytest.mark.unit
    def test_topk_dim_and_axis_raises(self):
        """Passing both `dim=` and `axis=` raises TypeError."""
        with pytest.raises(TypeError) as exc:
            F.TopK(uid("topk_collide"), num_outputs=2, dim=1, axis=2)
        msg = str(exc.value)
        assert "'dim'" in msg and "'axis'" in msg
        assert "TopK" in msg
