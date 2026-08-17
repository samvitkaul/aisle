"""Dynamic ops — D.cat, D.stack, D.topk — invoked from inside Module.forward().

These tests exercise ``src/front/dynamic.py`` through both:
  1) the explicit ``_trace_ctx`` push (mirrors the existing convention in
     ``test_tensor_ops.py``), and
  2) a real ``Module.__call__`` path so that the trace-context push/pop
     wrapper is exercised, which is the way user code actually drives these
     helpers (see ``workloads/BasicLLM.py``, ``workloads/BasicMoE.py``).
"""
import pytest

from src.utils.data_types import DataType
from src.front.tensor import make_front_tensor
import src.front.module as nn
import src.front.dynamic as D
import numpy as np


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DynTestModule(nn.Module):
    """Bare Module used as the dynamic-op host."""
    def __init__(self, name):
        super().__init__(name)


def _make_linked_tensor(shape, dtype='float32', module=None):
    """Create a FrontTensor registered with `module` and push the trace ctx
    so dynamic ops can find an active module. Mirrors the helper in
    tests/test_front/test_tensor_ops.py.
    """
    if module is None:
        module = _DynTestModule(uid("dmod"))
    t = make_front_tensor(name=uid("dt"), shape=list(shape), dtype=dtype)
    module._tensors[t.name] = t
    from src.front.module import _trace_ctx
    _trace_ctx.current_module = module
    return t, module


def _clear_trace_ctx():
    from src.front.module import _trace_ctx
    _trace_ctx.current_module = None


# ===================================================================
# D.cat — variadic concat helper
# ===================================================================

class TestDynCat:

    @pytest.mark.unit
    def test_cat_axis0_two_tensors(self):
        m = _DynTestModule(uid("m_cat"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        y = D.cat([a, b], dim=0)
        assert y.shape == [5, 4]
        assert y.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_cat_axis_middle(self):
        m = _DynTestModule(uid("m_cat"))
        a, _ = _make_linked_tensor([2, 3, 4], module=m)
        b, _ = _make_linked_tensor([2, 5, 4], module=m)
        y = D.cat([a, b], dim=1)
        assert y.shape == [2, 8, 4]

    @pytest.mark.unit
    def test_cat_three_tensors(self):
        m = _DynTestModule(uid("m_cat3"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        c, _ = _make_linked_tensor([7, 4], module=m)
        y = D.cat([a, b, c], dim=0)
        assert y.shape == [12, 4]

    @pytest.mark.unit
    def test_cat_negative_dim(self):
        """Negative dim should be normalized inside concat_sinf."""
        m = _DynTestModule(uid("m_cat_neg"))
        a, _ = _make_linked_tensor([2, 3, 4], module=m)
        b, _ = _make_linked_tensor([2, 3, 5], module=m)
        y = D.cat([a, b], dim=-1)
        assert y.shape == [2, 3, 9]

    @pytest.mark.unit
    def test_cat_dtype_propagation(self):
        m = _DynTestModule(uid("m_cat_dt"))
        a, _ = _make_linked_tensor([2, 4], dtype='bfloat16', module=m)
        b, _ = _make_linked_tensor([3, 4], dtype='bfloat16', module=m)
        y = D.cat([a, b], dim=0)
        assert y.dtype == DataType.BFLOAT16

    @pytest.mark.unit
    def test_cat_non_tensor_raises(self):
        m = _DynTestModule(uid("m_cat_x"))
        _make_linked_tensor([2], module=m)  # push trace ctx
        with pytest.raises(TypeError, match="not a Tensor"):
            D.cat(["not_a_tensor"], dim=0)

    @pytest.mark.unit
    def test_cat_empty_list_raises(self):
        """An empty list should be rejected. Today the underlying Concat
        arity check catches it; this test pins the externally observable
        behaviour."""
        m = _DynTestModule(uid("m_cat_e"))
        _make_linked_tensor([2], module=m)
        with pytest.raises((ValueError, TypeError)):
            D.cat([], dim=0)

    @pytest.mark.unit
    def test_cat_single_input_raises(self):
        """A single-element list is rejected by the Concat arity check
        (min 2 inputs). Pin this so the contract is explicit."""
        m = _DynTestModule(uid("m_cat_one"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        with pytest.raises(ValueError):
            D.cat([a], dim=0)

    @pytest.mark.unit
    def test_cat_dim_out_of_range_raises(self):
        m = _DynTestModule(uid("m_cat_oob"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        with pytest.raises(ValueError, match="out of bounds|range"):
            D.cat([a, b], dim=5)

    @pytest.mark.unit
    def test_cat_mismatched_non_axis_dim_raises(self):
        m = _DynTestModule(uid("m_cat_mis"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 5], module=m)
        with pytest.raises(ValueError, match="Incompatible|match"):
            D.cat([a, b], dim=0)

    @pytest.mark.unit
    def test_cat_no_active_module_raises(self):
        _clear_trace_ctx()
        # Build tensors with no module context active.
        a = make_front_tensor(name=uid("a"), shape=[2, 4], dtype='float32')
        b = make_front_tensor(name=uid("b"), shape=[3, 4], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            D.cat([a, b], dim=0)

    @pytest.mark.unit
    def test_cat_registers_op_in_module(self):
        m = _DynTestModule(uid("m_cat_reg"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        ops_before = set(m._op_hndls.keys())
        y = D.cat([a, b], dim=0)
        ops_after = set(m._op_hndls.keys())
        # Exactly one new op handle (the Concat) was added.
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        new_op_name = next(iter(new_ops))
        assert 'cat' in new_op_name

    @pytest.mark.unit
    def test_cat_parent_child_wiring(self):
        """Inputs see the Concat op as a consumer; output sees Concat as
        producer."""
        m = _DynTestModule(uid("m_cat_wire"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        y = D.cat([a, b], dim=0)
        # Output tensor has the Concat op as its producer.
        assert len(y.op_out) == 1
        producer = y.op_out[0]
        # Both inputs have the Concat op as a consumer.
        assert producer in a.op_in
        assert producer in b.op_in
        # Producer op's inList/outList reference the right tensors.
        op = m._op_hndls[producer].op
        assert a.name in op.inList
        assert b.name in op.inList
        assert y.name in op.outList

    @pytest.mark.unit
    def test_cat_repeatability_in_module_forward(self):
        """Two cat calls inside the same forward() must produce unique op
        names — exercises the DynName counter."""
        class TwoCatMod(_DynTestModule):
            def forward(self, a, b):
                y1 = D.cat([a, b], dim=0)
                y2 = D.cat([a, b], dim=0)
                return y1, y2

        # Explicitly clear the trace ctx so we can assert the push/pop
        # contract precisely (prior tests may have left a module set).
        _clear_trace_ctx()
        m = TwoCatMod(uid("m_cat_two"))
        a = make_front_tensor(name=uid("a"), shape=[2, 4], dtype='float32')
        b = make_front_tensor(name=uid("b"), shape=[3, 4], dtype='float32')
        y1, y2 = m(a, b)
        assert y1.name != y2.name
        assert y1.op_out[0] != y2.op_out[0]
        # Trace ctx must have been popped after __call__.
        from src.front.module import _trace_ctx
        assert getattr(_trace_ctx, 'current_module', None) is None


# ===================================================================
# D.stack — multi-step (Unsqueeze-per-input + Concat) variadic helper
# ===================================================================

class TestDynStack:
    """Tests for ``D.stack``: per-input Unsqueeze + final Concat.

    Historical note: an earlier `tt`/`x` typo in ``src/front/dynamic.py``
    made every ``stack`` test below raise ``NameError`` inside the
    per-input loop. The typo is fixed; these tests now pin the
    *positive* behavior (shapes, dtype propagation, wiring), and the
    end-to-end variants in ``TestDynamicOpsInModuleForward`` pin the
    fact that the shared ``axesTensor`` does not introduce spurious
    edges into the constructed graph.
    """

    @pytest.mark.unit
    def test_stack_basic_new_axis0(self):
        m = _DynTestModule(uid("m_stk"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([2, 4], module=m)
        y = D.stack([a, b], dim=0)
        # Stack adds a new leading axis -> [2, 2, 4]
        assert y.shape == [2, 2, 4]
        assert y.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_stack_axis_last(self):
        m = _DynTestModule(uid("m_stk2"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([2, 4], module=m)
        c, _ = _make_linked_tensor([2, 4], module=m)
        y = D.stack([a, b, c], dim=-1)
        assert y.shape == [2, 4, 3]

    @pytest.mark.unit
    def test_stack_middle_axis(self):
        m = _DynTestModule(uid("m_stk3"))
        ts = [_make_linked_tensor([3, 5], module=m)[0] for _ in range(4)]
        y = D.stack(ts, dim=1)
        assert y.shape == [3, 4, 5]

    @pytest.mark.unit
    def test_stack_empty_list_raises(self):
        m = _DynTestModule(uid("m_stk_e"))
        _make_linked_tensor([2], module=m)
        with pytest.raises(ValueError, match="empty"):
            D.stack([], dim=0)

    @pytest.mark.unit
    def test_stack_non_list_raises(self):
        m = _DynTestModule(uid("m_stk_nl"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        with pytest.raises(TypeError, match="not a list"):
            D.stack((a, a), dim=0)  # tuple, not list

    @pytest.mark.unit
    def test_stack_shape_mismatch_raises(self):
        m = _DynTestModule(uid("m_stk_sm"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([2, 5], module=m)
        with pytest.raises(ValueError, match="same shape"):
            D.stack([a, b], dim=0)

    @pytest.mark.unit
    def test_stack_dtype_promotion(self):
        """Task 048.5(c): stack() policy aligned with Phase 048.3 concat_sinf
        promotion. Mixed-dtype inputs no longer raise — the internal
        unsqueeze+concat lower folds promote_types across all inputs
        (fp32 + bf16 -> fp32 per PyTorch tensor-tensor rules)."""
        m = _DynTestModule(uid("m_stk_dm"))
        a, _ = _make_linked_tensor([2, 4], dtype='float32', module=m)
        b, _ = _make_linked_tensor([2, 4], dtype='bfloat16', module=m)
        y = D.stack([a, b], dim=0)
        assert y.shape == [2, 2, 4]
        assert y.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_stack_dim_out_of_range_raises(self):
        m = _DynTestModule(uid("m_stk_oob"))
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([2, 4], module=m)
        # Rank is 2, valid dim range with stack is [-3, 2]; 5 is OOR.
        with pytest.raises(ValueError, match="out of range"):
            D.stack([a, b], dim=5)

    @pytest.mark.unit
    def test_stack_no_active_module_raises(self):
        _clear_trace_ctx()
        a = make_front_tensor(name=uid("a"), shape=[2, 4], dtype='float32')
        b = make_front_tensor(name=uid("b"), shape=[2, 4], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            D.stack([a, b], dim=0)

    @pytest.mark.unit
    def test_stack_non_tensor_raises(self):
        m = _DynTestModule(uid("m_stk_nt"))
        _make_linked_tensor([2], module=m)
        with pytest.raises(TypeError, match="not a Tensor"):
            D.stack(["not_a_tensor"], dim=0)

    @pytest.mark.unit
    def test_stack_shared_axes_tensor_registered_once(self):
        """``stack`` creates a single ``axes`` const-tensor shared by every
        Unsqueeze. Verify it is registered exactly once in the module
        and that its ``op_in`` lists every Unsqueeze as a consumer
        (one entry per input tensor, no duplicates)."""
        m = _DynTestModule(uid("m_stk_shared"))
        ts = [_make_linked_tensor([2, 4], module=m)[0] for _ in range(3)]
        _ = D.stack(ts, dim=0)

        axes_tensors = [t for n, t in m._tensors.items()
                        if n.endswith('.unsqueeze.axes')]
        assert len(axes_tensors) == 1, (
            f"Expected exactly one shared axes tensor, got {len(axes_tensors)}")
        axes_t = axes_tensors[0]
        assert axes_t.is_const
        assert list(axes_t.shape) == [1]
        assert int(axes_t.data[0]) == 0

        # axes_t must be consumed by all three Unsqueeze ops, no duplicates.
        unsqueeze_ops = [n for n in m._op_hndls.keys() if 'unsqueeze_' in n]
        assert len(unsqueeze_ops) == 3
        assert sorted(axes_t.op_in) == sorted(unsqueeze_ops)
        # Distinct entries — sanity check that we are not accidentally
        # double-appending the same op name to op_in.
        assert len(axes_t.op_in) == len(set(axes_t.op_in))


# ===================================================================
# D.topk
# ===================================================================

class TestDynTopK:

    @pytest.mark.unit
    def test_topk_default_dim(self):
        """Default axis is -1 (last dim)."""
        m = _DynTestModule(uid("m_tk"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = D.topk(x, k=3)
        assert values.shape == [4, 3]
        assert indices.shape == [4, 3]

    @pytest.mark.unit
    def test_topk_explicit_dim_via_alias(self):
        """`dim=` is the PyTorch alias for ONNX `axis`."""
        m = _DynTestModule(uid("m_tk_dim"))
        x, _ = _make_linked_tensor([2, 3, 16], module=m)
        values, indices = D.topk(x, k=4, dim=2)
        assert values.shape == [2, 3, 4]
        assert indices.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_topk_negative_dim(self):
        m = _DynTestModule(uid("m_tk_neg"))
        x, _ = _make_linked_tensor([2, 4, 8], module=m)
        values, indices = D.topk(x, k=2, dim=-2)
        assert values.shape == [2, 2, 8]
        assert indices.shape == [2, 2, 8]

    @pytest.mark.unit
    def test_topk_dtype_propagation(self):
        """Values inherit input dtype; indices are INT64."""
        m = _DynTestModule(uid("m_tk_dt"))
        x, _ = _make_linked_tensor([4, 8], dtype='bfloat16', module=m)
        values, indices = D.topk(x, k=2)
        assert values.dtype == DataType.BFLOAT16
        assert indices.dtype == DataType.INT64

    @pytest.mark.unit
    def test_topk_k_equal_dim_size(self):
        """k == axis-size is allowed."""
        m = _DynTestModule(uid("m_tk_ke"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = D.topk(x, k=8)
        assert values.shape == [4, 8]

    @pytest.mark.unit
    def test_topk_k_too_large_raises(self):
        m = _DynTestModule(uid("m_tk_kl"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="K value"):
            D.topk(x, k=99)

    @pytest.mark.unit
    def test_topk_k_negative_raises(self):
        m = _DynTestModule(uid("m_tk_kn"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="K value"):
            D.topk(x, k=-1)

    @pytest.mark.unit
    def test_topk_dim_out_of_range_raises(self):
        m = _DynTestModule(uid("m_tk_oob"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="out of bounds|range"):
            D.topk(x, k=3, dim=5)

    @pytest.mark.unit
    def test_topk_no_active_module_raises(self):
        _clear_trace_ctx()
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            D.topk(x, k=3)

    @pytest.mark.unit
    def test_topk_registers_op_in_module(self):
        m = _DynTestModule(uid("m_tk_reg"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        ops_before = set(m._op_hndls.keys())
        D.topk(x, k=3)
        ops_after = set(m._op_hndls.keys())
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        assert 'topk' in next(iter(new_ops))

    @pytest.mark.unit
    def test_topk_parent_child_wiring(self):
        m = _DynTestModule(uid("m_tk_wire"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = D.topk(x, k=3)
        # both outputs share the same producer op
        assert len(values.op_out) == 1
        assert len(indices.op_out) == 1
        assert values.op_out[0] == indices.op_out[0]
        producer = values.op_out[0]
        # input sees the TopK op as a consumer
        assert producer in x.op_in
        # op's inList contains the data tensor and the k-tensor
        op = m._op_hndls[producer].op
        assert x.name in op.inList
        assert any(name.endswith('.k') for name in op.inList)
        # op's outList contains both output tensors
        assert values.name in op.outList
        assert indices.name in op.outList

    @pytest.mark.unit
    def test_topk_k_tensor_registered(self):
        """The dynamically created `k` const-tensor must end up in
        module._tensors so get_forward_graph() can bind it."""
        m = _DynTestModule(uid("m_tk_kt"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        D.topk(x, k=3)
        k_tensors = [t for n, t in m._tensors.items() if n.endswith('.k')]
        assert len(k_tensors) == 1
        kt = k_tensors[0]
        assert kt.is_const
        assert kt.dtype == DataType.INT64
        assert list(kt.shape) == [1]
        assert int(kt.data[0]) == 3

    @pytest.mark.unit
    def test_topk_repeatability_in_module_forward(self):
        """Two topk calls in one forward() pass must yield distinct op
        names and distinct output tensors."""
        class TwoTopK(_DynTestModule):
            def forward(self, x):
                v1, i1 = D.topk(x, k=2)
                v2, i2 = D.topk(x, k=3)
                return v1, i1, v2, i2

        m = TwoTopK(uid("m_tk_two"))
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        v1, i1, v2, i2 = m(x)
        assert v1.name != v2.name
        assert i1.name != i2.name
        assert v1.shape == [4, 2]
        assert v2.shape == [4, 3]


# ===================================================================
# FrontTensor.topk — method form (x.topk(k, ...))
# ===================================================================

class TestTensorTopK:

    @pytest.mark.unit
    def test_topk_default_dim(self):
        m = _DynTestModule(uid("m_mtk"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = x.topk(k=3)
        assert values.shape == [4, 3]
        assert indices.shape == [4, 3]

    @pytest.mark.unit
    def test_topk_explicit_dim(self):
        m = _DynTestModule(uid("m_mtk_dim"))
        x, _ = _make_linked_tensor([2, 3, 16], module=m)
        values, indices = x.topk(k=4, dim=2)
        assert values.shape == [2, 3, 4]
        assert indices.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_topk_explicit_axis(self):
        m = _DynTestModule(uid("m_mtk_axis"))
        x, _ = _make_linked_tensor([2, 3, 16], module=m)
        with pytest.warns(DeprecationWarning, match="ONNX-canonical"):
            values, indices = x.topk(k=4, axis=2)
        assert values.shape == [2, 3, 4]
        assert indices.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_topk_negative_dim(self):
        m = _DynTestModule(uid("m_mtk_neg"))
        x, _ = _make_linked_tensor([2, 4, 8], module=m)
        values, indices = x.topk(k=2, dim=-2)
        assert values.shape == [2, 2, 8]
        assert indices.shape == [2, 2, 8]

    @pytest.mark.unit
    def test_topk_dtype_propagation(self):
        m = _DynTestModule(uid("m_mtk_dt"))
        x, _ = _make_linked_tensor([4, 8], dtype='bfloat16', module=m)
        values, indices = x.topk(k=2)
        assert values.dtype == DataType.BFLOAT16
        assert indices.dtype == DataType.INT64

    @pytest.mark.unit
    def test_topk_k_equal_dim_size(self):
        m = _DynTestModule(uid("m_mtk_ke"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = x.topk(k=8)
        assert values.shape == [4, 8]

    @pytest.mark.unit
    def test_topk_k_too_large_raises(self):
        m = _DynTestModule(uid("m_mtk_kl"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="K value"):
            x.topk(k=99)

    @pytest.mark.unit
    def test_topk_k_negative_raises(self):
        m = _DynTestModule(uid("m_mtk_kn"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="K value"):
            x.topk(k=-1)

    @pytest.mark.unit
    def test_topk_dim_out_of_range_raises(self):
        m = _DynTestModule(uid("m_mtk_oob"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="out of bounds|range"):
            x.topk(k=3, dim=5)

    @pytest.mark.unit
    def test_topk_no_active_module_raises(self):
        _clear_trace_ctx()
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            x.topk(k=3)

    @pytest.mark.unit
    def test_topk_registers_op_in_module(self):
        m = _DynTestModule(uid("m_mtk_reg"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        ops_before = set(m._op_hndls.keys())
        x.topk(k=3)
        ops_after = set(m._op_hndls.keys())
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        assert 'topk' in next(iter(new_ops))

    @pytest.mark.unit
    def test_topk_k_tensor_registered(self):
        m = _DynTestModule(uid("m_mtk_kt"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        x.topk(k=3)
        k_tensors = [t for n, t in m._tensors.items() if n.endswith('.k')]
        assert len(k_tensors) == 1
        kt = k_tensors[0]
        assert kt.is_const
        assert kt.dtype == DataType.INT64
        assert list(kt.shape) == [1]
        assert int(kt.data[0]) == 3

    @pytest.mark.unit
    def test_topk_parent_child_wiring(self):
        m = _DynTestModule(uid("m_mtk_wire"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        values, indices = x.topk(k=3)
        assert len(values.op_out) == 1
        assert len(indices.op_out) == 1
        assert values.op_out[0] == indices.op_out[0]
        producer = values.op_out[0]
        assert producer in x.op_in
        op = m._op_hndls[producer].op
        assert x.name in op.inList
        assert any(name.endswith('.k') for name in op.inList)
        assert values.name in op.outList
        assert indices.name in op.outList

    @pytest.mark.unit
    def test_topk_repeatability_in_module_forward(self):
        class TwoTopK(_DynTestModule):
            def forward(self, x):
                v1, i1 = x.topk(k=2)
                v2, i2 = x.topk(k=3)
                return v1, i1, v2, i2

        m = TwoTopK(uid("m_mtk_two"))
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        v1, i1, v2, i2 = m(x)
        assert v1.name != v2.name
        assert i1.name != i2.name
        assert v1.shape == [4, 2]
        assert v2.shape == [4, 3]

    @pytest.mark.unit
    def test_dtopk_forwards_to_method(self):
        """D.topk() and x.topk() must produce equivalent TopK ops with
        distinct (freshly-allocated) names."""
        m = _DynTestModule(uid("m_tk_fwd"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        v_d, i_d = D.topk(x, k=3)
        v_m, i_m = x.topk(k=3)

        prod_d = v_d.op_out[0]
        prod_m = v_m.op_out[0]
        assert prod_d != prod_m

        op_d = m._op_hndls[prod_d]
        op_m = m._op_hndls[prod_m]
        assert op_d.op.optype == 'TopK'
        assert op_m.op.optype == 'TopK'
        assert op_d.op.attrs == op_m.op.attrs

        assert v_d.shape == v_m.shape
        assert i_d.shape == i_m.shape
        assert v_d.dtype == v_m.dtype
        assert i_d.dtype == i_m.dtype


# ===================================================================
# FrontTensor.squeeze — x.squeeze(dim=...) / x.squeeze(axes=[...])
# ===================================================================

class TestTensorSqueeze:

    @pytest.mark.unit
    def test_squeeze_dim_int(self):
        m = _DynTestModule(uid("m_sq"))
        x, _ = _make_linked_tensor([1, 4, 8], module=m)
        y = x.squeeze(dim=0)
        assert y.shape == [4, 8]

    @pytest.mark.unit
    def test_squeeze_axes_list(self):
        m = _DynTestModule(uid("m_sq_ax"))
        x, _ = _make_linked_tensor([1, 4, 1, 8], module=m)
        y = x.squeeze(axes=[0, 2])
        assert y.shape == [4, 8]

    @pytest.mark.unit
    def test_squeeze_negative_dim(self):
        m = _DynTestModule(uid("m_sq_neg"))
        x, _ = _make_linked_tensor([4, 1, 8], module=m)
        y = x.squeeze(dim=-2)
        assert y.shape == [4, 8]

    @pytest.mark.unit
    def test_squeeze_dim_and_axes_both_raises(self):
        m = _DynTestModule(uid("m_sq_both"))
        x, _ = _make_linked_tensor([1, 4, 1, 8], module=m)
        with pytest.raises(ValueError, match="exactly one of dim= or axes="):
            x.squeeze(dim=0, axes=[2])

    @pytest.mark.unit
    def test_squeeze_no_args_raises(self):
        """F.Squeeze requires an axes input tensor (BinaryOperator), so the
        tensor-method also requires either dim= or axes=."""
        m = _DynTestModule(uid("m_sq_none"))
        x, _ = _make_linked_tensor([1, 4, 1, 8], module=m)
        with pytest.raises(ValueError, match="must specify dim= or axes="):
            x.squeeze()

    @pytest.mark.unit
    def test_squeeze_op_registered_in_module(self):
        m = _DynTestModule(uid("m_sq_reg"))
        x, _ = _make_linked_tensor([1, 4, 8], module=m)
        ops_before = set(m._op_hndls.keys())
        x.squeeze(dim=0)
        ops_after = set(m._op_hndls.keys())
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        assert 'squeeze' in next(iter(new_ops))

    @pytest.mark.unit
    def test_squeeze_axes_tensor_registered(self):
        m = _DynTestModule(uid("m_sq_axt"))
        x, _ = _make_linked_tensor([1, 4, 1, 8], module=m)
        x.squeeze(axes=[0, 2])
        axes_tensors = [t for n, t in m._tensors.items()
                        if n.endswith('.axes') and 'squeeze' in n]
        assert len(axes_tensors) == 1
        at = axes_tensors[0]
        assert at.is_const
        assert at.dtype == DataType.INT64
        assert list(at.shape) == [2]
        assert list(at.data) == [0, 2]

    @pytest.mark.unit
    def test_squeeze_parent_child_wiring(self):
        m = _DynTestModule(uid("m_sq_wire"))
        x, _ = _make_linked_tensor([1, 4, 8], module=m)
        y = x.squeeze(dim=0)
        assert len(y.op_out) == 1
        producer = y.op_out[0]
        assert producer in x.op_in
        op = m._op_hndls[producer].op
        assert x.name in op.inList
        assert any(name.endswith('.axes') for name in op.inList)
        assert y.name in op.outList

    @pytest.mark.unit
    def test_squeeze_no_active_module_raises(self):
        _clear_trace_ctx()
        x = make_front_tensor(name=uid("x"), shape=[1, 4, 8], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            x.squeeze(dim=0)

    @pytest.mark.unit
    def test_squeeze_repeatability_in_module_forward(self):
        class TwoSqueeze(_DynTestModule):
            def forward(self, x):
                y1 = x.squeeze(dim=0)
                y2 = x.squeeze(dim=0)
                return y1, y2

        m = TwoSqueeze(uid("m_sq_two"))
        x = make_front_tensor(name=uid("x"), shape=[1, 4, 8], dtype='float32')
        y1, y2 = m(x)
        assert y1.name != y2.name
        producer1 = y1.op_out[0]
        producer2 = y2.op_out[0]
        assert producer1 != producer2
        axes_names = [n for n in m._tensors if n.endswith('.axes') and 'squeeze' in n]
        assert len(axes_names) == 2
        assert axes_names[0] != axes_names[1]


# ===================================================================
# FrontTensor.unsqueeze — x.unsqueeze(dim=...) / x.unsqueeze(axes=[...])
# ===================================================================

class TestTensorUnsqueeze:

    @pytest.mark.unit
    def test_unsqueeze_dim_int(self):
        m = _DynTestModule(uid("m_unsq"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.unsqueeze(dim=0)
        assert y.shape == [1, 4, 8]

    @pytest.mark.unit
    def test_unsqueeze_axes_list(self):
        """ONNX Unsqueeze: axes are positions in the OUTPUT tensor,
        applied sequentially. [4,8] + axes=[0,2] -> [1,4,1,8]."""
        m = _DynTestModule(uid("m_unsq_ax"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.unsqueeze(axes=[0, 2])
        assert y.shape == [1, 4, 1, 8]

    @pytest.mark.unit
    def test_unsqueeze_negative_dim(self):
        m = _DynTestModule(uid("m_unsq_neg"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.unsqueeze(dim=-1)
        assert y.shape == [4, 8, 1]

    @pytest.mark.unit
    def test_unsqueeze_dim_and_axes_both_raises(self):
        m = _DynTestModule(uid("m_unsq_both"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="exactly one of dim= or axes="):
            x.unsqueeze(dim=0, axes=[1])

    @pytest.mark.unit
    def test_unsqueeze_neither_dim_nor_axes_raises(self):
        m = _DynTestModule(uid("m_unsq_none"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(ValueError, match="must specify dim= or axes="):
            x.unsqueeze()

    @pytest.mark.unit
    def test_unsqueeze_op_registered_in_module(self):
        m = _DynTestModule(uid("m_unsq_reg"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        ops_before = set(m._op_hndls.keys())
        x.unsqueeze(dim=0)
        ops_after = set(m._op_hndls.keys())
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        assert 'unsqueeze' in next(iter(new_ops))

    @pytest.mark.unit
    def test_unsqueeze_axes_tensor_registered(self):
        m = _DynTestModule(uid("m_unsq_axt"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        x.unsqueeze(axes=[0, 2])
        axes_tensors = [t for n, t in m._tensors.items()
                        if n.endswith('.axes') and 'unsqueeze' in n]
        assert len(axes_tensors) == 1
        at = axes_tensors[0]
        assert at.is_const
        assert at.dtype == DataType.INT64
        assert list(at.shape) == [2]
        assert list(at.data) == [0, 2]

    @pytest.mark.unit
    def test_unsqueeze_parent_child_wiring(self):
        m = _DynTestModule(uid("m_unsq_wire"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.unsqueeze(dim=0)
        assert len(y.op_out) == 1
        producer = y.op_out[0]
        assert producer in x.op_in
        op = m._op_hndls[producer].op
        assert x.name in op.inList
        assert any(name.endswith('.axes') for name in op.inList)
        assert y.name in op.outList

    @pytest.mark.unit
    def test_unsqueeze_no_active_module_raises(self):
        _clear_trace_ctx()
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            x.unsqueeze(dim=0)

    @pytest.mark.unit
    def test_unsqueeze_repeatability_in_module_forward(self):
        class TwoUnsqueeze(_DynTestModule):
            def forward(self, x):
                y1 = x.unsqueeze(dim=0)
                y2 = x.unsqueeze(dim=0)
                return y1, y2

        m = TwoUnsqueeze(uid("m_unsq_two"))
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        y1, y2 = m(x)
        assert y1.name != y2.name
        producer1 = y1.op_out[0]
        producer2 = y2.op_out[0]
        assert producer1 != producer2
        axes_names = [n for n in m._tensors if n.endswith('.axes') and 'unsqueeze' in n]
        assert len(axes_names) == 2
        assert axes_names[0] != axes_names[1]


# ===================================================================
# FrontTensor.softmax — x.softmax(dim=-1) / x.softmax(axis=-1)
# ===================================================================

class TestTensorSoftmax:

    @pytest.mark.unit
    def test_softmax_default_dim(self):
        m = _DynTestModule(uid("m_sm"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.softmax()
        assert y.shape == [4, 8]
        assert y.dtype == DataType.FLOAT32

    @pytest.mark.unit
    def test_softmax_explicit_dim(self):
        m = _DynTestModule(uid("m_sm_dim"))
        x, _ = _make_linked_tensor([2, 3, 16], module=m)
        y = x.softmax(dim=1)
        assert y.shape == [2, 3, 16]
        producer = y.op_out[0]
        assert m._op_hndls[producer].op.attrs['axis'] == 1

    @pytest.mark.unit
    def test_softmax_negative_dim(self):
        m = _DynTestModule(uid("m_sm_neg"))
        x, _ = _make_linked_tensor([2, 4, 8], module=m)
        y = x.softmax(dim=-2)
        assert y.shape == [2, 4, 8]
        producer = y.op_out[0]
        # softmax_sinf normalizes the axis to non-negative form.
        assert m._op_hndls[producer].op.attrs['axis'] == 1

    @pytest.mark.unit
    def test_softmax_axis_kwarg(self):
        """`axis=` is the ONNX-canonical name; supported via the
        F.Softmax alias table (emits DeprecationWarning)."""
        m = _DynTestModule(uid("m_sm_ax"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.warns(DeprecationWarning):
            y = x.softmax(axis=0)
        assert y.shape == [4, 8]
        producer = y.op_out[0]
        assert m._op_hndls[producer].op.attrs['axis'] == 0

    @pytest.mark.unit
    def test_softmax_dtype_propagation(self):
        m = _DynTestModule(uid("m_sm_dt"))
        x, _ = _make_linked_tensor([4, 8], dtype='bfloat16', module=m)
        y = x.softmax()
        assert y.dtype == DataType.BFLOAT16

    @pytest.mark.unit
    def test_softmax_op_registered_in_module(self):
        m = _DynTestModule(uid("m_sm_reg"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        ops_before = set(m._op_hndls.keys())
        x.softmax()
        ops_after = set(m._op_hndls.keys())
        new_ops = ops_after - ops_before
        assert len(new_ops) == 1
        assert 'softmax' in next(iter(new_ops))

    @pytest.mark.unit
    def test_softmax_parent_child_wiring(self):
        m = _DynTestModule(uid("m_sm_wire"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x.softmax()
        assert len(y.op_out) == 1
        producer = y.op_out[0]
        assert producer in x.op_in
        # Single producer, single child of x for this op.
        assert x.op_in.count(producer) == 1
        op = m._op_hndls[producer].op
        assert op.inList == [x.name]
        assert op.outList == [y.name]

    @pytest.mark.unit
    def test_softmax_no_active_module_raises(self):
        _clear_trace_ctx()
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        with pytest.raises(RuntimeError, match="No active module context"):
            x.softmax()

    @pytest.mark.unit
    def test_softmax_repeatability_in_module_forward(self):
        class TwoSoftmax(_DynTestModule):
            def forward(self, x):
                y1 = x.softmax(dim=-1)
                y2 = x.softmax(dim=0)
                return y1, y2

        m = TwoSoftmax(uid("m_sm_two"))
        x = make_front_tensor(name=uid("x"), shape=[4, 8], dtype='float32')
        y1, y2 = m(x)
        assert y1.name != y2.name
        producer1 = y1.op_out[0]
        producer2 = y2.op_out[0]
        assert producer1 != producer2


# ===================================================================
# Driving dynamic ops from a real Module.__call__ (trace-ctx push/pop)
# ===================================================================

class TestDynamicOpsInModuleForward:
    """Drives the dynamic helpers through Module.__call__ rather than the
    explicit _trace_ctx push, so we exercise the trace-context wrapper
    used by real workloads (workloads/BasicLLM.py, GQA.py, BasicMoE.py)."""

    @pytest.mark.unit
    def test_cat_inside_forward_then_graph(self):
        class CatMod(nn.Module):
            def __init__(self, name):
                super().__init__(name)

            def forward(self, a, b):
                return D.cat([a, b], dim=1)

        mod = CatMod(uid("catmod"))
        a = make_front_tensor(name=uid("a"), shape=[2, 3, 4], dtype='float32')
        b = make_front_tensor(name=uid("b"), shape=[2, 5, 4], dtype='float32')
        y = mod(a, b)
        assert y.shape == [2, 8, 4]
        G = mod.get_forward_graph(a, b)
        assert G.get_node_count() == 1   # exactly one Concat node
        assert G.get_edge_count() == 0   # no downstream consumers

    @pytest.mark.unit
    def test_topk_inside_forward_then_graph(self):
        class TopKMod(nn.Module):
            def __init__(self, name, k):
                super().__init__(name)
                self.k = k

            def forward(self, x):
                v, _i = D.topk(x, k=self.k)
                return v

        mod = TopKMod(uid("tkmod"), k=4)
        x = make_front_tensor(name=uid("x"), shape=[2, 16], dtype='float32')
        y = mod(x)
        assert y.shape == [2, 4]
        G = mod.get_forward_graph(x)
        # exactly one TopK op was created
        assert G.get_node_count() == 1

    @pytest.mark.unit
    def test_stack_inside_forward_then_graph(self):
        """End-to-end smoke: stack inside Module.__call__, then build the
        graph. Pins (a) the previously broken `tt`/`x` path now works,
        (b) the shared `axesTensor` does NOT create spurious edges, and
        (c) the resulting graph is well-formed (3 unsqueezes + 1 concat,
        with each unsqueeze feeding the concat exactly once)."""
        N = 3

        class StackMod(nn.Module):
            def forward(self, *xs):
                return D.stack(list(xs), dim=0)

        mod = StackMod(uid("stkmod"))
        xs = [make_front_tensor(name=uid("x"), shape=[2, 4], dtype='float32')
              for _ in range(N)]
        y = mod(*xs)
        assert y.shape == [N, 2, 4]

        G = mod.get_forward_graph(*xs)
        # N Unsqueeze nodes + 1 Concat node
        assert G.get_node_count() == N + 1
        # Each Unsqueeze produces a tensor consumed only by the Concat;
        # the shared axes-tensor has no producer and contributes 0 edges.
        # Total: exactly N internal edges.
        assert G.get_edge_count() == N

        # Identify nodes
        node_names = list(G.get_ordered_nodes())
        unsqueeze_nodes = [n for n in node_names if 'unsqueeze_' in n]
        concat_nodes = [n for n in node_names if n.endswith('.concat')]
        assert len(unsqueeze_nodes) == N
        assert len(concat_nodes) == 1
        concat_node = concat_nodes[0]

        # Each unsqueeze must have exactly one successor: the concat.
        for u in unsqueeze_nodes:
            succ = G.get_successors(u)
            assert succ == [concat_node], (
                f"Unsqueeze {u} successors={succ}, expected [{concat_node}]")

        # No duplicate edges between any unsqueeze and the concat.
        # (MultiDiGraph would allow parallel edges if op_in were appended
        # twice for the same consumer — a regression to guard against.)
        for u in unsqueeze_nodes:
            n_edges = G._graph.number_of_edges(u, concat_node)
            assert n_edges == 1, (
                f"Expected exactly 1 edge {u}->{concat_node}, got {n_edges}")

    @pytest.mark.unit
    def test_multiple_forward_calls_grow_ops(self):
        """Each forward() call creates fresh op handles; module state
        grows monotonically. This pins the current behaviour so any
        change (caching, dedup) shows up here."""
        class CatMod(nn.Module):
            def forward(self, a, b):
                return D.cat([a, b], dim=0)

        mod = CatMod(uid("catmod"))
        a = make_front_tensor(name=uid("a"), shape=[2, 4], dtype='float32')
        b = make_front_tensor(name=uid("b"), shape=[3, 4], dtype='float32')
        n0 = len(mod._op_hndls)
        mod(a, b)
        n1 = len(mod._op_hndls)
        mod(a, b)
        n2 = len(mod._op_hndls)
        assert n1 == n0 + 1
        assert n2 == n0 + 2


# ===================================================================
# DynName — uniqueness & collision guard
# ===================================================================

class TestDynNameCollision:

    @pytest.mark.unit
    def test_subsequent_calls_yield_distinct_names(self):
        """Back-to-back ``DynName.get`` calls on the same module + optype
        must yield distinct names (the counter advances). Pinning this
        keeps any future caching/dedup refactor honest."""
        m = _DynTestModule(uid("m_collide"))
        n1 = D.DynName.get(m, 'op')
        m._op_hndls[n1] = object()
        n2 = D.DynName.get(m, 'op')
        assert n1 != n2

    @pytest.mark.unit
    def test_collision_actually_raises(self):
        """Force a real collision by resetting the counter so the next
        emitted name matches one already registered in ``_op_hndls``.
        ``DynName.get`` is required to raise instead of returning a
        duplicate."""
        # Reset first so `n` is emitted with a known low counter value;
        # otherwise the bounded loop below may never reach the collision.
        D.DynName.reset_counter()
        m = _DynTestModule(uid("m_collide2"))
        n = D.DynName.get(m, 'op')
        m._op_hndls[n] = object()
        # Rewind the global counter so the next emission collides.
        D.DynName.reset_counter()
        with pytest.raises(RuntimeError, match="not unique"):
            # If counters realign exactly, this raises immediately. If
            # the module name differs, walk the counter until we hit a
            # collision \u2014 bounded by the counter value that produced n.
            for _ in range(64):
                D.DynName.get(m, 'op')


# ===================================================================
# F1: FrontTensor.__getitem__ with a tensor index (single tensor, single axis)
# Lowers to a single F.Gather op via torch2onnx_slice_plan's tensor_gathers.
# ===================================================================

class TestTensorGetitemTensorIndex:

    def _idx(self, shape, module, dtype='int64'):
        """Helper: build an int64 index FrontTensor wired to `module`."""
        t = make_front_tensor(name=uid("idx"), shape=list(shape), dtype=dtype)
        module._tensors[t.name] = t
        return t

    @pytest.mark.unit
    def test_axis0_rank1_index(self):
        """tokens: [N, dE], idx: [K] -> tokens[idx]: [K, dE]."""
        m = _DynTestModule(uid("m_tg1"))
        tokens, _ = _make_linked_tensor([100, 32], module=m)
        idx = self._idx([4], m)
        y = tokens[idx]
        assert y.shape == [4, 32]

    @pytest.mark.unit
    def test_axis0_rank2_index(self):
        """tokens: [N, dE], idx: [nE, capacity] -> [nE, capacity, dE].
        Mirrors the SparseMoE dispatch shape exactly."""
        m = _DynTestModule(uid("m_tg2"))
        tokens, _ = _make_linked_tensor([100, 32], module=m)
        idx = self._idx([8, 5], m)
        y = tokens[idx]
        assert y.shape == [8, 5, 32]

    @pytest.mark.unit
    def test_axis1_rank1_index(self):
        """x: [A, B, C], x[:, idx] with idx: [K] -> [A, K, C]."""
        m = _DynTestModule(uid("m_tg3"))
        x, _ = _make_linked_tensor([4, 8, 16], module=m)
        idx = self._idx([3], m)
        y = x[:, idx]
        assert y.shape == [4, 3, 16]

    @pytest.mark.unit
    def test_dtype_propagation(self):
        """bfloat16 data + int64 index -> bfloat16 output."""
        m = _DynTestModule(uid("m_tg4"))
        tokens, _ = _make_linked_tensor([100, 32], dtype='bfloat16', module=m)
        idx = self._idx([8, 5], m)
        y = tokens[idx]
        assert y.shape == [8, 5, 32]
        assert y.dtype == DataType.BFLOAT16

    @pytest.mark.unit
    def test_op_count_is_one_gather(self):
        """tokens[idx] must lower to exactly one Gather op (no Slice, no Unsqueeze)."""
        m = _DynTestModule(uid("m_tg5"))
        tokens, _ = _make_linked_tensor([100, 32], module=m)
        idx = self._idx([8, 5], m)
        n0 = len(m._op_hndls)
        _ = tokens[idx]
        new_ops = [op for nm, op in m._op_hndls.items()][n0:]
        assert len(new_ops) == 1, f"expected 1 op, got {[o.op.optype for o in new_ops]}"
        assert new_ops[0].op.optype == 'Gather'
        assert new_ops[0].op.attrs.get('axis', 0) == 0

    @pytest.mark.unit
    def test_equivalence_with_explicit_F_Gather(self):
        """tokens[idx] (sugar) vs F.Gather(name)(tokens, idx) (explicit):
        identical output shape, dtype, and Gather axis attribute."""
        import src.front.functional as F
        m = _DynTestModule(uid("m_tg6"))
        tokens, _ = _make_linked_tensor([100, 32], module=m)
        idx = self._idx([8, 5], m)

        y_sugar = tokens[idx]

        explicit = F.Gather(uid("gth_ex"), axis=0)
        m._op_hndls[explicit.name] = explicit
        y_explicit = explicit(tokens, idx)

        assert y_sugar.shape == y_explicit.shape
        assert y_sugar.dtype == y_explicit.dtype
        # Find the sugar's Gather op and compare axis attr to the explicit one.
        sugar_gathers = [op for nm, op in m._op_hndls.items()
                         if op is not explicit and op.op.optype == 'Gather']
        assert len(sugar_gathers) == 1
        assert sugar_gathers[0].op.attrs.get('axis', 0) == explicit.op.attrs.get('axis', 0)

    @pytest.mark.unit
    def test_combined_with_slice(self):
        """x[:, idx, :2] lowers to Gather(axis=1) + Slice (axis 2 only;
        trivial slice(None) on axis 0 is suppressed when a tensor index is present)."""
        m = _DynTestModule(uid("m_tg7"))
        x, _ = _make_linked_tensor([4, 8, 16], module=m)
        idx = self._idx([3], m)
        n0 = len(m._op_hndls)
        y = x[:, idx, :2]
        assert y.shape == [4, 3, 2]
        new_ops = [op for nm, op in m._op_hndls.items()][n0:]
        optypes = [o.op.optype for o in new_ops]
        assert optypes == ['Gather', 'Slice'], f"got {optypes}"
        # Gather must be axis=1 (the tensor-index axis in the original spec).
        assert new_ops[0].op.attrs.get('axis', 0) == 1
        # Slice should target axis 2 only (the :2 axis), with no trivial axis-0 entry.
        slice_axes = new_ops[1].op.attrs.get('axes', None)
        # Slice carries axes via input tensor, not attrs -- inspect the const input.
        # The 4th input (idx 3) is the axes const.
        axes_in = new_ops[1].itensors[3]
        assert list(axes_in.data) == [2]

    @pytest.mark.unit
    def test_unknown_index_type_still_raises(self):
        """A non-(int|slice|None|Ellipsis|Tensor) index must still hit the
        legacy AssertionError fallback so future surprises surface loudly."""
        m = _DynTestModule(uid("m_tg8"))
        t, _ = _make_linked_tensor([4, 8], module=m)
        with pytest.raises(AssertionError, match="Non-slice object found where slice expected"):
            _ = t[:, {'not': 'an index'}]

    @pytest.mark.unit
    def test_no_active_module_raises(self):
        _clear_trace_ctx()
        tokens = make_front_tensor(name=uid("tok"), shape=[100, 32], dtype='float32')
        idx = make_front_tensor(name=uid("idx"), shape=[4], dtype='int64')
        with pytest.raises(RuntimeError, match="No active module context"):
            _ = tokens[idx]


# ===================================================================
# F2: multi-axis slice (slice_sinf bug fix) -- regression-pin tests
# ===================================================================

class TestMultiAxisSlice:

    @pytest.mark.unit
    def test_rank2_slice_axis1(self):
        """Regression-pin: x[:, 2:5] on [4, 8] -> [4, 3]. Before F2 the
        slice_sinf loop iterated `startsT.rank()` (= 1) and only processed
        axis 0, returning the input shape [4, 8] unchanged."""
        m = _DynTestModule(uid("m_ms1"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x[:, 2:5]
        assert y.shape == [4, 3]

    @pytest.mark.unit
    def test_rank3_slice_axis2(self):
        m = _DynTestModule(uid("m_ms2"))
        x, _ = _make_linked_tensor([2, 4, 8], module=m)
        y = x[:, :, 2:5]
        assert y.shape == [2, 4, 3]

    @pytest.mark.unit
    def test_rank2_slice_both_axes(self):
        m = _DynTestModule(uid("m_ms3"))
        x, _ = _make_linked_tensor([6, 8], module=m)
        y = x[1:4, 2:5]
        assert y.shape == [3, 3]

    @pytest.mark.unit
    def test_rank3_slice_middle_axis(self):
        m = _DynTestModule(uid("m_ms4"))
        x, _ = _make_linked_tensor([2, 6, 8], module=m)
        y = x[:, 1:4, :]
        assert y.shape == [2, 3, 8]

    @pytest.mark.unit
    def test_rank3_two_slices(self):
        m = _DynTestModule(uid("m_ms5"))
        x, _ = _make_linked_tensor([2, 6, 8], module=m)
        y = x[:, 1:4, 2:6]
        assert y.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_regression_pin_bug_case(self):
        """Direct regression-pin on `x[:, i:j]` on a rank-2 tensor. Asserts the
        correct output shape so any future regression of slice_sinf's multi-axis
        loop fails loudly."""
        m = _DynTestModule(uid("m_ms_pin"))
        x, _ = _make_linked_tensor([4, 8], module=m)
        y = x[:, 1:3]
        assert y.shape == [4, 2], (
            "F2 regression: slice_sinf is once again only processing axis 0; "
            "see src/bten/shape_inference.py::slice_sinf loop bound.")


# ===================================================================
# Symbolic-dim slice planning -- regression for the SymExpr `<` failure
# in torch2onnx_slice_plan triggered by DenseMoE's
# `full_weights[:, expert_id:expert_id+1]` under symbolic tracing.
# ===================================================================

class TestSymbolicSlice:
    """Planner-level pins: torch2onnx_slice_plan must not call `<` / `>`
    against a SymDim/SymExpr. Exercises the same code paths used by
    psim.py's symbolic tracing of the DenseMoE workload."""

    @pytest.mark.unit
    def test_symbolic_dim_slice_concrete_bounds(self):
        """`x[:, 0:1]` over a [N, SymDim('nE')] tensor must plan without
        raising. This is the exact shape DenseMoE.forward feeds into the
        per-expert weight slice. Pre-fix this raised:
            TypeError: '<' not supported between instances of 'SymExpr' and 'SymExpr'
        from `e = min(e, dim)` in dynamic.py."""
        from src.utils.sym import sym
        nE = sym("nE")
        plan = D.torch2onnx_slice_plan([4, nE], (slice(None), slice(0, 1)))
        # Axis 0 is trivial-full on a concrete dim; axis 1 is a concrete
        # int slice. Both must lower cleanly.
        assert plan['slice'] is not None
        # The trivial full-axis slice on a concrete dim is still emitted
        # (pre-existing behavior pinned by test_trailing_dims). Axis 1
        # bounds must be concrete ints.
        s = plan['slice']
        # Find the entry for axis 1 (post-int-gather axis).
        axis1_idx = s['axes'].index(1)
        assert s['starts'][axis1_idx] == 0
        assert s['ends'][axis1_idx] == 1
        assert plan['output_shape'][1] == 1

    @pytest.mark.unit
    def test_symbolic_axis_slice_concrete_bounds(self):
        """`x[0:1, :]` where THE SLICED AXIS is concrete but the trailing
        axis is symbolic. Pre-fix the planner blew up while iterating the
        symbolic-axis slice (clip-against-dim)."""
        from src.utils.sym import sym, is_symbolic
        nE = sym("nE")
        plan = D.torch2onnx_slice_plan([4, nE], (slice(0, 1),))
        # Axis 0 is a concrete [0:1] slice -> length 1.
        assert plan['output_shape'][0] == 1
        # Axis 1 inherits the symbolic dim; its length is a SymExpr that
        # resolves to `nE` under runtime substitution.
        assert is_symbolic(plan['output_shape'][1])
        assert plan['output_shape'][1].subs({"nE": 7}) == 7

    @pytest.mark.unit
    def test_symbolic_dim_slice_full_axis(self):
        """`x[:, :]` over a symbolic axis must plan without raising.
        Trivial full-axis slices on symbolic dims are suppressed at plan
        time (their `ends` would be a SymExpr that can't lower to an
        ONNX int constant)."""
        from src.utils.sym import sym, is_symbolic
        nE = sym("nE")
        plan = D.torch2onnx_slice_plan([4, nE], (slice(None), slice(None)))
        # The symbolic axis (1) must not appear in the slice spec --
        # suppressed to keep `ends` int-only.
        if plan['slice'] is not None:
            assert 1 not in plan['slice']['axes']
        # Output shape on the symbolic axis is a SymExpr that resolves
        # to the original symbolic dim under substitution.
        assert plan['output_shape'][0] == 4
        assert is_symbolic(plan['output_shape'][1])
        assert plan['output_shape'][1].subs({"nE": 11}) == 11

    @pytest.mark.unit
    def test_symbolic_dim_int_index(self):
        """`x[:, 0]` on a [N, SymDim] tensor must plan without raising.
        Defensive guard for the int-index branch's upper-bound check
        (`idx < working_shape[axis]` was the failure mode)."""
        from src.utils.sym import sym
        nE = sym("nE")
        plan = D.torch2onnx_slice_plan([4, nE], (slice(None), 0))
        assert plan['gathers'] is not None
        assert plan['gathers'][0] == (1, 0)
        # After the int-index gather, the symbolic axis is removed.
        assert plan['output_shape'] == [4]

    @pytest.mark.unit
    def test_symbolic_dim_int_index_negative_raises(self):
        """Negative int indices on a symbolic axis cannot be resolved at
        trace time (we'd need `dim + spec` to compare against `dim`).
        The `0 > idx` guard still catches genuine negatives that resolve
        below zero -- but with a symbolic dim we cannot know. Confirm
        the existing behavior: SymExpr `0 > idx` raises (no false ok)."""
        from src.utils.sym import sym
        nE = sym("nE")
        # Negative index resolves to `nE + (-1)` which is a SymExpr; the
        # `0 > idx` comparison against SymExpr raises TypeError. This
        # documents an intentional limitation -- negative int-indexing on
        # symbolic axes is not supported and surfaces loudly.
        with pytest.raises(TypeError):
            D.torch2onnx_slice_plan([4, nE], (slice(None), -1))


# ===================================================================
# Task 041: symbolic-dim support for trailing (negative-index) slices.
# A trailing window `x[..., -W:, ...]` on a SYMBOLIC axis must lower
# without folding the negative start into a SymExpr. ONNX Slice supports
# literal negative starts (clamped at runtime), so the planner keeps the
# literal `-W` start + an open-end sentinel, and slice_sinf emits the
# concrete window length W on that axis.
# ===================================================================

class TestSymbolicTrailingWindowSlice:

    SENTINEL = int(np.iinfo(np.int64).max)

    @pytest.mark.unit
    def test_symbolic_trailing_window_plan(self):
        """`x[:, :, -W:, :]` on a [B, H, SymDim('S'), D] tensor must keep the
        literal `-W` start and the open-end sentinel, and plan a concrete
        output length W on the symbolic axis."""
        from src.utils.sym import sym
        S = sym("S")
        W = 128
        plan = D.torch2onnx_slice_plan(
            [2, 4, S, 8], (slice(None), slice(None), slice(-W, None), slice(None)))
        assert plan['slice'] is not None
        s = plan['slice']
        axis2_idx = s['axes'].index(2)
        # Literal negative start kept (not folded to `S - W`).
        assert s['starts'][axis2_idx] == -W
        # Open-end sentinel that ONNX clamps at runtime.
        assert s['ends'][axis2_idx] == self.SENTINEL
        assert s['steps'][axis2_idx] == 1
        # Both initializer values are concrete int64-materializable.
        assert isinstance(s['starts'][axis2_idx], int)
        assert isinstance(s['ends'][axis2_idx], int)
        # Concrete planned window length on the symbolic axis.
        assert plan['output_shape'][2] == W

    @pytest.mark.unit
    def test_symbolic_trailing_window_ellipsis_plan(self):
        """The `[..., -W:, ...]` spelling (ellipsis form) lowers identically."""
        from src.utils.sym import sym
        S = sym("S")
        W = 64
        plan = D.torch2onnx_slice_plan(
            [2, 4, S, 8], (Ellipsis, slice(-W, None), slice(None)))
        s = plan['slice']
        axis2_idx = s['axes'].index(2)
        assert s['starts'][axis2_idx] == -W
        assert s['ends'][axis2_idx] == self.SENTINEL
        assert plan['output_shape'][2] == W

    @pytest.mark.unit
    def test_symbolic_trailing_window_end_to_end(self):
        """Full lowering through __getitem__ + slice_sinf: the sliced
        symbolic axis must resolve to a CONCRETE length W."""
        from src.utils.sym import sym
        S = sym("S")
        W = 128
        m = _DynTestModule(uid("m_symwin"))
        x, _ = _make_linked_tensor([2, 4, S, 8], module=m)
        y = x[:, :, -W:, :]
        assert y.shape[0] == 2
        assert y.shape[1] == 4
        # The previously-symbolic kv-length axis is now a concrete window.
        assert y.shape[2] == W
        assert y.shape[3] == 8

    @pytest.mark.unit
    def test_concrete_trailing_window_regression(self):
        """Same trailing slice on a CONCRETE axis behaves exactly as before:
        the negative start is folded against the concrete dim (no sentinel)."""
        plan = D.torch2onnx_slice_plan(
            [2, 4, 256, 8], (slice(None), slice(None), slice(-128, None), slice(None)))
        s = plan['slice']
        axis2_idx = s['axes'].index(2)
        # Folded concrete start = dim - W = 256 - 128 = 128; end = dim = 256.
        assert s['starts'][axis2_idx] == 128
        assert s['ends'][axis2_idx] == 256
        assert s['ends'][axis2_idx] != self.SENTINEL
        assert plan['output_shape'][2] == 128

    @pytest.mark.unit
    def test_concrete_trailing_window_regression_end_to_end(self):
        """Concrete-axis trailing slice lowers to the same shape it always
        has (guards against regressing test_trailing_dims-style behavior)."""
        m = _DynTestModule(uid("m_conc_win"))
        x, _ = _make_linked_tensor([2, 4, 256, 8], module=m)
        y = x[:, :, -128:, :]
        assert y.shape == [2, 4, 128, 8]

    @pytest.mark.unit
    def test_short_concrete_axis_clamps(self):
        """When the (concrete) axis is shorter than W, the window clamps to
        the axis length rather than over-reading (min(W, dim))."""
        m = _DynTestModule(uid("m_short_win"))
        x, _ = _make_linked_tensor([2, 4, 5, 8], module=m)
        y = x[:, :, -128:, :]
        # Only 5 positions available -> window clamps to 5, not 128.
        assert y.shape == [2, 4, 5, 8]
