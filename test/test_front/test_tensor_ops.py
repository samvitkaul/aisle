"""Dynamic layer — FrontTensor methods (view/reshape/transpose), operator
overloads (+,-,*,/,**,@), reflected ops, __getitem__, and D.cat."""
import pytest
import math

from src import make_tensor
from src.front.tensor import make_front_tensor
import src.front.module as nn
import src.front.dynamic as D


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


# ---------------------------------------------------------------------------
# Helpers: every dynamic op needs an active module context.
# ---------------------------------------------------------------------------

class _DynTestModule(nn.Module):
    """Small module that hosts dynamic ops."""
    def __init__(self, name):
        super().__init__(name)


def _make_linked_tensor(shape, dtype='float32', module=None):
    """Create a FrontTensor linked to a module so dynamic ops can work."""
    if module is None:
        module = _DynTestModule(uid("dmod"))
    t = make_front_tensor(name=uid("dt"), shape=shape, dtype=dtype)
    module._tensors[t.name] = t
    # Set thread-local module context so dynamic ops can find the active module
    from src.front.module import _trace_ctx
    _trace_ctx.current_module = module
    return t, module


# ===================================================================
# tensor.view / tensor.reshape
# ===================================================================

class TestTensorView:

    @pytest.mark.unit
    def test_exact_shape(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.view(4, 3, 2)
        assert y.shape == [4, 3, 2]

    @pytest.mark.unit
    def test_flatten(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.view(24)
        assert y.shape == [24]

    @pytest.mark.unit
    def test_infer_minus1(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.view(2, -1)
        assert y.shape == [2, 12]

    @pytest.mark.unit
    def test_infer_middle(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.view(2, -1, 2)
        assert y.shape == [2, 6, 2]

    @pytest.mark.unit
    def test_reshape_alias(self):
        """tensor.reshape is the same as tensor.view."""
        t, _ = _make_linked_tensor([6, 4])
        y = t.reshape(3, 8)
        assert y.shape == [3, 8]

    @pytest.mark.unit
    def test_view_with_tuple_shape(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.view((4, 6))
        assert y.shape == [4, 6]

    @pytest.mark.unit
    def test_incompatible_raises(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        with pytest.raises(ValueError):
            t.view(5, 5)

    @pytest.mark.unit
    def test_two_minus1_raises(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        with pytest.raises(ValueError):
            t.view(-1, -1)


# ===================================================================
# tensor.transpose
# ===================================================================

class TestTensorTranspose:

    @pytest.mark.unit
    def test_swap_last_two(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.transpose(1, 2)
        assert y.shape == [2, 4, 3]

    @pytest.mark.unit
    def test_negative_dims(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.transpose(-2, -1)
        assert y.shape == [2, 4, 3]

    @pytest.mark.unit
    def test_identity(self):
        t, _ = _make_linked_tensor([2, 3, 4])
        y = t.transpose(0, 0)
        assert y.shape == [2, 3, 4]

    @pytest.mark.unit
    def test_4d_attention(self):
        """[B, S, nH, dH] → [B, nH, S, dH] via transpose(1, 2)."""
        t, _ = _make_linked_tensor([1, 9, 3, 16])
        y = t.transpose(1, 2)
        assert y.shape == [1, 3, 9, 16]

    @pytest.mark.unit
    def test_oob_raises(self):
        t, _ = _make_linked_tensor([2, 3])
        with pytest.raises(ValueError):
            t.transpose(0, 5)

    @pytest.mark.unit
    def test_rank_check(self):
        """Transpose on rank-0 tensor should raise."""
        t, _ = _make_linked_tensor([])
        with pytest.raises(ValueError, match="rank must be at least 1"):
            t.transpose(0, 1)


# ===================================================================
# Operator overloads: a OP b where a, b are FrontTensor
# ===================================================================

class TestTensorBinaryOps:

    @pytest.mark.unit
    def test_add(self):
        m = _DynTestModule(uid("m_add"))
        a, _ = _make_linked_tensor([2, 3], module=m)
        b, _ = _make_linked_tensor([2, 3], module=m)
        y = a + b
        assert y.shape == [2, 3]

    @pytest.mark.unit
    def test_sub(self):
        m = _DynTestModule(uid("m_sub"))
        a, _ = _make_linked_tensor([4, 5], module=m)
        b, _ = _make_linked_tensor([4, 5], module=m)
        y = a - b
        assert y.shape == [4, 5]

    @pytest.mark.unit
    def test_mul(self):
        m = _DynTestModule(uid("m_mul"))
        a, _ = _make_linked_tensor([3], module=m)
        b, _ = _make_linked_tensor([3], module=m)
        y = a * b
        assert y.shape == [3]

    @pytest.mark.unit
    def test_div(self):
        m = _DynTestModule(uid("m_div"))
        a, _ = _make_linked_tensor([2, 3], module=m)
        b, _ = _make_linked_tensor([2, 3], module=m)
        y = a / b
        assert y.shape == [2, 3]

    @pytest.mark.unit
    def test_pow(self):
        m = _DynTestModule(uid("m_pow"))
        a, _ = _make_linked_tensor([2, 3], module=m)
        b, _ = _make_linked_tensor([2, 3], module=m)
        y = a ** b
        assert y.shape == [2, 3]

    @pytest.mark.unit
    def test_matmul_operator(self):
        m = _DynTestModule(uid("m_mm"))
        a, _ = _make_linked_tensor([2, 3, 4], module=m)
        b, _ = _make_linked_tensor([2, 4, 5], module=m)
        y = a @ b
        assert y.shape == [2, 3, 5]

    @pytest.mark.unit
    def test_add_broadcast(self):
        """x + bias where bias is [D] and x is [B, S, D]."""
        m = _DynTestModule(uid("m_bcast"))
        x, _ = _make_linked_tensor([2, 5, 64], module=m)
        bias, _ = _make_linked_tensor([64], module=m)
        y = x + bias
        assert y.shape == [2, 5, 64]

    @pytest.mark.unit
    def test_mul_scalar_broadcast(self):
        """Scale attention scores: QK * scalar."""
        m = _DynTestModule(uid("m_scale"))
        qk, _ = _make_linked_tensor([1, 3, 9, 9], module=m)
        scale = make_tensor(name=uid("scale"), shape=[], dtype='float32',
                            data=1.0 / math.sqrt(16))
        m._tensors[scale.name] = scale
        y = qk * scale
        assert y.shape == [1, 3, 9, 9]


# ===================================================================
# Reflected ops: lhs OP rhs where lhs is plain Tensor, rhs is FrontTensor
# ===================================================================

class TestTensorReflectedOps:

    @pytest.mark.unit
    def test_radd(self):
        m = _DynTestModule(uid("m_radd"))
        lhs = make_tensor(name=uid("lhs"), shape=[2, 3], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([2, 3], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs + rhs  # triggers rhs.__radd__(lhs)
        assert y.shape == [2, 3]

    @pytest.mark.unit
    def test_rsub(self):
        m = _DynTestModule(uid("m_rsub"))
        lhs = make_tensor(name=uid("lhs"), shape=[4], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([4], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs - rhs
        assert y.shape == [4]

    @pytest.mark.unit
    def test_rmul(self):
        m = _DynTestModule(uid("m_rmul"))
        lhs = make_tensor(name=uid("lhs"), shape=[3, 5], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([3, 5], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs * rhs
        assert y.shape == [3, 5]

    @pytest.mark.unit
    def test_rtruediv(self):
        m = _DynTestModule(uid("m_rdiv"))
        lhs = make_tensor(name=uid("lhs"), shape=[2], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([2], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs / rhs
        assert y.shape == [2]

    @pytest.mark.unit
    def test_rpow(self):
        m = _DynTestModule(uid("m_rpow"))
        lhs = make_tensor(name=uid("lhs"), shape=[3], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([3], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs ** rhs
        assert y.shape == [3]

    @pytest.mark.unit
    def test_rmatmul(self):
        m = _DynTestModule(uid("m_rmm"))
        lhs = make_tensor(name=uid("lhs"), shape=[3, 4], dtype='float32')
        m._tensors[lhs.name] = lhs
        rhs, _ = _make_linked_tensor([4, 5], module=m)
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        y = lhs @ rhs
        assert y.shape == [3, 5]


# ===================================================================
# tensor.__getitem__ — slicing, indexing, newaxis, ellipsis
# ===================================================================

class TestTensorGetItem:

    @pytest.mark.unit
    def test_simple_slice(self):
        m = _DynTestModule(uid("m_gi1"))
        t, _ = _make_linked_tensor([10, 20], module=m)
        y = t[1:5]
        assert y.shape is not None
        assert y.shape[0] == 4

    @pytest.mark.unit
    def test_integer_index(self):
        m = _DynTestModule(uid("m_gi2"))
        t, _ = _make_linked_tensor([10, 20], module=m)
        y = t[0]
        assert y.shape == [20]

    @pytest.mark.unit
    def test_newaxis(self):
        m = _DynTestModule(uid("m_gi3"))
        t, _ = _make_linked_tensor([10, 20], module=m)
        y = t[None, :]
        assert 1 in y.shape

    @pytest.mark.unit
    def test_combined_int_and_slice(self):
        m = _DynTestModule(uid("m_gi4"))
        t, _ = _make_linked_tensor([10, 20, 30], module=m)
        y = t[0, 5:15]
        assert y.shape is not None

    @pytest.mark.unit
    def test_ellipsis(self):
        m = _DynTestModule(uid("m_gi5"))
        t, _ = _make_linked_tensor([10, 20, 30], module=m)
        y = t[..., 0:5]
        assert y.shape is not None


# ===================================================================
# D.cat — variadic concat helper
# ===================================================================

class TestCatFunction:

    @pytest.mark.unit
    def test_basic_cat(self):
        m = _DynTestModule(uid("m_cat"))
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        a, _ = _make_linked_tensor([2, 4], module=m)
        b, _ = _make_linked_tensor([3, 4], module=m)
        y = D.cat([a, b], dim=0)
        assert y.shape == [5, 4]

    @pytest.mark.unit
    def test_cat_non_tensor_raises(self):
        m = _DynTestModule(uid("m_cat2"))
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        with pytest.raises(TypeError, match="not a Tensor"):
            D.cat(["not_a_tensor"], dim=0)

    @pytest.mark.unit
    def test_cat_axis1(self):
        m = _DynTestModule(uid("m_cat3"))
        from src.front.module import _trace_ctx
        _trace_ctx.current_module = m
        a, _ = _make_linked_tensor([2, 3], module=m)
        b, _ = _make_linked_tensor([2, 5], module=m)
        y = D.cat([a, b], dim=1)
        assert y.shape == [2, 8]
