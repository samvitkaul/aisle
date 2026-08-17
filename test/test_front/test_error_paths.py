"""Edge cases and negative-path tests for the front-end."""
import numpy as np
import pytest

import src.front.functional as F
import src.front.module as nn
from src import make_tensor

_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestEdgeCases:

    @pytest.mark.unit
    def test_module_call_not_implemented(self):
        """Base Module.__call__ should raise."""
        m = nn.Module(uid("base"))
        with pytest.raises(NotImplementedError):
            m()

    @pytest.mark.unit
    def test_tensor_op_handle_wrong_input_count(self):
        """Passing wrong number of inputs to a unary op should raise."""
        op = F.Softmax(uid("sm_neg"))
        a = make_tensor(name=uid("a"), shape=[3], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[3], dtype='float32')
        with pytest.raises(ValueError):
            op(a, b)

    @pytest.mark.unit
    def test_binary_op_wrong_input_count(self):
        """Passing 1 input to a binary op should raise."""
        op = F.Add(uid("add_neg"))
        a = make_tensor(name=uid("a"), shape=[3], dtype='float32')
        with pytest.raises(ValueError):
            op(a)

    @pytest.mark.unit
    def test_matmul_shape_mismatch(self):
        op = F.MatMul(uid("mm_neg2"))
        a = make_tensor(name=uid("a"), shape=[3, 4], dtype='float32')
        b = make_tensor(name=uid("b"), shape=[5, 6], dtype='float32')
        with pytest.raises((ValueError, AssertionError)):
            op(a, b)

    @pytest.mark.unit
    def test_squeeze_non1_dim_raises(self):
        """Squeezing a dim that is not 1 should raise."""
        op = F.Squeeze(uid("sq_neg"))
        x = make_tensor(name=uid("x"), shape=[2, 3, 4], dtype='float32')
        axes_data = np.array([1], dtype=np.int64)
        axes = make_tensor(name=uid("axes"), shape=[1], data=axes_data, dtype='int64', is_const=True)
        with pytest.raises(ValueError):
            op(x, axes)


class TestFrontErrorPaths:

    @pytest.mark.unit
    def test_modulelist_non_module_raises(self):
        with pytest.raises(TypeError):
            nn.ModuleList(["not a module"])

    @pytest.mark.unit
    def test_op_handle_bad_ipos_raises(self):
        with pytest.raises(ValueError, match="ipos"):
            F.UniversalOperator(uid("bad"), optype="Add", params=[], ipos=[])

    @pytest.mark.unit
    def test_op_handle_arity_mismatch_raises(self):
        op = F.Add(uid("add_err"))
        a = make_tensor(name=uid("a"), shape=[2, 3], dtype='float32')
        with pytest.raises(ValueError, match="don't match"):
            op(a)
