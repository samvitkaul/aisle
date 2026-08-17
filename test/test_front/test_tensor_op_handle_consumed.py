"""
Tests for TensorOpHandle one-shot invocation guard (NA-12).

Verifies that TensorOpHandle raises RuntimeError on second invocation,
preventing silent operation graph corruption.
"""
import numpy as np
import pytest

import src.front.functional as F
from src import make_tensor

_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestTensorOpHandleOneShot:
    """TensorOpHandle should reject second invocation on same instance."""

    @pytest.mark.unit
    def test_unary_single_invocation_succeeds(self):
        """Normal single invocation of unary op should work."""
        op = F.Softmax(uid("softmax"))
        x = make_tensor(name=uid("x"), shape=[3, 5], dtype='float32')
        y = op(x)
        assert y is not None
        assert y.shape == x.shape

    @pytest.mark.unit
    def test_unary_second_invocation_raises(self):
        """Second invocation on same unary op handle should raise RuntimeError."""
        op = F.Softmax(uid("softmax"))
        x1 = make_tensor(name=uid("x1"), shape=[3, 5], dtype='float32')
        x2 = make_tensor(name=uid("x2"), shape=[3, 5], dtype='float32')

        # First invocation succeeds
        y1 = op(x1)
        assert y1 is not None

        # Second invocation raises RuntimeError
        with pytest.raises(RuntimeError, match="can only be invoked once"):
            op(x2)

    @pytest.mark.unit
    def test_unary_error_message_includes_op_info(self):
        """Error message should include op name and type for debugging."""
        op = F.Relu(uid("relu_op"))
        x1 = make_tensor(name=uid("x1"), shape=[2, 3], dtype='float32')
        y1 = op(x1)

        x2 = make_tensor(name=uid("x2"), shape=[2, 3], dtype='float32')
        with pytest.raises(RuntimeError) as exc_info:
            op(x2)

        error_msg = str(exc_info.value)
        assert "relu_op" in error_msg
        assert "Relu" in error_msg
        assert "can only be invoked once" in error_msg

    @pytest.mark.unit
    def test_binary_single_invocation_succeeds(self):
        """Normal single invocation of binary op should work."""
        op = F.Add(uid("add"))
        x = make_tensor(name=uid("x"), shape=[3, 5], dtype='float32')
        y = make_tensor(name=uid("y"), shape=[3, 5], dtype='float32')
        z = op(x, y)
        assert z is not None
        assert z.shape == x.shape

    @pytest.mark.unit
    def test_binary_second_invocation_raises(self):
        """Second invocation on same binary op handle should raise RuntimeError."""
        op = F.MatMul(uid("matmul"))
        x1 = make_tensor(name=uid("x1"), shape=[4, 8], dtype='float32')
        y1 = make_tensor(name=uid("y1"), shape=[8, 3], dtype='float32')
        x2 = make_tensor(name=uid("x2"), shape=[4, 8], dtype='float32')
        y2 = make_tensor(name=uid("y2"), shape=[8, 3], dtype='float32')

        # First invocation succeeds
        z1 = op(x1, y1)
        assert z1 is not None

        # Second invocation raises RuntimeError
        with pytest.raises(RuntimeError, match="can only be invoked once"):
            op(x2, y2)

    @pytest.mark.unit
    def test_variadic_single_invocation_succeeds(self):
        """Normal single invocation of variadic op should work."""
        op = F.Concat(uid("concat"), axis=0)
        x1 = make_tensor(name=uid("x1"), shape=[2, 5], dtype='float32')
        x2 = make_tensor(name=uid("x2"), shape=[3, 5], dtype='float32')
        y = op(x1, x2)
        assert y is not None
        assert y.shape == [5, 5]

    @pytest.mark.unit
    def test_variadic_second_invocation_raises(self):
        """Second invocation on same variadic op handle should raise RuntimeError."""
        op = F.Concat(uid("concat"), axis=1)
        x1_1 = make_tensor(name=uid("x1_1"), shape=[4, 2], dtype='float32')
        x1_2 = make_tensor(name=uid("x1_2"), shape=[4, 3], dtype='float32')
        x2_1 = make_tensor(name=uid("x2_1"), shape=[4, 2], dtype='float32')
        x2_2 = make_tensor(name=uid("x2_2"), shape=[4, 3], dtype='float32')

        # First invocation succeeds
        y1 = op(x1_1, x1_2)
        assert y1 is not None

        # Second invocation raises RuntimeError
        with pytest.raises(RuntimeError, match="can only be invoked once"):
            op(x2_1, x2_2)

    @pytest.mark.unit
    def test_multi_output_op_single_invocation_succeeds(self):
        """Normal single invocation of multi-output op should work."""
        op = F.TopK(uid("topk"), num_outputs=2)
        x = make_tensor(name=uid("x"), shape=[10], dtype='float32')
        k_tensor = make_tensor(
            name=uid("k"),
            shape=[1],
            data=np.array([2], dtype=np.int64),
            dtype='int64',
            is_const=True,
        )
        values, indices = op(x, k_tensor)
        assert values is not None
        assert indices is not None

    @pytest.mark.unit
    def test_multi_output_op_second_invocation_raises(self):
        """Second invocation on same multi-output op handle should raise RuntimeError."""
        op = F.TopK(uid("topk"), num_outputs=2)
        x1 = make_tensor(name=uid("x1"), shape=[10], dtype='float32')
        x2 = make_tensor(name=uid("x2"), shape=[10], dtype='float32')
        k_tensor1 = make_tensor(
            name=uid("k1"),
            shape=[1],
            data=np.array([3], dtype=np.int64),
            dtype='int64',
            is_const=True,
        )
        k_tensor2 = make_tensor(
            name=uid("k2"),
            shape=[1],
            data=np.array([3], dtype=np.int64),
            dtype='int64',
            is_const=True,
        )

        # First invocation succeeds
        v1, i1 = op(x1, k_tensor1)
        assert v1 is not None
        assert i1 is not None

        # Second invocation raises RuntimeError
        with pytest.raises(RuntimeError, match="can only be invoked once"):
            op(x2, k_tensor2)

    @pytest.mark.unit
    def test_new_instance_each_iteration(self):
        """Creating new op instance for each iteration is the correct pattern."""
        # This test demonstrates the correct workaround:
        # Create a new op handle for each invocation
        for i in range(3):
            op = F.Gelu(uid(f"gelu_{i}"))
            x = make_tensor(name=uid(f"x_{i}"), shape=[2, 3], dtype='float32')
            y = op(x)
            assert y is not None
            assert y.shape == [2, 3]

    @pytest.mark.unit
    def test_transpose_with_params_single_invocation(self):
        """Ops with params should work for single invocation."""
        op = F.Transpose(uid("transpose"), perm=[1, 0])
        x = make_tensor(name=uid("x"), shape=[3, 5], dtype='float32')
        y = op(x)
        assert y is not None
        assert y.shape == [5, 3]

    @pytest.mark.unit
    def test_transpose_with_params_second_invocation_raises(self):
        """Second invocation on ops with params should also raise."""
        op = F.Transpose(uid("transpose"), perm=[1, 0])
        x1 = make_tensor(name=uid("x1"), shape=[3, 5], dtype='float32')
        x2 = make_tensor(name=uid("x2"), shape=[3, 5], dtype='float32')

        y1 = op(x1)
        assert y1 is not None

        with pytest.raises(RuntimeError, match="can only be invoked once"):
            op(x2)

    @pytest.mark.unit
    def test_error_before_state_modification(self):
        """Guard should raise before modifying internal state."""
        op = F.Sigmoid(uid("sigmoid"))
        x1 = make_tensor(name=uid("x1"), shape=[2], dtype='float32')
        y1 = op(x1)

        # Store original state
        original_in_list = op.op.inList.copy()

        x2 = make_tensor(name=uid("x2"), shape=[2], dtype='float32')
        with pytest.raises(RuntimeError):
            op(x2)

        # State should not have been modified by the failed second invocation
        assert op.op.inList == original_in_list
