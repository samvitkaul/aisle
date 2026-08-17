"""FrontTensor subclass architecture — class identity, method presence."""
import pytest

from src.bten.tensor import Tensor
from src.front.tensor import FrontTensor, make_front_tensor
import src.front.functional as F


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestFrontTensorSubclass:

    @pytest.mark.unit
    def test_front_tensor_is_tensor_subclass(self):
        ft = FrontTensor("test_sub", shape=[2, 3], dtype='float32')
        assert isinstance(ft, Tensor)
        assert isinstance(ft, FrontTensor)

    @pytest.mark.unit
    def test_plain_tensor_has_no_view(self):
        t = Tensor("test_plain", shape=[2, 3])
        assert not hasattr(t, 'view')

    @pytest.mark.unit
    def test_front_tensor_has_view(self):
        ft = FrontTensor("test_view", shape=[2, 3], dtype='float32')
        assert hasattr(ft, 'view')
        assert callable(ft.view)

    @pytest.mark.unit
    def test_front_tensor_has_operators(self):
        """Verify dunder methods are defined on the class, not patched."""
        assert '__add__' in FrontTensor.__dict__
        assert '__sub__' in FrontTensor.__dict__
        assert '__mul__' in FrontTensor.__dict__
        assert '__truediv__' in FrontTensor.__dict__
        assert '__pow__' in FrontTensor.__dict__
        assert '__matmul__' in FrontTensor.__dict__
        assert '__getitem__' in FrontTensor.__dict__
        assert 'view' in FrontTensor.__dict__
        assert 'reshape' in FrontTensor.__dict__
        assert 'transpose' in FrontTensor.__dict__

    @pytest.mark.unit
    def test_plain_tensor_no_operators(self):
        """Plain Tensor should NOT have patched methods."""
        assert '__add__' not in Tensor.__dict__
        assert 'view' not in Tensor.__dict__
        assert 'transpose' not in Tensor.__dict__

    @pytest.mark.unit
    def test_op_output_is_front_tensor(self):
        """Op outputs should be FrontTensor, enabling chaining."""
        op = F.Softmax(uid("test_softmax_ft"))
        assert isinstance(op.otensors[0], FrontTensor)
        assert isinstance(op.otensors[0], Tensor)

    @pytest.mark.unit
    def test_make_front_tensor_factory(self):
        ft = make_front_tensor(name="factory_test", shape=[4, 5], dtype='float32')
        assert isinstance(ft, FrontTensor)
        assert ft.shape == [4, 5]
