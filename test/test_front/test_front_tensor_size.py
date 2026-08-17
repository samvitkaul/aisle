"""FrontTensor.size() accessor

`size()` mirrors `torch.Tensor.size`: full shape with no arg, per-axis length
with an int arg (including negative dims), and symbolic axes pass through as
their SymDim/SymExpr value (no int coercion). This is tensor-dimension length,
distinct from `DTensor.size(mesh_dim=...)` device-mesh semantics.
"""
import pytest

from src.front.tensor import FrontTensor
from src.utils.sym import sym, is_symbolic, SymDim, SymExpr


class TestFrontTensorSize:

    @pytest.mark.unit
    def test_size_no_arg_returns_full_shape(self):
        ft = FrontTensor("t_full", shape=[2, 3, 4], dtype='float32')
        assert ft.size() == ft.shape
        assert ft.size() == [2, 3, 4]

    @pytest.mark.unit
    def test_size_per_axis(self):
        ft = FrontTensor("t_axis", shape=[2, 3, 4], dtype='float32')
        assert ft.size(0) == 2
        assert ft.size(1) == 3
        assert ft.size(2) == 4

    @pytest.mark.unit
    def test_size_negative_dim(self):
        ft = FrontTensor("t_neg", shape=[2, 3, 4], dtype='float32')
        assert ft.size(-1) == 4
        assert ft.size(-2) == 3
        assert ft.size(-3) == 2

    @pytest.mark.unit
    def test_size_out_of_range_raises(self):
        ft = FrontTensor("t_oob", shape=[2, 3], dtype='float32')
        with pytest.raises(IndexError):
            ft.size(2)
        with pytest.raises(IndexError):
            ft.size(-3)

    @pytest.mark.unit
    def test_size_symbolic_passthrough(self):
        S = sym("S")
        ft = FrontTensor("t_sym", shape=[1, 4, S, 8], dtype='float32')
        d2 = ft.size(2)
        # Symbolic axis returns the symbolic value, not coerced to int.
        assert is_symbolic(d2)
        assert isinstance(d2, (SymDim, SymExpr))
        assert d2 is ft.shape[2]
        assert not isinstance(d2, int)

    @pytest.mark.unit
    def test_size_symbolic_negative_dim(self):
        S = sym("S")
        ft = FrontTensor("t_sym_neg", shape=[1, 2, S, 8], dtype='float32')
        # size(-2) reaches the symbolic axis via negative-dim normalization.
        assert ft.size(-2) is ft.shape[2]
        assert is_symbolic(ft.size(-2))
