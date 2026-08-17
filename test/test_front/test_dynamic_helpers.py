"""Dynamic-layer helpers — DynName uniqueness and torch2onnx_slice_plan."""
import pytest

import src.front.module as nn
import src.front.dynamic as D


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestDynName:

    @pytest.mark.unit
    def test_get_uniqueness(self):
        m = nn.Module(uid("m_dyn"))
        name1 = D.DynName.get(m, 'op')
        name2 = D.DynName.get(m, 'op')
        assert name1 != name2


class TestTorch2OnnxSlicePlan:

    @pytest.mark.unit
    def test_simple_slice(self):
        plan = D.torch2onnx_slice_plan([10, 20], (slice(1, 5),))
        assert plan['slice'] is not None
        assert plan['gathers'] is None

    @pytest.mark.unit
    def test_integer_index(self):
        plan = D.torch2onnx_slice_plan([10, 20], (3,))
        assert plan['gathers'] is not None
        assert len(plan['gathers']) == 1

    @pytest.mark.unit
    def test_negative_index(self):
        plan = D.torch2onnx_slice_plan([10, 20], (-1,))
        assert plan['gathers'] is not None
        # Negative index should be resolved to positive
        assert plan['gathers'][0][1] == 9

    @pytest.mark.unit
    def test_newaxis(self):
        plan = D.torch2onnx_slice_plan([10, 20], (None, slice(None)))
        assert plan['unsqueezes'] is not None

    @pytest.mark.unit
    def test_ellipsis(self):
        plan = D.torch2onnx_slice_plan([10, 20, 30], (Ellipsis, slice(0, 5)))
        assert plan['slice'] is not None

    @pytest.mark.unit
    def test_multiple_ellipsis_raises(self):
        with pytest.raises(AssertionError, match="Multiple ellipsis"):
            D.torch2onnx_slice_plan([10, 20], (Ellipsis, Ellipsis))

    @pytest.mark.unit
    def test_negative_step_raises(self):
        with pytest.raises(NotImplementedError, match="Negative steps"):
            D.torch2onnx_slice_plan([10], (slice(None, None, -1),))

    @pytest.mark.unit
    def test_too_many_indices_raises(self):
        with pytest.raises(AssertionError, match="Too many"):
            D.torch2onnx_slice_plan([10], (1, 2))

    @pytest.mark.unit
    def test_negative_start_stop(self):
        plan = D.torch2onnx_slice_plan([10], (slice(-3, -1),))
        assert plan['slice'] is not None
        assert plan['slice']['starts'][0] == 7
        assert plan['slice']['ends'][0] == 9

    @pytest.mark.unit
    def test_step_greater_than_1(self):
        plan = D.torch2onnx_slice_plan([10], (slice(0, 10, 2),))
        assert plan['slice'] is not None
        assert plan['slice']['steps'][0] == 2

    @pytest.mark.unit
    def test_full_slice(self):
        plan = D.torch2onnx_slice_plan([10, 20], (slice(None),))
        assert plan['slice'] is not None
        assert plan['slice']['starts'][0] == 0
        assert plan['slice']['ends'][0] == 10

    @pytest.mark.unit
    def test_out_of_bounds_raises(self):
        with pytest.raises(AssertionError, match="out of bounds"):
            D.torch2onnx_slice_plan([10], (15,))

    @pytest.mark.unit
    def test_trailing_dims(self):
        """Trailing unspecified dims should be filled with slice(None)."""
        plan = D.torch2onnx_slice_plan([10, 20, 30], (slice(0, 5),))
        assert plan['slice'] is not None
        # Should have slices for all 3 dims
        assert len(plan['slice']['axes']) == 3
