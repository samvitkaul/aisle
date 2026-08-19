"""Tests for `src.utils.data_types.promote_types` (Task 048.1).

Covers all 16 rows of the Acceptance Criteria table in
`docs/tasks/048_bidir_bcast_dtype_promotion.md` under Sub-task 048.1.
"""
from itertools import product

import pytest

from src.utils.data_types import DataType as DT
from src.utils.data_types import promote_types


_p = promote_types


@pytest.mark.unit
def test_identity_fp32():
    assert _p(DT.FLOAT32, DT.FLOAT32) == DT.FLOAT32


@pytest.mark.unit
def test_poison_propagation_forward():
    assert _p(DT.FLOAT32, DT.UNDEF) == DT.UNDEF


@pytest.mark.unit
def test_poison_propagation_reverse():
    assert _p(DT.UNDEF, DT.FLOAT32) == DT.UNDEF


@pytest.mark.unit
def test_poison_both_sides():
    assert _p(DT.UNDEF, DT.UNDEF) == DT.UNDEF


@pytest.mark.unit
def test_bool_identity_forward():
    assert _p(DT.BOOL, DT.INT32) == DT.INT32


@pytest.mark.unit
def test_bool_identity_reverse():
    assert _p(DT.INT32, DT.BOOL) == DT.INT32


@pytest.mark.unit
def test_same_kind_wider_wins_int():
    assert _p(DT.INT16, DT.INT32) == DT.INT32


@pytest.mark.unit
def test_signed_plus_unsigned_widen_and_sign():
    assert _p(DT.INT8, DT.UINT8) == DT.INT16


@pytest.mark.unit
def test_int_plus_float_pytorch_tensor_tensor_rule():
    # PyTorch tensor-tensor rule: int width is discarded, float wins.
    assert _p(DT.INT32, DT.FLOAT16) == DT.FLOAT16


@pytest.mark.unit
def test_int64_plus_fp32_stays_fp32():
    # Pinned separately: most surprising vs NumPy (which returns FLOAT64).
    assert _p(DT.INT64, DT.FLOAT32) == DT.FLOAT32


@pytest.mark.unit
def test_bf16_plus_fp16_promotes_to_fp32():
    assert _p(DT.BFLOAT16, DT.FLOAT16) == DT.FLOAT32


@pytest.mark.unit
def test_bf16_plus_fp32_promotes_to_fp32():
    assert _p(DT.BFLOAT16, DT.FLOAT32) == DT.FLOAT32


@pytest.mark.unit
def test_fp16_plus_fp32_promotes_to_fp32():
    assert _p(DT.FLOAT16, DT.FLOAT32) == DT.FLOAT32


@pytest.mark.unit
def test_tf32_plus_fp32_collapses_to_fp32():
    # TF32 is a compute-mode variant of fp32 storage.
    assert _p(DT.TENSOR_FLOAT32, DT.FLOAT32) == DT.FLOAT32


@pytest.mark.unit
def test_distinct_fp8_variants_promote_to_fp16():
    assert _p(DT.FLOAT8, DT.FLOAT8_e5m2) == DT.FLOAT16


@pytest.mark.unit
def test_named_fp8_identity():
    assert _p(DT.FLOAT8_e4m3, DT.FLOAT8_e4m3) == DT.FLOAT8_e4m3


@pytest.mark.unit
def test_symmetry_full_sweep():
    """For every (a, b) in DataType x DataType, promote_types is symmetric."""
    for a, b in product(DT, DT):
        assert _p(a, b) == _p(b, a), f"asymmetric: ({a}, {b})"


@pytest.mark.unit
def test_no_hidden_poisoning():
    """promote_types never fabricates UNDEF when neither input is UNDEF."""
    concrete = [d for d in DT if d is not DT.UNDEF]
    for a, b in product(concrete, concrete):
        assert _p(a, b) is not DT.UNDEF, f"unexpected UNDEF for ({a}, {b})"


# --- Additional pins ---------------------------------------------------------

@pytest.mark.unit
def test_reverse_signed_unsigned_symmetric():
    # Rule 6 in reverse to guard the signed/unsigned branch symmetry.
    assert _p(DT.UINT8, DT.INT8) == DT.INT16


@pytest.mark.unit
def test_int64_uint64_promotes_to_fp64():
    # NumPy rule -- pinned per §Risks / §Scope.
    assert _p(DT.INT64, DT.UINT64) == DT.FLOAT64


@pytest.mark.unit
def test_rejects_non_datatype_left():
    with pytest.raises(TypeError):
        _p('float32', DT.FLOAT32)  # type: ignore[arg-type]


@pytest.mark.unit
def test_rejects_non_datatype_right():
    with pytest.raises(TypeError):
        _p(DT.FLOAT32, 'float32')  # type: ignore[arg-type]
