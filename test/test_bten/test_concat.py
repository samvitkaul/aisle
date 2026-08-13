
import pytest

from src.bten.tensor import make_tensor
from src.bten.op import make_op
from src.utils.data_types import DataType, promote_types

import numpy as np
from functools import reduce as _reduce

_POS_TESTCASES = [
        ("2D tensors along axis 0",   [[2, 3], [4, 3]],             0, [6, 3]       ),
        ("2D tensors along axia 1",   [[2, 3], [2, 4]],             1, [2, 7]       ),
        ("Multiple inputs axis 0",    [[2, 3], [2, 3], [2, 3]],     0, [6, 3]       ),
        ("3D tensors along axis 2",   [[2, 3, 4], [2, 3, 5]],       2, [2, 3, 9]    ),
        ("4D tensors along axis 1",   [[1, 2, 3, 4], [1, 2, 3, 4]], 1, [1, 4, 3, 4] ),
        ("Single element tensors",    [[1, 1], [1, 1]],             0, [2, 1]       ),
        ("Zero sized dims",           [[0, 3], [2, 3]],             0, [2, 3]       ),
        ("Empty dim on concat axis",  [[2, 0], [2, 3]],             1, [2, 3]       ),
        ("Neg axis -1",               [[2, 3], [2, 4]],            -1, [2, 7]       ),
        ("Neg axis -2",               [[2, 3], [4, 3]],            -2, [6, 3]       ),
        ]

_NEG_TESTCASES = [
        ("Mismatched ranks",            [[2, 3], [2, 3, 4]],  0),
        ("Mismatched non-concat dims",  [[2, 3], [2, 4]],     0),
        ("Invalid axis",                [[2, 3], [2, 3]],     5),
        ("Neg out of bound axis",       [[2, 3], [2, 3]],    -5),
        ]

_DTYPE_CASES = [
        ('2in_identity_bf16',        ['bfloat16', 'bfloat16'],                         DataType.BFLOAT16),
        ('2in_mixed_bf16_fp32',      ['bfloat16', 'float32'],                          DataType.FLOAT32),
        ('3in_mixed_bf16_fp32_fp16', ['bfloat16', 'float32', 'float16'],               DataType.FLOAT32),
        ('3in_identity_bf16',        ['bfloat16', 'bfloat16', 'bfloat16'],             DataType.BFLOAT16),
        ('4in_fp32_at_end',          ['bfloat16', 'bfloat16', 'bfloat16', 'float32'],  DataType.FLOAT32),
        ('poison_bf16_ud',           ['bfloat16', 'undef'],                            DataType.UNDEF),
        ('poison_ud_bf16_fp32',      ['undef',    'bfloat16', 'float32'],              DataType.UNDEF),
        ]


def prepare_op(_tname, _ishapes, _axis, _dtypes=None):
    if _dtypes is not None:
        assert len(_ishapes) == len(_dtypes)
    else:
        _dtypes = ['float32' for _ in _ishapes]

    op_name = f"{_tname}"
    i_tensors = [
            make_tensor(name=f"X{i}", shape=shape, dtype=dtype)
            for i, (shape, dtype) in enumerate(zip(_ishapes, _dtypes))
            ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": "Concat",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": _axis},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("tname, ishapes, axis, expected", _POS_TESTCASES)
def test_concat(tname, ishapes, axis, expected):
    OP, IN, OUT = prepare_op(tname, ishapes, axis)

    OP(IN, OUT)
    inf_shape = OUT[0].shape
    assert inf_shape == expected

    #ref impl
    arrays = [np.random.randn(*shape) for shape in shapes]
    result = np.concatenate(arrays, axis=axis)
    ref_shape = list(result.shape)
    assert inf_shape == ref_shape


@pytest.mark.unit
@pytest.mark.parametrize("tname, ishapes, axis", _NEG_TESTCASES)
def test_concat_neg(tname, ishapes, axis):
    OP, IN, OUT = prepare_op(tname, ishapes, axis)
    with pytest.raises((ValueError, AssertionError)):
        OP(IN, OUT)

@pytest.mark.unit
@pytest.mark.parametrize("tname, dtypes, expected", _DTYPE_CASES)
def test_dtype_promotion(tname, dtypes, expected):
    ishapes = [[2, 3]for _ in dtypes]
    axis    = 0
    OP, IN, OUT = prepare_op(tname, ishapes, axis, dtypes)

    OP(IN, OUT)
    Y = OUT[0]
    assert Y.shape == [2 * len(dtypes), 3]
    assert Y.dtype == expected
    assert Y.dtype == _reduce(
            lambda acc, t: promote_types(acc, t.dtype),
            IN[1:], IN[0].dtype)
