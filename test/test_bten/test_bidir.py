
import pytest

from src.bten.tensor import make_tensor
from src.bten.op import make_op
from src.utils.data_types import DataType, promote_types

import numpy as np

_OPTYPES = ['Add', 'Sub', 'Mul', 'Div', 'Pow', ]
_POS_TESTCASES = [
        ("Scalar to 2D Broadcasting",          [],              [3, 4]          ),
        ("1D to 2D Broadcasting",              [4],             [3, 4]          ),
        ("Bidirectional Broadcasting",         [3, 1],          [1, 4]          ),
        ("Multidimensional Broadcasting",      [2, 1, 4],       [1, 3, 1]       ),
        ("No Broadcasting",                    [2, 3, 4],       [2, 3, 4]       ),
        ("Empty Dimension Broadcasting",       [1, 0],          [1]             ),
        ("High Dimension Broadcasting",        [1, 1, 1, 1, 4], [2, 3, 1, 1, 4] ),
        ("Complex Nested Broadcasting",        [1, 2, 1, 4],    [3, 1, 5, 1]    ),
        ("Zero Sized Broadcasting",            [0, 1],          [1, 0]          ),
        ("Scalar High Dimension Broadcasting", [],              [2, 3, 4, 5]    ),
        ]
_NEG_TESTCASES = [
        ("Incompatible shapes",               [3, 4], [5, 6] ),
        ("Mismatched non-broadcastable dims", [3, 4], [3, 5] ),
        ]
_DTYPE_PAIRS = [
        ('float32',  'float32',  DataType.FLOAT32), #identity
        ('bfloat16', 'float32',  DataType.FLOAT32), #motivating: GQA scale
        ('int64',    'float32',  DataType.FLOAT32), #torch tensor-tensor rule
        ('float16',  'bfloat16', DataType.FLOAT32), #two 16b floats -> fp32
        ('float32',  'undef',    DataType.UNDEF),   #poison propagation
        ]

def prepare_op(_optype, _tname, _tdim0, _tdim1, _dtype1='float32', _dtype2='float32'):
    op_name = f"{_optype}_{_tname}"
    i_tensors = [
        make_tensor(name="X0", shape=_tdim0, dtype=_dtype1),
        make_tensor(name="X1", shape=_tdim1, dtype=_dtype2),
        ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": _optype,
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("optype", _OPTYPES)
@pytest.mark.parametrize("tname, tdim0, tdim1", _POS_TESTCASES)
def test_bidir_bcast(optype, tname, tdim0, tdim1):
    OP, IN, OUT = prepare_op(optype, tname, tdim0, tdim1)

    OP(IN, OUT)
    inf_shape = OUT[0].shape

    _X0 = np.random.randn(*tdim0)
    _X1 = np.random.randn(*tdim1)
    _Y0 = np.add(_X0, _X1)
    ref_shape = list(_Y0.shape)

    assert inf_shape == ref_shape

@pytest.mark.unit
@pytest.mark.parametrize("optype", _OPTYPES)
@pytest.mark.parametrize("tname, tdim0, tdim1", _NEG_TESTCASES)
def test_bidir_bcast_neg(optype, tname, tdim0, tdim1):
    OP, IN, OUT = prepare_op(optype, tname, tdim0, tdim1)
    with pytest.raises((ValueError, AssertionError)):
        OP(IN, OUT)

@pytest.mark.unit
@pytest.mark.parametrize("optype", _OPTYPES)
@pytest.mark.parametrize("dtype_a, dtype_b, expected", _DTYPE_PAIRS)
def test_dtype_promotion(optype, dtype_a, dtype_b, expected):
    tname = f"{optype}_{dtype_a}_{dtype_b}"
    tdim0 = [2,3]
    tdim1 = [2,3]
    OP, IN, OUT = prepare_op(optype, tname, tdim0, tdim1, dtype_a, dtype_b)

    OP(IN, OUT)
    Y = OUT[0]
    assert Y.shape == [2, 3]
    assert Y.dtype == expected
    assert Y.dtype == promote_types(IN[0].dtype, IN[1].dtype)
