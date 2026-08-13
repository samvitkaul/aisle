
import pytest

from src.bten.tensor import make_tensor
from src.bten.op import make_op
from src.utils.data_types import DataType, promote_types

import numpy as np

_POS_TESTCASES = [
        ("2D Matrix-Matrix",       [2, 3],            [3, 4]       ),
        ("Vector-Matrix",          [4],               [4, 3]       ),
        ("Matrix-Vector",          [3, 4],            [4]          ),
        ("Vector-Vector",          [4],               [4]          ),
        ("Batched Matmul",         [2, 3, 4],         [2, 4, 5]    ),
        ("Single elem Matrices",   [1, 1],            [1, 1]       ),
        ("Empty dim",              [3, 0],            [0, 4]       ),
        ("Batched Vector Ops",     [2, 4],            [2, 4, 5]    ),
        ("Higer Dim Batched",      [2, 3, 4, 5],      [2, 3, 5, 6] ),
        ]

_NEG_TESTCASES = [
        ("Mismatched dims",       [3, 4],    [3, 5]    ),
        ("Mismatched batch dims", [2, 3, 4], [3, 4, 5] ),
        ("Mismatched inner dim",  [3, 4],    [5, 6]    ),
        ]

_MATMUL_DTYPE_PAIRS = [
        ('float32',  'float32',   DataType.FLOAT32),  #identity
        ('bfloat16', 'bfloat16',  DataType.BFLOAT16), #identity - bf16 fast path
        ('bfloat16', 'float32',   DataType.FLOAT32),  #motivating - mixed wt/act
        ('int64',    'float32',   DataType.FLOAT32),  #torch tensor-tensor rule
        ('float32',  'undef',     DataType.UNDEF),    #poison propagation
        ]

_MATMUL_SHAPES_CASES = [
        ("2Dx2D",   [3, 4],    [4, 5],    [3, 5]   ),
        ("1Dx1D",   [4],       [4],       []       ),
        ("batched", [2, 3, 4], [2, 4, 5], [2, 3, 5]),
        ]

def prepare_op(_tname, tdim0, tdim1, dtype0='float32', dtype1='float32'):
    op_name = f"{_tname}"
    i_tensors = [
            make_tensor(name="X0", shape=tdim0, dtype=dtype0),
            make_tensor(name="X1", shape=tdim1, dtype=dtype1),
            ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": "MatMul",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("tname, tdim0, tdim1", _POS_TESTCASES)
def test_matmul(tname, tdim0, tdim1):
    OP, IN, OUT = prepare_op(tname, tdim0, tdim1)

    OP(IN, OUT)
    inf_shape = OUT[0].shape

    #ref impl
    _X0 = np.random.randn(*tdim0)
    _X1 = np.random.randn(*tdim1)
    _Y  = np.matmul(_X0, _X1)
    ref_shape = list(_Y.shape)

    assert inf_shape == ref_shape


@pytest.mark.unit
@pytest.mark.parametrize("tname, tdim0, tdim1", _NEG_TESTCASES)
def test_matmul_neg(tname, tdim0, tdim1):
    OP, IN, OUT = prepare_op(tname, tdim0, tdim1)
    with pytest.raises((ValueError, AssertionError)):
        OP(IN, OUT)

@pytest.mark.unit
@pytest.mark.parametrize("dtype_a, dtype_b, expected", _MATMUL_DTYPE_PAIRS)
@pytest.mark.parametrize("shape_name, shape_a, shape_b, ref_shape", _MATMUL_SHAPES_CASES)
def test_dtype_promotion(dtype_a, dtype_b, expected, shape_name, shape_a, shape_b, ref_shape):
    tname = f"{shape_name}_{dtype_a}_{dtype_b}"
    OP, IN, OUT = prepare_op(tname, shape_a, shape_b, dtype_a, dtype_b)

    OP(IN, OUT)
    Y = OUT[0]
    assert Y.shape == ref_shape
    assert Y.dtype == expected
    assert Y.dtype == promote_types(IN[0].dtype, IN[1].dtype)
