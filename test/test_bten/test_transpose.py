
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
        ("2D Matrix", [3, 4],       [1, 0]      ),
        ("1D Vector", [5],          [0]         ),
        ("3D Vector", [2, 3, 4],    [1, 0, 2]   ),
        ("4D Vector", [2, 3, 4, 5], [3, 2, 1, 0]),
        ("Empty Dim", [3, 0, 2],    [2, 1, 0]   ),
        ("Scalar",    [],           []          ),
        ]

def prepare_op(_tname, _tshape, _perms):
    op_name = f"{_tname}"
    i_tensors = [make_tensor(name="X", shape=_tshape, dtype='float32')]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": "Transpose",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"perm": _perms},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("tname, tshape, perms", _POS_TESTCASES)
def test_transpose(tname, tshape, perms):
    OP, IN, OUT = prepare_op(tname, tshape, perms)

    OP(IN, OUT)
    inf_shape = OUT[0].shape

    #ref impl
    _X0 = np.random.randn(*tshape)
    _Y  = np.transpose(_X0, perms)
    ref_shape = list(_Y.shape)

    assert inf_shape == ref_shape
