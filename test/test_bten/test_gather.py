
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
        ("2D axis-0",     [3, 4],       [0, 2],           0),
        ("2D axis-1",     [3, 4],       [1, 3],           1),
        ("3D axis-0",     [2, 3, 4],    [1],              0),
        ("3D axis-2",     [2, 3, 4],    [[0, 1], [2, 3]], 2),
        ("empty indices", [3, 4],       [],               0),
        ("gather0",       [5, 4, 3, 2], [0, 1, 3],        0),
        ("gather1",       [5, 4, 3, 2], [0, 1, 3],        1),
        ("2D indices",    [3, 3],       [[0, 2]],         1),
        ("neg indices",   [10],         [0, -9, -10],     0),
        ]

def prepare_op(_tname, _tshape, _indices, _axis):
    op_name = f"{_tname}"
    idx_data = np.array(_indices, dtype=np.int64)
    i_tensors = [
            make_tensor(name="X", shape=_tshape, dtype='float32'),
            make_tensor(name="I", shape=list(idx_data.shape), data=idx_data, dtype='int64'),
            ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": "Gather",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": _axis},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("tname, tshape, indices, axis", _POS_TESTCASES)
def test_gather(tname, tshape, indices, axis):
    OP, IN, OUT = prepare_op(tname, tshape, indices, axis)

    OP(IN, OUT)
    inf_shape = OUT[0].shape

    #ref impl
    _X0 = np.random.randn(*tshape)
    _Y  = np.take(_X0, indices, axis=axis)
    ref_shape = list(_Y.shape)

    assert inf_shape == ref_shape
