
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
    ("Basic reshape with -1, allowzero=0",     [2, 3, 4], [2, -1],          0),
    ("Copy dim with 0, allowzero=0",           [2, 3, 4], [2, 0, -1],       0),
    ("All specified dims, allowzero=0",        [2, 3, 4], [4, 3, 2],        0),
    ("reordered_all_dims, allowzero=0",        [2, 3, 4], [4, 2, 3],        0),
    ("reordered_last_dims, allowzero=0",       [2, 3, 4], [2, 4, 3],        0),
    ("reduced_dims, allowzero=0",              [2, 3, 4], [2, 12],          0),
    ("extended_dims, allowzero=0",             [2, 3, 4], [2, 3, 2, 2],     0),
    ("one_dim, allowzero=0",                   [2, 3, 4], [24],             0),
    ("negative_dim, allowzero=0",              [2, 3, 4], [2, -1, 2],       0),
    ("negative_extended_dims, allowzero=0",    [2, 3, 4], [-1, 2, 3, 4],    0),
    ("zero_dim, allowzero=0",                  [2, 3, 4], [2, 0, 4, 1],     0),
    ("zero_and_negative_dim, allowzero=0",     [2, 3, 4], [2, 0, 1, -1],    0),
    ("Basic reshape with -1, allowzero=1",     [2, 3, 4], [2, -1],          1),
    ("All specified dims, allowzero=1",        [2, 3, 4], [4, 3, 2],        1),
    ("reordered_all_dims, allowzero=1",        [2, 3, 4], [4, 2, 3],        1),
    ("reordered_last_dims, allowzero=1",       [2, 3, 4], [2, 4, 3],        1),
    ("reduced_dims, allowzero=1",              [2, 3, 4], [2, 12],          1),
    ("extended_dims, allowzero=1",             [2, 3, 4], [2, 3, 2, 2],     1),
    ("one_dim, allowzero=1",                   [2, 3, 4], [24],             1),
    ("negative_dim, allowzero=1",              [2, 3, 4], [2, -1, 2],       1),
    ("negative_extended_dims, allowzero=1",    [2, 3, 4], [-1, 2, 3, 4],    1),
]

def prepare_op(_tname, _tshape, _oshape, _allowzero):
    op_name = f"{_tname}"
    s_data = np.array(_oshape, dtype=np.int64)
    i_tensors = [
            make_tensor(name="X", shape=_tshape, dtype='float32'),
            make_tensor(name="S", shape=list(s_data.shape), data=s_data, dtype='int64', is_const=True),
            ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": "Reshape",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"allowzero": _allowzero},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    return op_obj, i_tensors, o_tensors


@pytest.mark.unit
@pytest.mark.parametrize("tname, tshape, tgt_shape, allowzero", _POS_TESTCASES)
def test_reshape(tname, tshape, tgt_shape, allowzero):
    OP, IN, OUT = prepare_op(tname, tshape, tgt_shape, allowzero)

    OP(IN, OUT)
    inf_shape = OUT[0].shape

    #ref impl
    data = np.random.random_sample(tshape).astype(np.float32)
    shape = np.array(tgt_shape, dtype=np.int64)
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    _Y = np.reshape(data, new_shape)
    ref_shape = list(_Y.shape)

    assert inf_shape == ref_shape
