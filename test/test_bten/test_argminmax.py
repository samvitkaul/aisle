import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_ARGMINMAX_OPS= ['ArgMax', 'ArgMin', ]
_POS_TESTCASES = [

        ( '2d_ax0_kd',        [3, 4],        0,  1, 0, [1, 4],       ),
        ( '2d_ax1_kd',        [3, 4],        1,  1, 0, [3, 1],       ),
        ( '2d_ax0_nokd',      [3, 4],        0,  0, 0, [4],          ),
        ( '2d_ax1_nokd',      [3, 4],        1,  0, 0, [3],          ),
        ( '3d_ax0_kd',        [2, 3, 4],     0,  1, 0, [1, 3, 4],    ),
        ( '3d_ax1_kd',        [2, 3, 4],     1,  1, 0, [2, 1, 4],    ),
        ( '3d_ax2_kd',        [2, 3, 4],     2,  1, 0, [2, 3, 1],    ),
        ( '3d_ax2_nokd',      [2, 3, 4],     2,  0, 0, [2, 3],       ),
        ( '4d_ax0_kd',        [2, 3, 4, 5],  0,  1, 0, [1, 3, 4, 5], ),
        ( '4d_ax3_nokd',      [2, 3, 4, 5],  3,  0, 0, [2, 3, 4],    ),
        ( '1d_kd',            [5],           0,  1, 0, [1],          ),
        ( '1d_nokd',          [5],           0,  0, 0, [],           ),
        ( 'neg_axis_kd',      [2, 3, 4, 5], -1,  1, 0, [2, 3, 4, 1], ),
        ( 'neg_axis_nokd',    [2, 3, 4, 5], -2,  0, 0, [2, 3, 5],    ),
        #tname, shape, axis, keepdims, select_last_index, expected_shape
        ]

@pytest.mark.unit
@pytest.mark.parametrize("rop", _ARGMINMAX_OPS)
@pytest.mark.parametrize("tname, shape, axis, keepdims, select_last_index, expected_shape", _POS_TESTCASES)
def test_reduction(rop, tname, shape, axis, keepdims, select_last_index, expected_shape):
    i_tensors = [make_tensor(name='X', shape=shape, dtype='float32')]
    o_tensors = [make_tensor(name='Y')]
    opname = f'{rop}_{tname}'
    op_info = {
            'name': opname,
            'optype': rop,
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': axis, 'keepdims': keepdims, "select_last_index": select_last_index},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [opname]
    for x in o_tensors: x.op_out = [opname]
    op_obj(i_tensors, o_tensors)
    assert o_tensors[0].shape == expected_shape

