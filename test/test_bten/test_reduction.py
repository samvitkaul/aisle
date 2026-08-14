import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_REDUCE_OPS = [
        'ReduceL1',
        'ReduceL2',
        'ReduceLogSum',
        'ReduceLogSumExp',
        'ReduceMax',
        'ReduceMin',
        'ReduceMean',
        'ReduceProd',
        'ReduceSum',
        'ReduceSumSquare',
        ]
_POS_TESTCASES = [
        ( '1d_all_kd',        [8],          None,     1, 0, [1],          ),
        ( '2d_all_kd',        [3, 4],       None,     1, 0, [1, 1],       ),
        ( '3d_all_kd',        [2, 3, 4],    None,     1, 0, [1, 1, 1],    ),
        ( '2d_all_nokd',      [3, 4],       None,     0, 0, [],           ),
        ( '3d_all_nokd',      [2, 3, 4],    None,     0, 0, [],           ),
        ( '2d_noop',          [3, 4],       None,     1, 1, [3, 4],       ),
        ( '2d_ax0_kd',        [3, 4],       [0],      1, 0, [1, 4],       ),
        ( '2d_ax1_kd',        [3, 4],       [1],      1, 0, [3, 1],       ),
        ( '2d_ax0_nokd',      [3, 4],       [0],      0, 0, [4],          ),
        ( '3d_ax0_kd',        [2, 3, 4],    [0],      1, 0, [1, 3, 4],    ),
        ( '3d_ax1_kd',        [2, 3, 4],    [1],      1, 0, [2, 1, 4],    ),
        ( '3d_ax2_kd',        [2, 3, 4],    [2],      1, 0, [2, 3, 1],    ),
        ( '3d_ax2_nokd',      [2, 3, 4],    [2],      0, 0, [2, 3],       ),
        ( '3d_ax02_kd',       [2, 3, 4],    [0, 2],   1, 0, [1, 3, 1],    ),
        ( '3d_ax02_nokd',     [2, 3, 4],    [0, 2],   0, 0, [3],          ),
        ( '4d_ax23_kd',       [2, 3, 4, 5], [2, 3],   1, 0, [2, 3, 1, 1], ),
        ( '3d_neg_ax_kd',     [2, 3, 4],    [-1],     1, 0, [2, 3, 1],    ),
        ( '3d_neg_multi_kd',  [2, 3, 4],    [-1, -2], 1, 0, [2, 1, 1],    ),
        #tname, shape, axes, keepdims, noop, expected_shape
        ]

@pytest.mark.unit
@pytest.mark.parametrize("rop", _REDUCE_OPS)
@pytest.mark.parametrize("tname, shape, axes, keepdims, noop, expected_shape", _POS_TESTCASES)
def test_reduction(rop, tname, shape, axes, keepdims, noop, expected_shape):
    i_tensors = [make_tensor(name='X', shape=shape, dtype='float32')]
    if axes:
        axes_data = np.array(axes, dtype=np.int64)
        axes_shape= list(axes_data.shape)
        i_tensors.append(make_tensor(name='axes', shape=axes_shape, data=axes_data, dtype='int64'))
    o_tensors = [make_tensor(name='Y')]
    opname = f'{rop}_{tname}'
    op_info = {
            'name': opname,
            'optype': rop,
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'keepdims': keepdims, "noop_with_empty_axes": noop},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [opname]
    for x in o_tensors: x.op_out = [opname]
    op_obj(i_tensors, o_tensors)
    assert o_tensors[0].shape == expected_shape

