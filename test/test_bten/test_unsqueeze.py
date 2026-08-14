

import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
        # --- Basic single axis ---
        ("1d_axis0", (4,), [0], [1, 4]),
        ("1d_axis1", (4,), [1], [4, 1]),
        ("1d_axis_neg1", (4,), [-1], [4, 1]),
        # --- 2D input ---
        ("2d_axis0", (3, 4), [0], [1, 3, 4]),
        ("2d_axis1", (3, 4), [1], [3, 1, 4]),
        ("2d_axis2", (3, 4), [2], [3, 4, 1]),
        ("2d_axis_neg1", (3, 4), [-1], [3, 4, 1]),
        # --- 3D input ---
        ("3d_axis0", (2, 3, 4), [0], [1, 2, 3, 4]),
        ("3d_axis2", (2, 3, 4), [2], [2, 3, 1, 4]),
        ("3d_axis3", (2, 3, 4), [3], [2, 3, 4, 1]),
        # --- 4D NCHW ---
        ("4d_axis0", (1, 3, 8, 8), [0], [1, 1, 3, 8, 8]),
        ("4d_axis4", (1, 3, 8, 8), [4], [1, 3, 8, 8, 1]),
        # --- Multiple axes at once ---
        ("multi_0_1", (3, 4), [0, 1], [1, 1, 3, 4]),
        ("multi_0_3", (3, 4), [0, 3], [1, 3, 4, 1]),
        ("multi_1_3", (2, 3), [1, 3], [2, 1, 3, 1]),
        # --- Scalar input ---
        ("scalar_axis0", (), [0], [1]),
        ("scalar_multi", (), [0, 1], [1, 1]),
        ]

@pytest.mark.unit
@pytest.mark.parametrize("tname,shape,axes,expected", _POS_TESTCASES)
def test_unsqueeze(tname, shape, axes, expected):
    axes_np = np.array(axes, dtype=np.int64)
    i_tensors = [
            make_tensor(name='X', dtype='float32', shape=shape),
            make_tensor(name='axes', dtype='int64', data=axes_np, shape=list(axes_np.shape)),
            ]
    o_tensors = [make_tensor(name="Y")]

    op_info = {
        "name": "tname",
        "optype": "Unsqueeze",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
    }
    op_obj = make_op(**op_info)
    for t in i_tensors: t.op_in = [tname]
    for t in o_tensors: t.op_out = [tname]

    op_obj(i_tensors, o_tensors)

    assert o_tensors[0].shape == expected

