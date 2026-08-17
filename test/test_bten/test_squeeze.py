
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
        ("remove_first",           [1, 3, 4],       [0],       [3, 4],    ),
        ("remove_middle",          [3, 1, 4],       [1],       [3, 4],    ),
        ("remove_last",            [3, 4, 1],       [2],       [3, 4],    ),
        ("remove_two_leading",     [1, 1, 3, 4],    [0, 1],    [3, 4],    ),
        ("remove_non_adjacent",    [1, 3, 1, 4],    [0, 2],    [3, 4],    ),
        ("squeeze_all_to_scalar",  [1, 1, 1],       [0, 1, 2], [],        ),
        ("remove_three",           [1, 3, 1, 4, 1], [0, 2, 4], [3, 4],    ),
        ("no_axes_no_change",      [2, 3, 4],       [],        [2, 3, 4], ),  # empty axes, no size-1 dims
        ("single_elem_to_scalar",  [1],             [0],       [],        ),
        ("remove_one_of_two",      [1, 1],          [0],       [1],       ),
        ]


@pytest.mark.unit
@pytest.mark.parametrize("tname,shape,axes,expected", _POS_TESTCASES)
def test_squeeze(tname, shape, axes, expected):
    axes_np = np.asarray(axes, dtype=np.int64)

    i_tensors = [
            make_tensor(name='X', dtype='float32', shape=shape),
            make_tensor(name='axes', dtype='int64', data=axes_np, shape=list(axes_np.shape)),
            ]
    o_tensors = [make_tensor(name="Y")]

    op_info = {
        "name": tname,
        "optype": "Squeeze",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
    }
    op = make_op(**op_info)
    for t in i_tensors: t.op_in = [tname]
    for t in o_tensors: t.op_out = [tname]

    op(i_tensors, o_tensors)

    assert o_tensors[0].shape == expected

