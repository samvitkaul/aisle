
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor


def ref_impl_slice(data, starts, ends, axes=None, steps=None):
    """Reference implementation of ONNX Slice."""
    slices = [slice(None)] * len(data.shape)
    if axes is None:
        axes = list(range(len(starts)))
    for i, axis in enumerate(axes):
        s = starts[i]
        e = ends[i]
        st = steps[i] if steps is not None else 1
        slices[axis] = slice(int(s), int(e), int(st))
    return data[tuple(slices)]


def _make_slice_tensors(data0, starts, ends, axes=None, steps=None):
    """Build input tensor list for slice op (always 5 inputs)."""
    if axes is None: axes = list(range(len(starts)))
    if steps is None: steps = [1] * len(starts)

    data1 = np.array(starts, dtype=np.int64)
    data2 = np.array(ends,   dtype=np.int64)
    data3 = np.array(axes,   dtype=np.int64)
    data4 = np.array(steps,  dtype=np.int64)

    tensors = [
            make_tensor(name="X",      dtype='float32', shape=list(data0.shape), data=data0),
            make_tensor(name="starts", dtype='int64',   shape=list(data1.shape), data=data1),
            make_tensor(name="ends",   dtype='int64',   shape=list(data2.shape), data=data2),
            make_tensor(name="axes",   dtype='int64',   shape=list(data3.shape), data=data3),
            make_tensor(name="steps",  dtype='int64',   shape=list(data4.shape), data=data4),
    ]
    return tensors


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


_POS_TESTCASES = [
    # Basic slicing along different axes
    ("1D basic",             [10],         [2],       [7],       [0],       [1]      ),
    ("1D with step",         [10],         [1],       [8],       [0],       [2]      ),
    ("2D row slice",         [4, 6],       [1],       [3],       [0],       [1]      ),
    ("2D col slice",         [4, 6],       [2],       [5],       [1],       [1]      ),
    ("2D both axes",         [4, 6],       [1, 2],    [3, 5],    [0, 1],    [1, 1]   ),
    ("3D single axis",       [2, 4, 6],    [1],       [3],       [1],       [1]      ),
    ("3D two axes",          [2, 4, 6],    [0, 2],    [2, 5],    [0, 2],    [1, 1]   ),
    ("3D all axes",          [2, 4, 6],    [0, 1, 2], [2, 3, 5], [0, 1, 2], [1, 1, 1]),
    ("4D NCHW spatial crop", [1, 3, 8, 8], [2, 2],    [6, 6],    [2, 3],    [1, 1]   ),
    ("4D batch slice",       [4, 3, 8, 8], [1],       [3],       [0],       [1]      ),
    # Slicing with steps
    ("1D step 3", [12], [0], [12], [0], [3]),
    ("2D step on rows", [8, 4], [0], [8], [0], [2]),
    ("2D step on both", [8, 6], [0, 0], [8, 6], [0, 1], [2, 3]),
    ("4D spatial stride 2", [1, 3, 8, 8], [0, 0], [8, 8], [2, 3], [2, 2]),
    # Edge cases: slice to single element
    ("Single row", [4, 6], [2], [3], [0], [1]),
    ("Single col", [4, 6], [3], [4], [1], [1]),
    ("Single element 2D", [4, 6], [1, 2], [2, 3], [0, 1], [1, 1]),
    # Full slice (no-op)
    ("Full 1D", [8], [0], [8], [0], [1]),
    ("Full 2D", [3, 4], [0, 0], [3, 4], [0, 1], [1, 1]),
    # Negative indices
    ("1D negative end", [10], [0], [-2], [0], [1]),
    ("2D negative start and end", [6, 8], [-4], [-1], [0], [1]),
    # Large tensor
    ("Large 4D crop", [2, 16, 32, 32], [8, 8], [24, 24], [2, 3], [1, 1]),
]

_NEG_TESTCASES = [
    ("Empty result", [4, 6], [2], [2], [0], [1]),
    ("Step larger than range", [10], [0], [3], [0], [5]),
    ("Slice on zero-dim", [0, 4], [0], [0], [0], [1]),
]


@pytest.mark.unit
@pytest.mark.parametrize("tname, data_shape, starts, ends, axes, steps", _POS_TESTCASES)
def test_slice(tname, data_shape, starts, ends, axes, steps):
    op_name = tname
    # Generate random data
    data = np.array(np.random.randn(*data_shape), dtype=np.float32)

    # Compute reference output to get expected shape
    ref_output = ref_impl_slice(data, starts, ends, axes, steps)
    expected_shape = list(ref_output.shape)

    # Build input tensors
    i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
    o_tensors = [make_tensor(name="Y")]

    op_info = {
            "name": op_name,
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_out = [op_name]

    op_obj(i_tensors, o_tensors)
    inf_shape = o_tensors[0].shape
    assert inf_shape == expected_shape


# _NEG_TESTCASES not working?? TODO:
#@pytest.mark.unit
#@pytest.mark.parametrize("tname, data_shape, starts, ends, axes, steps", _NEG_TESTCASES)
#def test_slice_neg(tname, data_shape, starts, ends, axes, steps):
#    op_name = f"{tname}_neg"
#
#    # Generate random data
#    data = np.empty(data_shape, dtype=np.float32)
#    try:
#        ref_output = ref_impl_slice(data, starts, ends, axes, steps)
#        expected_shape = list(ref_output.shape)
#    except Exception:
#        expected_shape = [0]
#
#    i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
#    o_tensors = [make_tensor(name="Y")]
#
#    op_info = {
#            "name": op_name,
#            "optype": "Slice",
#            "inList": [x.name for x in i_tensors],
#            "outList": [x.name for x in o_tensors],
#            "attrs": {"out_shape": expected_shape},
#            }
#    op_obj = make_op(**op_info)
#    for x in i_tensors: x.op_in = [op_name]
#    for x in o_tensors: x.op_out = [op_name]
#
#    with pytest.raises((ValueError, AssertionError, IndexError)):
#        op_obj(i_tensors, o_tensors)
#
