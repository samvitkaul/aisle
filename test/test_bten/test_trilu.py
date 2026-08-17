
import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_UPPER_TESTCASES = [
        ("triu",             10, (4, 5),    None),
        ("triu_neg",         10, (4, 5),      -1),
        ("triu_out_neg_out", 10, (4, 5),      -7),
        ("triu_pos",         10, (4, 5),       2),
        ("triu_out_pos",     10, (4, 5),       6),
        ("triu_square",      10, (2, 3, 3), None),
        ("triu_square_neg",  10, (2, 3, 3),   -1),
        ("triu_one_row",     10, (3, 1, 5),    1),
        ("triu_zero",        10, (0, 5),       6),
        ]

_LOWER_TESTCASES = [
        ("tril",             10, (4, 5),    None),
        ("tril_neg",         10, (4, 5),      -1),
        ("tril_out_neg_out", 10, (4, 5),      -7),
        ("tril_pos",         10, (4, 5),       2),
        ("tril_out_pos",     10, (4, 5),       6),
        ("tril_square",      10, (2, 3, 3), None),
        ("tril_square_neg",  10, (2, 3, 3),   -1),
        ("tril_one_row",     10, (3, 1, 5),    1),
        ("tril_zero",        10, (0, 5),       6),
        ]


@pytest.mark.unit
@pytest.mark.parametrize("tname,tdim,tsize,kval", _UPPER_TESTCASES)
def test_trilu_upper(tname, tdim, tsize, kval):
    x = np.random.randint(tdim, size=tsize).astype(np.int64)
    if kval:
        k = np.array(kval).astype(np.int64)
        y = np.triu(x, int(k))
    else:
        k = None
        y = np.triu(x)

    i_tensors = [make_tensor(name='X', shape=list(x.shape), dtype='int64', data=x)]
    if k:
        i_tensors.append(make_tensor(name='K', shape=list(k.shape), dtype='int64', data=k))

    o_tensors = [make_tensor(name="Y")]

    op_info = {
        "name": tname,
        "optype": "Trilu",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
        "attrs": {"upper": 1}
    }
    op_obj = make_op(**op_info)
    for t in i_tensors: t.op_in = [tname]
    for t in o_tensors: t.op_out = [tname]

    op_obj(i_tensors, o_tensors)

    assert o_tensors[0].shape == list(y.shape)

@pytest.mark.unit
@pytest.mark.parametrize("tname,tdim,tsize,kval", _LOWER_TESTCASES)
def test_trilu_lower(tname, tdim, tsize, kval):
    x = np.random.randint(tdim, size=tsize).astype(np.int64)
    if kval:
        k = np.array(kval).astype(np.int64)
        y = np.triu(x, int(k))
    else:
        k = None
        y = np.triu(x)

    i_tensors = [make_tensor(name='X', shape=list(x.shape), dtype='int64', data=x)]
    if k:
        i_tensors.append(make_tensor(name='K', shape=list(k.shape), dtype='int64', data=k))

    o_tensors = [make_tensor(name="Y")]

    op_info = {
        "name": tname,
        "optype": "Trilu",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
        "attrs": {"upper": 0}
    }
    op_obj = make_op(**op_info)
    for t in i_tensors: t.op_in = [tname]
    for t in o_tensors: t.op_out = [tname]

    op_obj(i_tensors, o_tensors)

    assert o_tensors[0].shape == list(y.shape)

