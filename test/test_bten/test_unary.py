
import pytest

from src.bten.tensor import make_tensor
from src.bten.op import make_op

import numpy as np


@pytest.mark.unit
@pytest.mark.parametrize("optype", ['Relu', 'Tanh', 'Sigmoid'])
@pytest.mark.parametrize(
        "tname,tdim", [
            ["OD", []],
            ["1D", [4]],
            ["2D", [3, 1]],
            ["3D", [2, 1, 4]],
            ],
        )
def test_unary(optype, tname, tdim):
    op_name = f"{optype}_{tname}"
    i_tensors = [make_tensor(name="X", shape=tdim, dtype='float32')]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            "name": op_name,
            "optype": optype,
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_in = [op_name]

    op_obj(i_tensors, o_tensors)
    inf_shape = o_tensors[0].shape

    _X0 = np.random.randn(*tdim)
    _Y0 = np.abs(_X0)
    ref_shape = list(_Y0.shape)

    assert inf_shape == ref_shape

