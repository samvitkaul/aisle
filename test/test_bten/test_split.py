

import numpy as np
import pytest

from src.bten.op import make_op
from src.bten.tensor import make_tensor

_POS_TESTCASES = [
        {
            "name": "test_split_equal_parts_1d_opset13",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "inputs": ["input"],
            "outputs": ["output_1", "output_2", "output_3"],
            "axis": 0,
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0]).astype(np.float32),
                np.array([5.0, 6.0]).astype(np.float32),
                ],
            },
        {
            "name": "test_split_variable_parts_1d_opset13",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "axis": 0,
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                ],
            },
        {
            "name": "test_split_equal_parts_2d_opset13",
            "X": np.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
                ).astype(np.float32),
            "inputs": ["input"],
            "outputs": ["output_1", "output_2"],
            "axis": 1,
            "expected_outputs": [
                np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
                np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
                ],
            },
        {
            "name": "test_split_variable_parts_2d_opset13",
            "X": np.array([
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "axis": 1,
            "expected_outputs": [
                np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
                np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                    np.float32
                    ),
                ],
            },
        {
                "name": "test_split_equal_parts_default_axis_opset13",
                "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                "inputs": ["input"],
                "outputs": ["output_1", "output_2", "output_3"],
                "expected_outputs": [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0]).astype(np.float32),
                    np.array([5.0, 6.0]).astype(np.float32),
                    ],
                },
    {
            "name": "test_split_variable_parts_default_axis_opset13",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_zero_size_splits_opset13",
            "X": np.array([]).astype(np.float32),  # 1D
            "split": np.array([0, 0, 0]).astype(
                np.int64
                ),  # Split emtpy tensor to tensors of size zero
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2", "output_3"],
            "expected_outputs": [
                np.array([]).astype(np.float32),
                np.array([]).astype(np.float32),
                np.array([]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_equal_parts_1d_opset18",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "inputs": ["input"],
            "outputs": ["output_1", "output_2", "output_3"],
            "axis": 0,
            "num_outputs": 3,
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0]).astype(np.float32),
                np.array([5.0, 6.0]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_variable_parts_1d_opset18",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "axis": 0,
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_equal_parts_2d",
            "X": np.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
                ).astype(np.float32),
            "inputs": ["input"],
            "outputs": ["output_1", "output_2"],
            "axis": 1,
            "num_outputs": 2,
            "expected_outputs": [
                np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
                np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_variable_parts_2d_opset18",
            "X": np.array([
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                ]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "axis": 1,
            "expected_outputs": [
                np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
                np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                    np.float32
                    ),
                ],
            },
    {
            "name": "test_split_equal_parts_default_axis_opset18",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "inputs": ["input"],
            "outputs": ["output_1", "output_2", "output_3"],
            "num_outputs": 3,
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0]).astype(np.float32),
                np.array([5.0, 6.0]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_variable_parts_default_axis_opset18",
            "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            "split": np.array([2, 4]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2"],
            "expected_outputs": [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                ],
            },
    {
            "name": "test_split_zero_size_splits_opset18",
            "X": np.array([]).astype(np.float32),
            "split": np.array([0, 0, 0]).astype(np.int64),
            "inputs": ["input", "split"],
            "outputs": ["output_1", "output_2", "output_3"],
            "expected_outputs": [
                np.array([]).astype(np.float32),
                np.array([]).astype(np.float32),
                np.array([]).astype(np.float32),
                ],
            },
    # FAILS RIGHT NOW!!
    # {
    #        'name'   : "test_split_1d_uneven_split_opset18",
    #        'X'      : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32),
    #        'inputs' : ["input"],
    #        'outputs': ["output_1", "output_2", "output_3", "output_4"],
    #        'num_outputs': 4,
    #        'expected_outputs': [
    #            np.array([1.0, 2.0]).astype(np.float32),
    #            np.array([3.0, 4.0]).astype(np.float32),
    #            np.array([5.0, 6.0]).astype(np.float32),
    #            np.array([7.0]).astype(np.float32),
    #            ]
    #        },
    # {
    #        'name'   : "test_split_2d_uneven_split_opset18",
    #        'X'      : np.array( [ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    #                              [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    #                              ]).astype(np.float32),
    #        'inputs' : ["input"],
    #        'outputs': ["output_1", "output_2", "output_3"],
    #        'axis'   : 1,
    #        'num_outputs': 3,
    #        'expected_outputs': [
    #            np.array([[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]).astype(np.float32),
    #            np.array([[4.0, 5.0, 6.0], [12.0, 13.0, 14.0]]).astype(np.float32),
    #            np.array([[7.0, 8.0], [15.0, 16.0]]).astype(np.float32),
    #            ]
    #        }
]


@pytest.mark.unit
@pytest.mark.parametrize("tno, trec", enumerate(_POS_TESTCASES))
def test_split(tno, trec):
    opname = f"Split_{tno}"
    XShape = list(trec['X'].shape)
    i_tensors = [make_tensor(name='X', shape=XShape, dtype='float32')]

    if "split" in trec:
        split_data = trec['split']
        i_tensors.append(
                make_tensor(name="S", shape=list(split_data.shape), dtype='int64', data=split_data)
                )

    attrs = {}
    if 'axis' in trec: attrs['axis'] = trec['axis']
    if 'num_outputs' in trec: attrs['num_outputs'] = trec['num_outputs']

    num_outputs = len(trec['expected_outputs'])
    o_tensors = [make_tensor(name=f"O{i}") for i in range(num_outputs)]

    op_info = {
        "name": opname,
        "optype": "Split",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
        "attrs": attrs,
    }
    op_obj = make_op(**op_info)
    for t in i_tensors: t.op_in = [opname]
    for t in o_tensors: t.op_out = [opname]

    op_obj(i_tensors, o_tensors)

    assert all(o_tensors[i].shape == list(trec['expected_outputs'][i].shape) for i in range(num_outputs))
