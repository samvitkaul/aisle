

from functools import reduce as _reduce_ln

import numpy as np
import pytest
from onnx.backend.test.case.node.layernormalization import _layer_normalization, calculate_normalized_shape

from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.utils.data_types import DataType
from src.utils.data_types import promote_types as _promote_types_ln

_POS_TESTCASES = [
        {
            "name": "test_layer_normalization_4d",
            "x": [2, 3, 4, 5],
            "in": ["X", "W", "B"],
            "out": ["Y", "Mean", "InvStdDev"],
            },
        {
            "name": "test_layer_normalization_default_axis",
            "x": [2, 3, 4, 5],
            "in": ["X", "W", "B"],
            "out": ["Y", "Mean", "InvStdDev"],
            },
        {
            "name": "test_layer_normalization_2d",
            "x": [3, 4],
            "in": ["X", "W", "B"],
            "out": ["Y", "Mean", "InvStdDev"],
            },
        {
            "name": "test_layer_normalization_3d_epsilon",
            "x": [2, 3, 5],
            "in": ["X", "W", "B"],
            "out": ["Y", "Mean", "InvStdDev"],
            "eps": 1e-1,
            },
        ]

_LN_DTYPE_CASES = [
        ('2in_identity_fp32',        'float32',  'float32', None,      DataType.FLOAT32),
        ('2in_mixed_bf16_fp32',      'bfloat16', 'float32', None,      DataType.FLOAT32),
        ('3in_mixed_bf16_fp32_fp32', 'bfloat16', 'float32', 'float32', DataType.FLOAT32),
        ('poison_bf16_undef',        'bfloat16', 'undef',   None,      DataType.UNDEF),
        ]

@pytest.mark.unit
@pytest.mark.parametrize("tno, trec", enumerate(_POS_TESTCASES))
def test_layernorm(tno, trec):
    tname = trec['name']
    if tname.endswith('default_axis'):
        axes = [-1]
        names = [tname]
    else:
        xrank = len(trec['x'])
        axes = [i for i in range(xrank)]
        axes += [i - xrank for i in range(xrank)]
        names = [f"{tname}_neg_axis_{-a}" if a < 0 else f"{tname}_axis_{a}" for a in axes]
    trec['axes']  = axes
    trec['names'] = names

    for cno, axis in enumerate(trec['axes']):
        test_name = trec['names'][cno]
        op_name = f"{test_name}_{tno}_{cno}"

        XShape = trec['x']
        normalized_shape = calculate_normalized_shape(XShape, axis)
        X = np.random.randn(*XShape).astype(np.float32)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        B = np.random.randn(*normalized_shape).astype(np.float32)
        attrs = {'axis': axis}
        if 'eps' in trec:
            eps = trec['eps']
            attrs['epsilon'] = eps
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, eps)
        else:
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)

        o0Shape = list(Y.shape)
        o1Shape = list(mean.shape)
        o2Shape = list(inv_std_dev.shape)

        i_tensors = [
                make_tensor(name="X", shape=XShape,           dtype='float32'), #data
                make_tensor(name="W", shape=normalized_shape, dtype='float32'), #scale
                make_tensor(name="B", shape=normalized_shape, dtype='float32'), #bias
                ]
        o_tensors = [
                make_tensor(name="Y"),
                make_tensor(name="mean"),
                make_tensor(name="inv_std_dev"),
                ]
        op_info = {
                'name': op_name,
                'optype': 'LayerNormalization',
                'inList': [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs': attrs,
                }
        op_obj = make_op(**op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj(i_tensors, o_tensors)
        assert o_tensors[0].shape == o0Shape
        assert o_tensors[1].shape == o1Shape
        assert o_tensors[2].shape == o2Shape


@pytest.mark.unit
@pytest.mark.parametrize("case_name, x_dtype, scale_dtype, bias_dtype, expected_Y", _LN_DTYPE_CASES)
def test_layernorm_dtype_promo(case_name, x_dtype, scale_dtype, bias_dtype, expected_Y):
    """
      Y.dtype must equal reduce(promote_types, [X, scale, bias?])
      mean/inv_std_dev remain pinned to X.dtype - need to update with stash_type
      current guard to assert this assumption explicitly
    """
    XShape = [3, 4]
    normalized_shape = [4]
    op_name = 'LN_promote_{case_name}'
    i_tensors = [
                make_tensor(name="X", shape=XShape,           dtype=x_dtype),
                make_tensor(name="W", shape=normalized_shape, dtype=scale_dtype),
                ]
    if bias_dtype is not None:
        i_tensors.append(
                make_tensor(name="B", shape=normalized_shape, dtype=bias_dtype)
                )

    o_tensors = [
            make_tensor(name="Y"),
            make_tensor(name="mean"),
            make_tensor(name="inv_std_dev"),
            ]
    op_info = {
            'name': op_name,
            'optype': 'LayerNormalization',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': -1},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_out = [op_name]
    op_obj(i_tensors, o_tensors)

    assert o_tensors[0].shape == XShape
    assert o_tensors[0].dtype == expected_Y

    expected_fold = _reduce_ln(
            _promote_types_ln,
            [t.dtype for t in i_tensors],
            )
    assert o_tensors[0].dtype == expected_fold
    #mean/inv_std_dev remain pinned to X.dtype; TODO: update when stash_type is implemented
    assert o_tensors[1].dtype == i_tensors[0].dtype
    assert o_tensors[2].dtype == i_tensors[0].dtype

@pytest.mark.unit
def test_layernorm_dtype_promo_single_output_2in():
    XShape = [2, 4]
    normalized_shape = [4]
    op_name = 'LN_promote_single_out'
    i_tensors = [
                make_tensor(name="X", shape=XShape,           dtype='float32'),
                make_tensor(name="W", shape=normalized_shape, dtype='float32'),
                ]
    o_tensors = [make_tensor(name="Y")]
    op_info = {
            'name': op_name,
            'optype': 'LayerNormalization',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': -1},
            }
    op_obj = make_op(**op_info)
    for x in i_tensors: x.op_in = [op_name]
    for x in o_tensors: x.op_out = [op_name]
    op_obj(i_tensors, o_tensors)
    assert o_tensors[0].shape == XShape
    assert o_tensors[0].dtype == DataType.FLOAT32
