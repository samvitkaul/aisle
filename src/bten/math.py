
from .registry import register_ops
from .shape_inference import (
    bidir_bcast,
    gelu_sinf,
    matmul_sinf,
    softmax_sinf,
    topk_sinf,
    unary_fwd,
)


def register_math_ops():
    _binary_optbl = [
            ['Add',    2,  2, 1,  1,  bidir_bcast, ],
            ['Sub',    2,  2, 1,  1,  bidir_bcast, ],
            ['Mul',    2,  2, 1,  1,  bidir_bcast, ],
            ['Div',    2,  2, 1,  1,  bidir_bcast, ],
            ['Pow',    2,  2, 1,  1,  bidir_bcast, ],
            ['MatMul', 2,  2, 1,  1,  matmul_sinf, ],
            ]

    _unary_optbl = [
            ['Relu',    1,  1,  1,  1,  unary_fwd,  ],
            ['Tanh',    1,  1,  1,  1,  unary_fwd,  ],
            ['Sigmoid', 1,  1,  1,  1,  unary_fwd,  ],
            ]

    _x_optbl = [
            ['Softmax', 1,  1,  1,  1,  softmax_sinf,  {'axis'},                      {'dim': 'axis'}],
            ['Gelu',    1,  1,  1,  1,  gelu_sinf,     {'approximate'},                              ],
            ['TopK',    2,  2,  2,  2,  topk_sinf,     {'axis', 'largest', 'sorted'}, {'dim': 'axis'}],
            ]

    register_ops('math', _binary_optbl + _unary_optbl + _x_optbl)
