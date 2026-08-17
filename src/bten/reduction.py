
from .registry import register_ops
from .shape_inference import argmax_sinf, reduce_sinf


def register_reduction_ops():
    _optbl = [
            ['ArgMax',          1, 1, 1, 1, argmax_sinf],
            ['ArgMin',          1, 1, 1, 1, argmax_sinf],

            ['ReduceL1',        2, 1, 1, 1, reduce_sinf],
            ['ReduceL2',        2, 1, 1, 1, reduce_sinf],
            ['ReduceLogSum',    2, 1, 1, 1, reduce_sinf],
            ['ReduceLogSumExp', 2, 1, 1, 1, reduce_sinf],
            ['ReduceMax',       2, 1, 1, 1, reduce_sinf],
            ['ReduceMin',       2, 1, 1, 1, reduce_sinf],
            ['ReduceMean',      2, 1, 1, 1, reduce_sinf],
            ['ReduceProd',      2, 1, 1, 1, reduce_sinf],
            ['ReduceSum',       2, 1, 1, 1, reduce_sinf],
            ['ReduceSumSquare', 2, 1, 1, 1, reduce_sinf],
            ]


    register_ops('reduction', _optbl)
