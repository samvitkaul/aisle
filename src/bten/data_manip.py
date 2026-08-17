from .registry import register_ops
from .shape_inference import (
    concat_sinf,
    gather_sinf,
    reshape_sinf,
    scatternd_sinf,
    slice_sinf,
    split_sinf,
    squeeze_sinf,
    transpose_sinf,
    trilu_sinf,
    unsqueeze_sinf,
)


def register_data_manip_ops():
    _optbl = [

            ['Transpose', 1, 1,  1,  1, transpose_sinf, {'perm'}     ],
            ['Gather',    2, 2,  1,  1, gather_sinf,    {'axis'}     ],
            ['ScatterND', 3, 3,  1,  1, scatternd_sinf, {'reduction'}],
            ['Reshape',   2, 2,  1,  1, reshape_sinf                 ],
            ['Slice',     5, 3,  1,  1, slice_sinf                   ],
            ['Trilu',     2, 1,  1,  1, trilu_sinf                   ],
            ['Unsqueeze', 2, 2,  1,  1, unsqueeze_sinf               ],
            ['Squeeze',   2, 1,  1,  1, squeeze_sinf                 ],
            ['Split',     2, 1,  2147483647,  1, split_sinf, {'axis'}],
            ['Concat',    2147483647, 1, 1, 1,  concat_sinf, {'axis'}],

            ]
    register_ops('data_manip', _optbl)
