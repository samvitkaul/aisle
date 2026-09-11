
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from itertools import pairwise

import src.front.functional as F
import src.front.module as nn
from src import make_tensor


def get_activation(act: str):
    _tbl = {
            'RELU': F.Relu,
            'GELU': F.Gelu,
            'SIGMOID': F.Sigmoid,
            }
    return _tbl[act.upper()]

class BasicMLP(nn.Module):
    def __init__(self, name, **cfg):
        super().__init__(name)
        self.mm_dims    = cfg['mm_dims']
        self.bias       = cfg.get('bias', False)
        self.activation = cfg.get('activation', 'Relu')
        self.dtype      = cfg.get('dtype', 'float32')
        self.seqlen     = cfg.get('seqlen', 5)
        self.act_cls    = get_activation(self.activation)

        self.opblk_count = 0
        for i, (M,N) in enumerate(pairwise(self.mm_dims)):
            setattr(self,
                    f'linear_{i}',
                    nn.Linear(self.name + f'.Linear{i}',
                             M,
                             N,
                             dtype=self.dtype,
                             bias=self.bias)
                    )
            setattr(self,
                    f'act_{i}',
                    self.act_cls(self.name + f'.{self.activation}{i}')
                    )
            self.opblk_count += 1

    def inputs(self, bs=1):
        x = make_tensor(name='x', shape=[bs, self.seqlen, self.mm_dims[0]], dtype=self.dtype)
        return (x,)

    def forward(self, x):
        linear_ops = [getattr(self, f'linear_{i}') for i in range(self.opblk_count)]
        act_ops    = [getattr(self, f'act_{i}') for i in range(self.opblk_count)]

        y = x
        for l,a in zip(linear_ops, act_ops):
            y = l(y)
            y = a(y)
        return y

if __name__ == '__main__':
    from src.graph import graph2onnx

    cfg = {
            'mm_dims': [32, 128, 256, 64, 10],
            'bias': True,
            'activation': 'gelu'
            }
    M = BasicMLP('BasicMLP', **cfg)
    x = M.inputs()
    y = M(*x)
    G = M.get_forward_graph(*x)
    graph2onnx(G, 'BasicMLP.onnx')

