from .registry import register_ops
from .shape_inference import ln_sinf

def register_nn_ops():
    _optbl = [
            ['LayerNormalization', 3,  2,  3,  1, ln_sinf,],    
            ]

    register_ops('nn', _optbl)
    return

