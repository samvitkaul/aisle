from ..utils.common import prod_ints
from ..utils.data_types import DataType, str2dt, get_bpe
from ..utils.sym import SymDim, SymExpr

from typing import List, Optional, Any 


class Tensor:
    __slots__ = (
            'name',
            'dtype',
            'shape',
            'op_in',
            'op_out',
            'is_param',
            'is_const',
            'is_view',
            'data',
            'location',
            )
    _KNOWN_KWARGS = frozenset({
            'dtype',
            'shape',
            'op_in',
            'op_out',
            'is_param',
            'is_const',
            'is_view',
            'data',
            'location',
        })

    def __init__(self, name: str, **kwargs):
        unknown = set(kwargs) - self._KNOWN_KWARGS
        if unknown:
            raise TypeError(f"Tensor() got unexpected keyword args: {unknown}")

        self.name      : str           = name
        self.dtype     : DataType      = str2dt(kwargs.get('dtype', 'undef'))
        self.shape     : Optional[list]= kwargs.get('shape')
        self.op_in     : List[str]     = kwargs.get('op_in', [])
        self.op_out    : List[str]     = kwargs.get('op_out', [])
        self.is_param  : bool          = kwargs.get('is_param', False)
        self.is_const  : bool          = kwargs.get('is_const', False)
        self.is_view   : bool          = kwargs.get('is_view',  False)
        self.data      : Optional[Any] = kwargs.get('data',     None)
        self.location  : Optional[Any] = kwargs.get('resolve',  None)

    def rank(self): return len(self.shape) #type: ignore[arg-type]

    def nelems(self):
        trank = self.rank()
        if trank > 0:
            res = prod_ints(self.shape) #type: ignore[arg-type]
        elif trank == 0:
            res = 1
        else:
            raise ValueError(f"What kinda tensor is this!!\n{self}")
        return res

    def nbytes(self):
        if not self._is_concrete():
            raise ValueError(
                    f"Cannot compute nbytes() on tensor with symbolic shape."
                    f"Use nbytes_assuming_concrete() instead for a best effort estimate"
                    )
        return self.nelems() * get_bpe(self.dtype)

    def nbytes_assuming_concrete(self):
        """ preserves symbolic exprs if they exist """
        return self.nelems() * get_bpe(self.dtype)

    def check_shape(self) -> bool:
        return self.rank() == 0 or (self.shape and all(isinstance(d, (int, SymDim, SymExpr)) for d in self.shape))

    def _is_concrete(self) -> bool:
        return self.rank() == 0 or (self.shape is not None and all(isinstance(d, int) for d in self.shape))

    def clone(self) -> 'Tensor':
        """ Fast shallow clone for per experiment isolation """
        new = object.__new__(Tensor)
        new.name      = self.name      
        new.dtype     = self.dtype     
        new.shape     = self.shape     
        new.op_in     = self.op_in     
        new.op_out    = self.op_out    
        new.is_param  = self.is_param  
        new.is_const  = self.is_const  
        new.data      = self.data      
        new.location  = self.location  
        return new


    def __str__(self):
        s = f"Tensor({self.name}) shape={self.shape}, dtype={self.dtype.name}, "
        s += f"is_param={self.is_param}, "
        s += f"is_const={self.is_const}, "
        s += f"is_view={self.is_view}, "
        s += f"op_in={self.op_in}, "
        s += f"op_out={self.op_out}, "
        if self.data is None:
            s += f"data={self.data}"
        elif self.rank() > 0 and self.nelems() > 5:
            s += "data=(...)"
        else:
            s += f"data={self.data}"
        return s


def make_tensor(**kwargs): return Tensor(**kwargs)
