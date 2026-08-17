
import threading

from ..bten.tensor import Tensor
from ..bten.op import TensorOp
from ..graph import WorkloadGraph
from .functional import TensorOpHandle, UniversalOperator, MatMul, Gather

from typing import Iterator, Optional

_trace_ctx = threading.local()

def get_active_module():
    """Returns the innermost Module whose __call__ is on the stack."""
    return getattr(_trace_ctx, 'current_module', None)

class Module:

    def __init__(self, name):
        self.name         = name
        self._tensors     = {}
        self._op_hndls    = {}
        self._submodules  = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._tensors[name] = value
        elif isinstance(value, TensorOpHandle):
            self._op_hndls[name] = value
            if hasattr(value, 'params') and len(value.params) > 0:
                for _,ptensor in value.params:
                    self._tensors[ptensor.name] = ptensor
        elif isinstance(value, Module):
            self._submodules[name] = value
        elif isinstance(value, ModuleList):
            for m in value:
                self._submodules[name + "." + m.name] = m
        else:
            pass
        super().__setattr__(name, value)

    def get_tensors(self, tbl):
        for k,v in self._tensors.items():
            tbl[v.name] = v
        for k,v in self._submodules.items():
            v.get_tensors(tbl)
        return tbl

    def get_ops(self, tbl: dict):
        for k,v in self._op_hndls.items():
            if v.op is not None:
                tbl[v.op.name] = v.op
        for k,v in self._submodules.items():
            v.get_ops(tbl)
        return tbl

    def get_forward_graph(self, *input_tensors):
        #Get Tensors
        ttbl = {}
        def _collect(obj, idx):
            # Accept a Tensor, a (possibly nested) list/tuple of Tensors, or
            # None (Task 047: the decode ``past_kv`` is a nested
            # ``(global_list, local_list)`` structure of placeholder tensors).
            if obj is None:
                return
            if isinstance(obj, Tensor):
                ttbl[obj.name] = obj
            elif isinstance(obj, (list, tuple)):
                for sub in obj:
                    _collect(sub, idx)
            else:
                raise TypeError(
                    f"input_tensor-{idx} should be an instance of "
                    f"(Tensor|List[Tensor]|Tuple[...])!!\n{obj}")
        for ti, t in enumerate(input_tensors):
            _collect(t, ti)

        self.get_tensors(ttbl)

        #Get Ops...
        otbl = {} #type: ignore
        self.get_ops(otbl)

        #Graph Construction...
        gg = WorkloadGraph(self.name)

        #Add Tensors to Graph...
        for _,tensor in ttbl.items():
            gg.add_tensor(tensor)

        #Add Ops to Graph...
        for _,op in otbl.items():
            gg.add_op(op)

        #Construct Graph
        gg.construct_graph()

        return gg

    def __str__(self, indent_width=0):
        indent0 = ' ' * indent_width * 4
        indent1 = ' ' * (indent_width+1) * 4
        indent2 = ' ' * (indent_width+2) * 4
        s = f"{indent0}MODULE: {self.name}\n"
        s += f"{indent0}TENSORS:\n"
        for k,v in self._tensors.items():
            s += f"{indent1}{k}:{v}\n"

        s += f"{indent0}OPS:\n"
        for k,v in self._op_hndls.items():
            s += f"{indent1}{k}:{v.op}\n"
            if hasattr(v, 'params') and len(v.params) > 0:
                s += f"{indent2}PARAMS:\n"
                for _,ptensor in v.params:
                    s += f"{indent2}{ptensor}\n"
            if len(v.implicit_inputs) > 0:
                s += f"{indent2}IMPLICIT_INPUTS:\n"
                for itensor in v.implicit_inputs:
                    s += f"{indent2}{itensor}\n"

        s += f"{indent0}SUBMODULES:\n"
        for k,v in self._submodules.items():
            s += f"{indent1}{k}:\n"
            s += v.__str__(indent_width+1)
        return s

    def __call__(self, *args, **kwargs):
        parent = getattr(_trace_ctx, 'current_module', None)
        _trace_ctx.current_module = self
        try:
            return self.forward(*args, **kwargs)
        finally:
            _trace_ctx.current_module = parent

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f'forward() not implemented for {self.__class__.__name__}::{self.name}'
        )

    def inputs(self):
        raise NotImplementedError(
            f'inputs() not implemented for {self.__class__.__name__}::{self.name}'
        )

class ModuleList:
    def __init__(self, modules):
        self._modules_in_list = {}

        if len(modules) <= 0:
            raise ValueError("Empty ModuleList at construction!!")

        for i, module in enumerate(modules):
            if module is None:
                raise ValueError("'None' module passed to ModuleList")
            if not isinstance(module, Module):
                raise TypeError(f"{module} is not a Module subclass")
            self._modules_in_list[str(i)] = module

        #check all module names in the list are unique...
        if len(self) != len(set(m.name for m in self._modules_in_list.values())):
            raise ValueError(f"Module Names in ModuleList are not unique : {[m.name for m in self._modules_in_list.values()]}!!")

    def __len__(self):
        return len(self._modules_in_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._modules_in_list[str(i)] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            idx = idx + len(self) if idx < 0 else idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f'out-of-bound-index: {idx}')
            return self._modules_in_list[str(idx)]
        else:
            raise TypeError(f'Invalid index Type: {type(idx)}')

    def __iter__(self) -> Iterator[Module]:
        for i in range(len(self)):
            yield self[i]

    #we want to make this immutable after construction...
    # so restricting setitem / delitem / append / insert / extend
    def __setitem__(self, idx, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __delitem__(self, idx):
        raise RuntimeError("ModuleList is immutable after construction")

    def append(self, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def extend(self, modules):
        raise RuntimeError("ModuleList is immutable after construction")

    def insert(self, index, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __call__(self, *x):
        raise RuntimeError("ModuleList is not Callable")

############## Concrete Modules ################
class Embedding(Module):
    def __init__(self, name, tbl_size, emb_dim, **kwargs):
        super().__init__(name)
        dtype = kwargs.get('dtype', 'bfloat16')
        self.emb_wt = Tensor(name + '.param', shape=[tbl_size, emb_dim], dtype=dtype, is_param=True)
        self.gather = Gather(name)

    def forward(self, x):
        return self.gather(self.emb_wt, x)

class Linear(Module):
    def __init__(self, name, in_features, out_features, dtype='bfloat16', bias=False):
        super().__init__(name)
        self.in_features : int = in_features
        self.out_features: int = out_features
        self.matmul      : TensorOp = MatMul(name +'.matmul')
        self.param       : Tensor = Tensor(name + '.param', shape=[in_features, out_features], is_param=True, dtype=dtype)
        self.bias        : Optional[Tensor] = Tensor(name + '.bias',  shape=[out_features], is_param=True, dtype=dtype) if bias else None

    def forward(self, x):
        Y = self.matmul(x, self.param)
        if self.bias:
            Y += self.bias
        return Y

class Dropout(Module):
    def __init__(self, name, prob=None, train_mode=None, /, **kwargs):
        super().__init__(name)
        dtype = kwargs.get('dtype', 'bfloat16')

        params_list = []
        if prob:
            params_list.append((1,
                    Tensor(name + '.ratio', shape=[], data=prob, dtype=dtype, is_const=True)
                                ))
        if train_mode:
            params_list.append((2,
                    Tensor(name + '.training_mode', shape=[], data=train_mode, dtype='bool',    is_const=True)
                                ))

        self.drop = UniversalOperator(name +'.op', 'Dropout',
                                      params=params_list,
                                      ipos=[0],
                                      **kwargs
                                      )
    def forward(self, x):
        return self.drop(x)

class LayerNorm(Module):
    def __init__(self, name, count, /, **kwargs):
        super().__init__(name)

        dtype = kwargs.get('dtype', 'bfloat16')
        self.scale = Tensor(name + '.scale', shape=[count], is_param=True, dtype=dtype)
        self.bias  = Tensor(name + '.bias', shape=[count], is_param=True, dtype=dtype)
        self.lnorm = UniversalOperator(name +'.op', 'LayerNormalization',
                                      params= [(1,self.scale), (2,self.bias)],
                                      ipos=[0],
                                      **kwargs
                                      )
    def forward(self, x):
        """
        Note:
         ONNX LayerNorm can generate upto 3 outputs, but we are only generating 1
         Ok for now, because simple LLMs behave the same way...
        """
        return self.lnorm(x)

