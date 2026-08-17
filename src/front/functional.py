
from ..bten.registry import get_op_registry
from ..bten.op import TensorOp
from ..bten.tensor import Tensor
from .tensor import FrontTensor

from functools import partial
import warnings

def get_active_module():
    from .module import get_active_module as _gam
    return _gam()

def _canonicalize_kwargs(optype, opinfo, kwargs):
    """
       module-private: resolve PyTorch aliases in-place on kwargs

       for each (alias, canonical) pair in opinfo.aliases:
         - if both present -> TypeError
         - if only PyTorch present -> rename to canonical
         - if only canonical ONNX present -> emit Deprecation/Warning and leave as is
    """
    aliases = opinfo.aliases
    for alias_key, canon_key in aliases.items():
        has_alias = alias_key in kwargs
        has_canon = canon_key in kwargs
        if has_alias and has_canon:
            raise TypeError(
                    f"F.{optype}: cannot pass both {canon_key!r} (ONNX-canonical) and "
                    f" {alias_key!r} (PyTorch alias); supply exactly one"
                    )
        if has_alias:
            kwargs[canon_key] = kwargs.pop(alias_key)
        elif has_canon:
            warnings.warn(
                    f"F.{optype}: kwarg {canon_key!r} (ONNX-canonical) is "
                    f"deprecated at the F.{optype}(...) call site; use the "
                    f"PyTorch kwarg {alias_key!r} instead. The canonical key "
                    f"remains {canon_key!r} on op.attrs (unchanged)",
                    DeprecationWarning,
                    stacklevel=3
                    )
    return kwargs

def get_op_attrs(optype, kwargs):
    opinfo = get_op_registry().get_op(optype)
    _canonicalize_kwargs(optype, opinfo, kwargs)
    attrs  = {}
    for k in opinfo.attrs:
        if k in kwargs:
            attrs[k] = kwargs[k]
            del kwargs[k]
    return attrs

class TensorOpHandle:
    """
    Helper type for easy constuction and usage of TensorOp
    Required: 1-1 Mapping between TensorOp -> TensorOpHandle

    Two Cases:
      1) for each param, we store the position in the input tensor list with the tensor via
         (pos, tensor) e.g. params = [(0, param_tensor0), (3, param_tensor1), (6, param_tensor2)]
      2) Variadic inputs, ipos[0] == (min, max) ==> valid input range

    then when we get the inputs in the __call__, we can create the extended input list with
    params at the correct positions

    IMPORTANT: Each TensorOpHandle instance is one-shot. Calling it multiple times on the same
    instance silently corrupts the operation graph (second invocation overwrites op attributes,
    adds duplicate edges). If you need multiple op instances, create a new handle
    """
    def __init__(self, name, optype, /, params, ipos, **kwargs):
        if len(ipos) < 1:
            raise ValueError("ipos should specify the input positions or variadic range")

        #validate that name is a str not a Tensor
        if isinstance(name, (Tensor, FrontTensor)):
            raise TypeError(
                    f"Expected 'name' to be a string, but got {type(name).__name__}. "
                    f"Did you forget to instantiate the operator? "
                    f"Use: F.{optype}('name') to create an instance, "
                    f"then call instance with tensor"
                    )

        if not isinstance(name, str):
            raise TypeError(
                    f"Expected 'name' to be a string, but got {type(name).__name__}. "
                    )

        self.name        = name
        self.attrs       = get_op_attrs(optype, kwargs)
        self.op          = TensorOp(self.name, optype=optype, attrs=self.attrs, **kwargs)
        self.params      = params
        self.ipos        = ipos
        self.itensors    = None
        self.num_out     = kwargs.get('num_outputs', 1)
        self.otensors    = [FrontTensor(f'{name}.out_{i}') for i in range(self.num_out)]

        if isinstance(self.ipos[0], int):
            self.is_varidic = False
            self.input_range = None
        elif isinstance(self.ipos[0], tuple) and len(self.ipos[0]) == 2 and all(isinstance(x, (int, float)) for x in self.ipos[0]):
            self.is_varidic = True
            self.input_range = self.ipos[0]
            lo, hi = self.input_range
            if lo >= hi:
                raise ValueError(f"Illegal input_range({self.input_range}) - {lo} >= {hi}!!")
        else:
            raise ValueError(f"Illegal ipos({self.ipos})!!!")

        self.implicit_inputs = []
        self._consumed = False

    def __call__(self, *xargs):
        """
           Invoke the op handle to trace an operation
        """
        if self._consumed:
            raise RuntimeError(
                    f"TensorOpHandle for op '{self.name}' (type={self.op.optype}) can only be invoked once per instance. "
                    "This guard prevents silent operation graph corruption"
                    )

        nargs = len(xargs)
        if self.is_varidic:
            if not (nargs >= self.input_range[0] and nargs < self.input_range[1]): #type: ignore[index]
                raise ValueError(f"Length for inputs {nargs} should be in range: {self.input_range}")
            self.itensors = [x for x in xargs]
        else:
            if nargs != len(self.ipos):
                raise ValueError(f"Length for inputs {nargs} & ipos {len(self.ipos)} don't match")
            all_itensors = self.params + list(zip(self.ipos, xargs))
            sorted_all_itensors = sorted(all_itensors, key=lambda v: v[0])
            self.itensors   = [x for _,x in sorted_all_itensors]

        self.op.inList  = [x.name for x in self.itensors]
        self.op.outList = [x.name for x in self.otensors]

        for x in self.itensors:
            x.op_in.append(self.op.name)

        for x in self.otensors:
            x.op_out.append(self.op.name)

        #forward the call to TensorOp
        self.op(self.itensors, self.otensors)

        module = get_active_module()
        if module is not None:
            for x in self.otensors:
                if x.name not in module._tensors:
                    module._tensors[x.name] = x

        #mark this handle as consumed after successful invocation
        self._consumed = True

        if len(self.otensors) == 1:
            return self.otensors[0]
        else:
            return (*self.otensors,)

def UniversalOperator(name, /, optype, params, ipos, **kwargs):
    return TensorOpHandle(name, optype, params=params, ipos=ipos, **kwargs)

#Unary Operators
UnaryOperator = partial(UniversalOperator, params=[], ipos=[0])
Softmax       = partial(UnaryOperator, optype='Softmax')
Transpose     = partial(UnaryOperator, optype='Transpose')
Relu          = partial(UnaryOperator, optype='Relu')
Gelu          = partial(UnaryOperator, optype='Gelu')
Sigmoid       = partial(UnaryOperator, optype='Sigmoid')
Split         = partial(UnaryOperator, optype='Split')

#Binary Operators
BinaryOperator = partial(UniversalOperator, params=[], ipos=[0,1])
Add            = partial(BinaryOperator, optype='Add')
Sub            = partial(BinaryOperator, optype='Sub')
Mul            = partial(BinaryOperator, optype='Mul')
Div            = partial(BinaryOperator, optype='Div')
Pow            = partial(BinaryOperator, optype='Pow')
MatMul         = partial(BinaryOperator, optype='MatMul')
Gather         = partial(BinaryOperator, optype='Gather')
Reshape        = partial(BinaryOperator, optype='Reshape')
Unsqueeze      = partial(BinaryOperator, optype='Unsqueeze')
Squeeze        = partial(BinaryOperator, optype='Squeeze')

#Variadic ops use ipos=[(lo, hi)] to specify accepted input count range
# float('inf') means unbounded upper limit
Concat    = partial(UniversalOperator, optype='Concat',    params=[], ipos=[(2, float("inf"))])
Trilu     = partial(UniversalOperator, optype='Trilu',     params=[], ipos=[(1, 2)])
Slice     = partial(UniversalOperator, optype='Slice',     params=[], ipos=[(3, 6)])
TopK      = partial(UniversalOperator, optype='TopK',      params=[], ipos=[(2, 3)], num_outputs=2)
ScatterND = partial(UniversalOperator, optype='ScatterND', params=[], ipos=[(3, 4)])
