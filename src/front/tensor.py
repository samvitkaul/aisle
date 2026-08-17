
from __future__ import annotations

import numpy as np

from ..bten.tensor import Tensor, make_tensor
from ..utils.sym import is_symbolic


def _alloc_dyn_op(module_tensor, op_kind, op_factory,
                  const_inputs=(), extra_inputs=()):
    """
       Allocate a per-call dynamic op handle and wire it into the active module

       Centeralizes the boilerplate shared by FrontTensor's dynamic allocation
       methods (topk, softmax, squeeze, etc.) active-module lookup, DynName-based
       op-naming, op-registration, synthesizing const-inputs, and self/input registration

       Args:
         module_tensor: the FrontTensor whose method is being invoked
         op_kind:       short string pased to DynName.get (e.g. topk, softmax, etc.)
         op_factory:    callable taking op_name -> constructed F.<Op> handle
         const_inputs:  iterable of (suffix, data_np_array, dtype_str) tuples;
                        a const tensor named op_name+suffix is synthesized,
                        registered in module._tensors (once), and appended.
         extra_inputs:  iterable of already constructed FrontTensors to register
                        (if not present) and append to the call inputs

       Returns:
         The result of op(self, *const_tensors, *extra_inputs) - a single FrontTensor
         or a tuple, depending on the op
    """

    from .dynamic import DynName
    from .module import get_active_module

    module = get_active_module()
    if module is None:
        raise RuntimeError(
                f"No active module context for dynamic op on {module_tensor.name}"
                )

    op_name = DynName.get(module, op_kind)
    op = op_factory(op_name)
    module._op_hndls[op.name] = op

    call_inputs = []
    for suffix, data, dtype in const_inputs:
        ct = make_tensor(
                name=op_name + suffix,
                shape=list(data.shape),
                data=data,
                is_const=True,
                dtype=dtype
                )
        if ct.name not in module._tensors:
            module._tensors[ct.name] = ct
        call_inputs.append(ct)

    for x in extra_inputs:
        if x.name not in module._tensors:
            module._tensors[x.name] = x
        call_inputs.append(x)

    if module_tensor.name not in module._tensors:
        module._tensors[module_tensor.name] = module_tensor

    return op(module_tensor, *call_inputs)

class FrontTensor(Tensor):
    """
       Frontend tensor with PyTroch like operator overloads

       Extends the IR level Tensor with methods that create graph nodes
       during tracing (view, transpose, __add__, etc.)
    """

    # --- View/Reshape ---
    def view(self, *shape):
        import src.front.functional as F

        from .dynamic import DynName
        from .module import get_active_module

        dims: list = list(shape)
        orig_numel = self.nelems()

        #Handle a singel shape passed as a tuple/list
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = list(dims[0])

        #Handle -1 for one of the dims
        infer_idx = None
        known = 1
        for i, d in enumerate(dims):
            if d == -1:
                if infer_idx is not None:
                    raise ValueError("Only one dimension can be inferred (-1_")
                infer_idx = i
            else:
                known *= d

        if infer_idx is not None:
            if is_symbolic(orig_numel) or is_symbolic(known):
                #defer inference: leave as symbolic expression
                dims[infer_idx] = orig_numel // known
            else:
                if known == 0:
                    raise ValueError("Known product of shape dims is zero")
                if orig_numel % known != 0:
                    raise ValueError("Shape is not compatible for view (cannot infer dimension)")
                dims[infer_idx] = orig_numel // known

        #Check total elems match (skip if symbolic)
        new_numel = 1
        for d in dims:
            new_numel *= d
        if not is_symbolic(new_numel) and not is_symbolic(orig_numel) and new_numel != orig_numel:
                raise ValueError("Shape is not compatible for view (cannot infer dimension)")

        module = get_active_module()
        if module is None:
            raise RuntimeError(f"No active module context for dynamic op on {self.name}")
        op_name = DynName.get(module, 'view')
        op = F.Reshape(op_name)
        module._op_hndls[op.name] = op
        shapeTensor = Tensor(op_name + '.fixshape', is_const=True, data=dims, dtype='int64')
        shapeTensor.shape = [len(dims)]
        for x in [self, shapeTensor]:
            if x.name not in module._tensors:
                module._tensors[x.name] = x
        return op(self, shapeTensor)

    def reshape(self, *shape):
        return self.view(*shape)

    # --- Size ---
    def size(self, dim=None):
        """
           torch.Tensor.size-like accessor over self.shape

           size() returns the full shape
           size(d) returns length of dimension d; Neg d is normalized
           Symbolic axis lengths pass through unchanged (no int coercion)

           Note: This is tensor-dim length, distinct from DTensor.size(mesh_dim=...)
           which queries device-mesh size
        """
        shape = self.shape
        if dim is None:
            return shape
        nd = len(shape) #type: ignore[arg-type]
        if dim < 0: dim += nd
        if dim < 0 or dim >= nd:
            raise IndexError(
                    f"Dimension out of range (expected to be in range of"
                    f" [{-nd}, {nd-1}], but got {dim}"
                    )
        return shape[dim] #type: ignore[index]

    # --- TopK ---
    def topk(self, k, **kwargs):
        import src.front.functional as F
        k_data = np.array([k], dtype=np.int64)
        return _alloc_dyn_op(
                self, 'topk',
                lambda name: F.TopK(name, **kwargs),
                const_inputs=[('.k', k_data, 'int64')],
                )

    # --- Softmax ---
    def softmax(self, dim=-1, **kwargs):
        import src.front.functional as F
        #Only inject default dim when caller hasn't already supplied the ONNX-canonical axis kwarg
        # F.Softmax aliases dim -> axis and would otherwise raise on duplicate keys
        if 'axis' not in kwargs:
            kwargs['axis'] = dim
        return _alloc_dyn_op(
                self, 'softmax',
                lambda name: F.Softmax(name, **kwargs),
                )

    # --- Unsqueeze ---
    def unsqueeze(self, dim=None, axes=None, **kwargs):
        import src.front.functional as F
        if dim is not None and axes is not None:
            raise ValueError("unsqueeze: pass exactly one of dim= or axes=, not both")
        if dim is None and axes is None:
            raise ValueError("unsqueeze: must specify dim= or axes=")
        if dim is not None:
            axes = [dim]
        axes_data = np.array(axes, dtype=np.int64)
        return _alloc_dyn_op(
                self, 'unsqueeze',
                lambda name: F.Unsqueeze(name, **kwargs),
                const_inputs=[('.axes', axes_data, 'int64')],
                )

    # --- Squeeze ---
    def squeeze(self, dim=None, axes=None, **kwargs):
        import src.front.functional as F
        if dim is not None and axes is not None:
            raise ValueError("squeeze: pass exactly one of dim= or axes=, not both")
        if dim is None and axes is None:
            raise ValueError("squeeze: must specify dim= or axes=")
        if dim is not None:
            axes = [dim]
        axes_data = np.array(axes, dtype=np.int64)
        return _alloc_dyn_op(
                self, 'squeeze',
                lambda name: F.Squeeze(name, **kwargs),
                const_inputs=[('.axes', axes_data, 'int64')],
                )

    # --- Transpose ---
    def transpose(self, dim0, dim1):
        import src.front.functional as F

        if self.rank() < 1:
            raise ValueError("Tensor rank must be at least 1")

        #Handle neg indices
        if dim0 < 0: dim0 = self.rank() + dim0
        if dim1 < 0: dim1 = self.rank() + dim1

        #Validate dims
        if dim0 < 0 or dim0 >= self.rank():
            raise ValueError(f"dim0 ({dim0}) is out of bounds for tensor rank {self.rank()}")
        if dim1 < 0 or dim1 >= self.rank():
            raise ValueError(f"dim1 ({dim1}) is out of bounds for tensor rank {self.rank()}")

        #Create permutation [0, 1, 2, ..., tensor_rank-1]
        perm = list(range(self.rank()))

        #swap dim0, dim1
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]

        return _alloc_dyn_op(
                self, 'transpose',
                lambda name: F.Transpose(name, perm=perm),
                )

    # --- Binary operator overloads ---
    def __add__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('Add', F.Add, other)

    def __sub__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('Sub', F.Sub, other)

    def __mul__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('Mul', F.Mul, other)

    def __truediv__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('Div', F.Div, other)

    def __pow__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('Pow', F.Pow, other)

    def __matmul__(self, other):
        import src.front.functional as F
        return self._dispatch_binary('matmul', F.MatMul, other)

    # --- Reflected Binary operator overloads ---
    def __radd__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('Add', F.Add, other)

    def __rsub__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('Sub', F.Sub, other)

    def __rmul__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('Mul', F.Mul, other)

    def __rtruediv__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('Div', F.Div, other)

    def __rpow__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('Pow', F.Pow, other)

    def __rmatmul__(self, other):
        import src.front.functional as F
        return self._dispatch_binary_r('matmul', F.MatMul, other)

    # --- Indexing ---
    def __getitem__(self, idx):
        import src.front.functional as F

        from .dynamic import DynName, torch2onnx_slice_plan
        from .module import get_active_module

        # Normalize scalar index to a tuple so torch2onnx_slice_plan can list() it
        if not isinstance(idx, tuple):
            idx = (idx,)

        plan = torch2onnx_slice_plan(self.shape, idx)

        module = get_active_module()
        if module is None:
            raise RuntimeError(f"No active module context for dynamic op on {self.name}")
        op_name = DynName.get(module, 'slice')
        op_sub_num = 0

        X = self

        # 1. Unsqueeze for new axes (from None in slice_spec)
        if plan['unsqueezes']:
            op_name = f"{op_name}.{op_sub_num}"
            op = F.Unsqueeze(op_name)
            module._op_hndls[op.name] = op
            op_sub_num += 1

            unsq_axes_data = np.array(plan['unsqueezes'], dtype=np.int64)
            unsq_axes = make_tensor(
                name=op_name + '.axes',
                shape=list(unsq_axes_data.shape),
                data=unsq_axes_data,
                is_const=True,
                dtype='int64',
            )
            if unsq_axes.name not in module._tensors:
                module._tensors[unsq_axes.name] = unsq_axes

            X = op(X, unsq_axes)

        # 2. Gather+Squeeze for integer indices
        if plan['gathers']:
            for i, (axis, idx_val) in enumerate(plan['gathers']):
                if not (0 <= axis < X.rank()):
                    raise IndexError(f"Gather axis {axis} out of bounds for {X.shape}")
                if not (0 <= idx_val < X.shape[axis]):  # type: ignore[index]
                    raise IndexError(
                        f"Gather idx {idx_val} out of bounds for axis {axis} (shape={X.shape})"
                    )
                op_name = f"{op_name}.{op_sub_num}"
                op = F.Gather(op_name, axis=axis)
                module._op_hndls[op.name] = op
                op_sub_num += 1
                idx_tensor_data = np.array([idx_val], dtype=np.int64)
                idx_tensor = make_tensor(
                    name=op.name + '.idx',
                    shape=list(idx_tensor_data.shape),
                    data=idx_tensor_data,
                    is_const=True,
                    dtype='int64',
                )
                module._tensors[idx_tensor.name] = idx_tensor
                X = op(X, idx_tensor)

                # Squeeze the gather axis (Gather with a 1-D length-1 index
                # inserts a singleton at `axis`, not at 0 -- the previous
                # squeeze-[0] worked only by coincidence with axis-0 gathers).
                op_name = f"{op_name}.{op_sub_num}"
                op = F.Squeeze(op_name)
                module._op_hndls[op.name] = op
                op_sub_num += 1
                axes_tensor_data = np.array([axis], dtype=np.int64)
                axes_tensor = make_tensor(
                    name=op.name + '.axes',
                    shape=list(axes_tensor_data.shape),
                    data=axes_tensor_data,
                    is_const=True,
                    dtype='int64',
                )
                module._tensors[axes_tensor.name] = axes_tensor
                X = op(X, axes_tensor)

        # 2b. Tensor-index gathers (F1). Each entry replaces one axis of X
        # with the full shape of the index tensor (rank-extending). Process
        # in axis-decreasing order so earlier (higher-axis) Gathers do not
        # invalidate the axis positions of later (lower-axis) Gathers. Note:
        # the current plan emits at most one tensor-gather (multi-tensor
        # advanced indexing is rejected up-front in torch2onnx_slice_plan).
        if plan.get('tensor_gathers'):
            for i, (axis, idx_tensor) in enumerate(
                    sorted(plan['tensor_gathers'], key=lambda kv: -kv[0])):
                op_name = f"{op_name}.tgather_{i}"
                op = F.Gather(op_name, axis=axis)
                module._op_hndls[op.name] = op
                op_sub_num += 1
                if idx_tensor.name not in module._tensors:
                    module._tensors[idx_tensor.name] = idx_tensor
                X = op(X, idx_tensor)

        # 3. Slice (if any)
        if plan['slice']:
            op_name = f"{op_name}.{op_sub_num}"
            op = F.Slice(op_name, out_shape=plan['output_shape'])
            module._op_hndls[op.name] = op
            op_sub_num += 1

            s = plan['slice']
            starts_init = make_tensor(name=f'{op_name}.starts', data=np.array(s['starts'], dtype=np.int64), dtype='int64', is_const=True)
            ends_init = make_tensor(name=f'{op_name}.ends', data=np.array(s['ends'], dtype=np.int64), dtype='int64', is_const=True)
            axes_init = make_tensor(name=f'{op_name}.axes', data=np.array(s['axes'], dtype=np.int64), dtype='int64', is_const=True)
            steps_init = make_tensor(name=f'{op_name}.steps', data=np.array(s['steps'], dtype=np.int64), dtype='int64', is_const=True)
            starts_init.shape = list(starts_init.data.shape)
            ends_init.shape = list(ends_init.data.shape)
            axes_init.shape = list(axes_init.data.shape)
            steps_init.shape = list(steps_init.data.shape)
            for t in [starts_init, ends_init, axes_init, steps_init]:
                if t.name not in module._tensors:
                    module._tensors[t.name] = t

            X = op(X, starts_init, ends_init, axes_init, steps_init)

        return X


    # --- Private dispatch helper ---
    def _dispatch_binary(self, optype, tensor_op_hndl, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"_binary_op {optype} arg= {other} not a Tensor!!")

        return _alloc_dyn_op(
                self, optype,
                lambda name: tensor_op_hndl(name),
                extra_inputs=[other],
                )

    def _dispatch_binary_r(self, optype, tensor_op_hndl, other):
        """Reflected binary op: other <op> self (other is LHS)"""
        from .dynamic import DynName
        from .module import get_active_module

        if not isinstance(other, Tensor):
            raise TypeError(f"_binary_op {optype} arg= {other} not a Tensor!!")

        module = get_active_module()
        if module is None:
            raise RuntimeError(f"No active module context for dynamic op on {self.name}")

        op_name = DynName.get(module, optype)
        op = tensor_op_hndl(op_name)
        module._op_hndls[op.name] = op

        for x in (other, self):
            if x.name not in module._tensors:
                module._tensors[x.name] = x
        return op(other, self)

def make_front_tensor(**kwargs) -> FrontTensor:
    """Create a FrontTensor - use in frontend/test contexts"""
    return FrontTensor(**kwargs)
