

from itertools import count

import numpy as np

import src.front.functional as F

from ..bten.tensor import Tensor, make_tensor
from ..utils.sym import is_symbolic
from .module import get_active_module


class DynName:
    dynamic_op_counter = count(start=1, step=1)

    @classmethod
    def reset_counter(cls):
        cls.dynamic_op_counter = count(start=1, step=1)

    @classmethod
    def get(cls, m, o):
        opname = f"{m.name}.{o}.impl_{next(cls.dynamic_op_counter)}"
        if opname in m._op_hndls:
            raise RuntimeError(f"Implicit opname({opname}) created via DynName not unique!!")
        return opname

def torch2onnx_slice_plan(input_shape, slice_spec):
    """
    Analyze PyTorch/Numpy-style slice_spec for a tensor of shape input_shape,
    and return an ONNX execution plan as a dict:
        {
            'unsqueezes'  : [axis, ...],  # axes at which to insert new dims (None in slice_spec)
            'gathers'     : [(axis, index), ...],
            'slice'       : {'axes': [...], 'starts': [...], 'ends': [...], 'steps': [...]},
            'squeeze_axes': [...],    # always [0, 1, ...] after gathers, or None
            'output_shape': list,     # shape after all slicing and unsqueezing
        }
    Raises AssertionError or NotImplementedError for invalid or unsupported cases.
    """
    # Ensure list for easy manipulation
    slice_spec = list(slice_spec)
    ndim       = len(input_shape)

    # Check for multiple ellipsis
    if slice_spec.count(Ellipsis) > 1:
        raise AssertionError("Multiple ellipsis not allowed in slice_spec")

    # Expand ellipsis
    if Ellipsis in slice_spec:
        idx = slice_spec.index(Ellipsis)
        n_specified = len([s for s in slice_spec if s is not Ellipsis and s is not None])
        num_missing = ndim - n_specified
        expanded_spec = (slice_spec[:idx] + [slice(None)] * num_missing + slice_spec[idx+1:])
    else:
        expanded_spec = slice_spec[:]

    # After ellipsis expansion, insert None axes remain untouched
    # Now, pad with slice(None) if needed (excluding None axes)
    n_not_none = len([s for s in expanded_spec if s is not None])
    if n_not_none < ndim:
        expanded_spec += [slice(None)] * (ndim - n_not_none)

    # --- Check for too many non-None indices ---
    n_non_none = len([s for s in expanded_spec if s is not None])
    if n_non_none > ndim:
        raise AssertionError(
            f"Too many non-None indices for tensor: expected <= {ndim}, got {n_non_none}"
        )

    # Identify where None occurs (axes for Unsqueeze)
    unsqueeze_axes = [i for i, s in enumerate(expanded_spec) if s is None]

    # Remove None entries to get the actual slice/gather spec for data axes
    data_spec = [s for s in expanded_spec if s is not None]
    if len(data_spec) != ndim:
        raise RuntimeError(
                f"Internal error: after removing Nones, number of axes is {len(data_spec)}, expected {ndim}"
                )

    #F1: detect single tensor advanced indexing upfront so we can suppress trivial slice(None)
    #axes that would otherwise inflate the op_count (tokens[idx] must lower to exactly one
    # F.Gather, not Gather+Slice)
    n_tensor_idx = sum(1 for s in data_spec if isinstance(s, Tensor))
    if n_tensor_idx > 1:
        raise AssertionError(
                f"Multi-tensor advanced indexing is not supported "
                f"found {n_tensor_idx} tensor indices in slice_spec"
                )
    has_tensor_idx = n_tensor_idx == 1

    # Build gathers
    gathers = []
    working_shape = list(input_shape)
    axes_removed = 0
    for i, spec in enumerate(data_spec):
        if isinstance(spec, int):
            axis = i - axes_removed
            idx = spec if spec >= 0 else working_shape[axis] + spec
            #Upper bound check is skipped when the axis dim is symbolic --
            # SymExpr/SymDim do not support < and ONNX Gather runtime performs
            # its own bounds check anyway
            if 0 > idx  or (not is_symbolic(working_shape[axis]) and not is_symbolic(idx) and idx >= working_shape[axis]):
                raise AssertionError(
                    f"Integer index {idx} out of bounds for axis {axis} with dim {working_shape[axis]}"
                )
            gathers.append((axis, idx))
            working_shape.pop(axis)
            axes_removed += 1

    # Remaining (non-int) axes: each entry is either a slice or a Tensor index
    # We walk them in source order, tracking 'cur_axis' -- the axis position
    # in the FINAL output frame (i.e. after int-gathers AND after tensor-gather
    # rank expansion). The caller therefore executes the plan as:
    #  int-gathers (+ per-gather squeeze) -> tensor-gathers -> slice
    post_gather_spec = [spec for spec in data_spec if not isinstance(spec, int)]
    post_gather_shape = working_shape
    slice_axes = []
    starts = []
    ends = []
    steps = []
    tensor_gathers = [] #[(axis_in_final_frame, FrontTensor), ...]
    out_shape = []
    cur_axis = 0
    for src_axis, spec in enumerate(post_gather_spec):
        dim = post_gather_shape[src_axis]
        if isinstance(spec, slice):
            s = 0 if spec.start is None else (spec.start if spec.start >= 0 else dim + spec.start)
            e = dim if spec.stop is None else (spec.stop if spec.stop >= 0 else dim + spec.stop)
            step = 1 if spec.step is None else spec.step
            if step < 0:
                raise NotImplementedError("Negative steps are not supported by ONNX Slice (opset 13)")
            #trailing window slice (x[...,-W:]) on a SYMBOLIC axis.
            #folding the neg start to dim + start = dim - W yields as SymExpr that __getitem__
            # cannot materialze into an int64 initializer, and that shape inference cannot simplify
            # back to W. ONNX Slice natively supports literal neg starts (clamped at runtime), so keep
            # the literal -W start and an open-end sentinel (np.iinfo(int64)).max, which ONNX clamps.
            # and record the concrete window magnitude W as the planned output length on this axis.
            # Both initializer values stay concrete int64, Only the open-end trailing window is handled
            # here
            if (spec.start is not None and spec.start < 0 and spec.stop is None
                and step == 1 and is_symbolic(dim)):
                s = int(spec.start)               #literal neg start(-W)
                e = int(np.iinfo(np.int64).max)   #open-end sentinel: ONNX clamps
                length = -int(spec.start)         #concrete window magnitude W
                out_shape.append(length)
                slice_axes.append(cur_axis)
                starts.append(s)
                ends.append(e)
                steps.append(step)
                cur_axis += 1
                continue

            #The clipping below is a trace time defensive bound; ONNX Slice clips at runtime.
            #When dim or e is symbolic, the comparison cannot be evaluated -- skipp the clip
            # and trust the user's e
            if step > 0:
                if not is_symbolic(e) and not is_symbolic(dim):
                    e = min(e, dim)
            else:
                if not is_symbolic(e):
                    e = max(e, -1)
            #compute length for this axis
            length = (e - s + (step -1)) // step if step > 0 else 0
            if not is_symbolic(length):
                length = max(0, length)
            out_shape.append(length)
            #Supress trival slice (None) - equivalent axes when there is also a tensor index
            # pure slice paths keep the all-axes Slice behavior they have today (see test_trailing_dims).
            # Also suppress when the bound is symbolic: ONNX Slice requires int constants for
            # ends, which we cannot materialize from a SymExpr at trace time
            is_trivial = (s == 0 and e == dim and step == 1)
            suppress = (has_tensor_idx and is_trivial) or (is_trivial and is_symbolic(e))
            if not suppress:
                slice_axes.append(cur_axis)
                starts.append(s)
                ends.append(e)
                steps.append(step)
            cur_axis += 1
        elif isinstance(spec, Tensor):
            #F1: single-tensor advanced indexing in this axis. Lowers to one F.Gather(axis=cur_axis)
            # that replaces this data axis with the full shape of the index Tensor
            if spec.shape is None:
                raise AssertionError(f"Index tensor {spec.name} has no shape")
            idx_shape = list(spec.shape)
            tensor_gathers.append((cur_axis, spec))
            out_shape.extend(idx_shape)
            cur_axis += len(idx_shape)
        else:
            raise TypeError(f"Non-slice object found where slice expected {spec}")


    squeeze_axes = list(range(len(gathers))) if gathers else None

    # Compute final output shape, including unsqueezes
    final_shape = out_shape
    for axis in unsqueeze_axes:
        # Each unsqueeze is on the original index before any insertions,
        # so as we insert, subsequent indices shift.
        if axis < 0 or axis > len(final_shape):
            raise AssertionError(f"Invalid None position at {axis}")
        final_shape.insert(axis, 1)

    plan = {
        'unsqueezes': unsqueeze_axes if unsqueeze_axes else None,
        'gathers': gathers if gathers else None,
        'tensor_gathers': tensor_gathers if tensor_gathers else None,
        'slice': {
            'axes': slice_axes,
            'starts': starts,
            'ends': ends,
            'steps': steps,
        } if slice_axes else None,
        'squeeze_axes': squeeze_axes,
        'output_shape': final_shape,
    }
    return plan

def cat(Alist, dim=0):
    for i,x in  enumerate(Alist):
        if not isinstance(x, Tensor):
            raise TypeError(f"cat: input[{i}] = {x} not a Tensor!!")
    module = get_active_module()
    if module is None:
        raise RuntimeError("No active module context for cat op")

    op_name = DynName.get(module, 'cat')
    op = F.Concat(op_name, axis=dim)
    module._op_hndls[op.name] = op
    for x in Alist:
        if x.name not in module._tensors:
            module._tensors[x.name] = x
    result = op(*Alist)
    return result

def stack(Alist, dim=0):
    if not isinstance(Alist,list):
        raise TypeError(f"stack: input is not a list: {Alist}")

    for i,x in enumerate(Alist):
        if not isinstance(x, Tensor):
            raise TypeError(f"stack: input[{i}] = {x} not a Tensor!!")

    if len(Alist) == 0:
        raise ValueError(f"stack: empty input list: {Alist}")

    shape_check = all(t.shape == Alist[0].shape for t in Alist)

    if not shape_check:
        raise ValueError("All tensors must have the same shape to stack")

    # Task 048.5(c): dtype policy aligned with Phase 048.3 concat_sinf N-ary
    # promotion. stack() lowers to unsqueeze-per-input + concat, so the
    # resulting concat node already carries reduce(promote_types, inputs) via
    # concat_sinf. Previous strict-match ValueError removed for parity.

    base_rank = Alist[0].rank()
    if dim < 0:
        dim += base_rank + 1 #because a new axis will be added

    if dim < 0 or dim > base_rank:
        raise ValueError(f"dim {dim} is out of range for tensors of rank {base_rank}!!")

    module = get_active_module()
    if module is None:
        raise RuntimeError("No active module context for stack op")

    op_name = DynName.get(module, 'stack')

    axesData = np.array([dim], dtype=np.int64)
    axesTensor = make_tensor(
            name = op_name + '.unsqueeze.axes',
            shape = list(axesData.shape),
            data=axesData,
            is_const = True,
            dtype = 'int64',
            )
    if axesTensor.name not in module._tensors:
        module._tensors[axesTensor.name] = axesTensor

    outTensors = []
    for i,x in enumerate(Alist):
        if x.name not in module._tensors:
            module._tensors[x.name] = x

        op = F.Unsqueeze(op_name + f'.unsqueeze_{i}')
        module._op_hndls[op.name] = op

        ott = op(x, axesTensor)
        outTensors.append(ott)

    op = F.Concat(op_name + '.concat', axis=dim)
    module._op_hndls[op.name] = op
    final_result = op(*outTensors)

    return final_result

def topk(X, /, k,  **kwargs):
    # Backward-compatible forwarder. Prefer x.topk(k, ...) at call sites.
    # FrontTensor is imported lazily here to avoid a top-level cycle with
    # src/front/tensor.py (which imports from this module).
    from .tensor import FrontTensor
    if not isinstance(X, FrontTensor):
        raise TypeError(f"D.topk: input is not a FrontTensor: {type(X).__name__}")
    return X.topk(k, **kwargs)
