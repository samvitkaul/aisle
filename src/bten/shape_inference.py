
from functools import reduce

import numpy as np

from ..utils.common import prod_ints
from ..utils.data_types import DataType, dt2np, promote_types
from ..utils.sym import is_symbolic
from .tensor import Tensor, make_tensor


def fill_data(tensor: Tensor):
    if tensor.rank() == 0:
        if tensor.dtype == DataType.FLOAT32:
            _data = 1.0
        else:
            assert False, "Only float32 rank-0 tensor clones supported right now!!!"
    else:
        if not tensor.is_concrete():
            return None
        _data = np.random.randn(*(tensor.shape)).astype(dt2np(tensor.dtype)) #type: ignore
        _data = _data.tolist() #type: ignore
    return _data

def clone_by_shape_n_fill(tensor, /, data_maybe_missing = True):
    if not tensor.check_shape():
        raise ValueError(f"Illegal Shape in Tensor {tensor}")

    if data_maybe_missing:
        if tensor.data is None:
            clone = Tensor(name=tensor.name + '.clone', shape=tensor.shape, dtype=tensor.dtype.name, data=fill_data(tensor), op_in=tensor.op_in, op_out=tensor.op_out)
        else:
            clone = tensor
    else:
        if tensor.data is None:
            raise ValueError(f"Missing Data in Tensor {tensor}")
        clone = tensor
    return clone

def unary_fwd(iTList, oTList, op, **kwargs):
    X,Y = iTList[0], oTList[0]
    if not X.check_shape():
        raise ValueError(f"Input tensor shape not defined: {X}")
    Y.shape = X.shape
    Y.dtype = X.dtype

def softmax_sinf(iTList, oTList, op, **kwargs):
    X, Y = iTList[0], oTList[0]
    if not X.check_shape():
        raise ValueError(f"Input tensor shape not defined: {X}")

    axis = op.attrs.get('axis', -1)
    if not isinstance(axis, int):
        raise TypeError(f"Softmax attribute 'axis' must be int; got {type(axis).__name__}={axis!r}")

    XRank = X.rank()
    if XRank < 1:
        raise ValueError(f"Softmax requires input rank >= 1; got tank {XRank} for {X}")

    norm_axis = axis + XRank if axis < 0 else axis
    if not (0 <= norm_axis < XRank):
        raise ValueError(f"Softmax axis {axis} out of range for input rank {XRank} ({X})")

    #persist the normalized axis on the node so downstream passes can use it
    op.attrs['axis'] = norm_axis

    Y.shape = X.shape
    Y.dtype = X.dtype

def gelu_sinf(iTList, oTList, op, **kwargs):
    X, Y = iTList[0], oTList[0]
    if not X.check_shape():
        raise ValueError(f"Input tensor shape not defined: {X}")

    approximate = op.attrs.get('approximate', 'none')
    if not isinstance(approximate, str):
        raise TypeError(f"Gelu attribute 'approximate' must be str; got {type(approximate).__name__}={approximate!r}")

    if approximate not in ('tanh', 'none'):
        raise ValueError("Gelu attribute 'approximate' must be (none|tanh)!!")

    #persist the canonical value on the node so downstream passes can use it
    op.attrs['approximate'] = approximate

    Y.shape = X.shape
    Y.dtype = X.dtype

def bidirectional_broadcast_shape_inference(shape1, shape2):
    max_len = max(len(shape1), len(shape2))
    padded1 = list(shape1[::-1]) + [1] * (max_len - len(shape1))
    padded2 = list(shape2[::-1]) + [1] * (max_len - len(shape2))
    result  = []
    for d1, d2 in zip(padded1, padded2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        elif is_symbolic(d1) or is_symbolic(d2):
            #assume compatible when either side is symbolic
            result.append(d1 if is_symbolic(d1) else d2)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcast-compatible")
    return result[::-1]

def bidir_bcast(iTList, oTList, op, **kwargs):
    X0, X1, Y = iTList[0], iTList[1], oTList[0]
    assert X0.check_shape(), f"Input tensor-0 shape not defined: {X0}"
    assert X1.check_shape(), f"Input tensor-1 shape not defined: {X1}"
    Y.shape = bidirectional_broadcast_shape_inference(X0.shape, X1.shape)
    Y.dtype = promote_types(X0.dtype, X1.dtype)

def matmul_sinf(iTList, oTList, op, **kwargs):
    A, B = iTList[0], iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"

    AShape, BShape, CShape = A.shape, B.shape, None
    if len(AShape) < 1 or len(BShape) < 1:
        raise ValueError("Shapes must have at least 1 dimension")

    # Handle 1D cases
    if len(AShape) == 1 and len(BShape) == 1:
        #Vector-Vector: [n] x [n] -> [] (scalar result)
        if AShape[0] != BShape[0] and not (is_symbolic(AShape[0]) or is_symbolic(BShape[0])):
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[0]}")
        CShape = [] # Scalar result
        _reduced_dim = AShape[0]
    elif len(AShape) == 1:
        #Vector-Matrix: [n] x [..., n, m] -> [..., m]
        if AShape[0] != BShape[-2] and not(is_symbolic(AShape[0]) or is_symbolic(BShape[-2])):
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[-2]}")
        CShape = BShape[:-2] + [BShape[-1]]
        _reduced_dim = AShape[0]
    elif len(BShape) == 1:
        #Matrix-Vector: [..., m, n] x [n] -> [..., m]
        if AShape[-1] != BShape[0] and not(is_symbolic(AShape[-1]) or is_symbolic(BShape[0])):
            raise ValueError(f"Matmul incompatible: {AShape[-1]} != {BShape[0]}")
        CShape = AShape[:-1]
        _reduced_dim = AShape[-1] #The last dim of matrix
    else:
        # Handle 2D+ cases
        batch1, mat1 = AShape[:-2], AShape[-2:]
        batch2, mat2 = BShape[:-2], BShape[-2:]

        # Check matrix multiplication compatibility
        if mat1[-1] != mat2[-2] and not(is_symbolic(mat1[-1]) or is_symbolic(mat2[-2])):
            raise ValueError(f"Matmul incompatible: {mat1[-1]} != {mat2[-2]}")
        # Broadcast batch dims and compute output shape
        broadcast_batch = bidirectional_broadcast_shape_inference(batch1, batch2)
        CShape = broadcast_batch + [mat1[0], mat2[-1]]
        _reduced_dim = mat1[-1] #inner dim

    oTList[0].shape = CShape
    oTList[0].dtype = promote_types(A.dtype, B.dtype)

def gather_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs.get('axis', 0)
    assert isinstance(axis, int), f"attribute axis ({axis}) is not an int!!"

    dataT    = iTList[0]
    indicesT = iTList[1]
    assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
    assert indicesT.check_shape(), f"Illegal input indicesT shape: {indicesT}!!"

    data_rank  = dataT.rank()
    data_shape = dataT.shape
    axis = axis if axis >= 0 else data_rank + axis
    assert axis >= 0 and axis < data_rank, f"Axis {axis} is out of bounds for dataT.shape {dataT.shape}"
    oTList[0].shape = data_shape[:axis] + indicesT.shape + data_shape[axis + 1:]
    oTList[0].dtype = dataT.dtype

def scatternd_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ScatterND (ONNX opset >= 13)
    """

    if len(iTList) != 3:
        raise ValueError(f"ScatterND expects 3 inputs, got {len(iTList)}")

    dataT    = iTList[0]
    indicesT = iTList[1]
    updatesT = iTList[2]
    assert dataT.check_shape(),    f"ScatterND: bad data shape: {dataT}!!"
    assert indicesT.check_shape(), f"ScatterND: bad indices shape: {indicesT}!!"
    assert updatesT.check_shape(), f"ScatterND: bad updates shape: {updatesT}!!"

    assert indicesT.dtype == DataType.INT64, f"ScatterND: indices dtype not int64: {indicesT.dtype}!!"
    assert updatesT.dtype == dataT.dtype, "ScatterND: requires updates.dtype == data.dtype; implicit promotion not spec-legal"

    data_rank = dataT.rank()
    assert data_rank >= 1, "P1"
    assert indicesT.rank() >= 1, "P2"

    k = indicesT.shape[-1]
    assert isinstance(k, int), "P3"
    assert (1 <= k <= data_rank), "P4"

    expected_updates_shape = list(indicesT.shape[:-1]) + list(dataT.shape[k:])
    assert len(updatesT.shape) == len(expected_updates_shape), "P5"

    for d_actual, d_expected in zip(updatesT.shape, expected_updates_shape):
        if d_actual == d_expected:
            continue
        if is_symbolic(d_actual) or is_symbolic(d_expected):
            continue
        raise ValueError("P5")

    oTList[0].shape = list(dataT.shape)
    oTList[0].dtype = dataT.dtype


def ln_sinf(iTList, oTList, op, **kwargs):
    _axis       = op.attrs.get('axis', -1)
    _epsilon    = op.attrs.get('epsilon', 1e-5)
    _stash_type = op.attrs.get('stash_type', 1)

    X       = iTList[0]
    _scaleT = iTList[1]
    _biasT  = iTList[2] if len(iTList) == 3 else None
    assert X.check_shape(), f"Illegal Shape for {X}"
    XShape = X.shape
    XRank  = X.rank()

    if _axis < 0: _axis += XRank
    unsqueezed_rank = XRank - _axis
    reduction_shape = XShape[0:_axis] + [1] * unsqueezed_rank

    oTList[0].shape = X.shape
    inputs_for_promo = [X.dtype, _scaleT.dtype]
    if _biasT is not None:
        inputs_for_promo.append(_biasT.dtype)
    oTList[0].dtype = reduce(promote_types, inputs_for_promo)

    if len(oTList) >= 2:
        #Mean/InvStdDev dtype should track op.attrs[`stash_type`]
        # per ONNX spec (opset >= 17). TODO
        oTList[1].shape = reduction_shape
        oTList[1].dtype = X.dtype

    if len(oTList) == 3:
        # reshape needed because of initial tensor-to-matrix reshape in Step-1.
        oTList[2].shape = reduction_shape
        oTList[2].dtype = X.dtype



def split_sinf(iTList, oTList, op, **kwargs):
    num_outputs = op.attrs.get('num_outputs', len(oTList))
    axis        = op.attrs.get('axis',0)

    A      = iTList[0]
    splitT = iTList[1] if len(iTList) == 2 else None
    assert A.check_shape(), "Illegal shape!!"

    if axis < 0: axis = A.rank() + axis
    assert axis in range(A.rank()), f"Split Shape Inference: axis={axis} should be in [0,{A.rank()})"


    if splitT is None or splitT.data is None:
        split_dim = A.shape[axis] // num_outputs
        split = [split_dim for i in range(num_outputs)]
    else:
        split = splitT.data
    assert len(split) == num_outputs, f"split mismatch len( {split} ) != {num_outputs}"

    outShapes = []
    for tout_idx in range(num_outputs):
        tout_shape = A.shape.copy()
        tout_shape[axis] = split[tout_idx]
        outShapes.append(tout_shape)

    for tidx, tout in enumerate(oTList):
        tshape0 = outShapes[tidx]
        tout.shape = tshape0
        tout.dtype = A.dtype


def transpose_sinf(iTList, oTList, op, **kwargs):
    perms  = op.attrs['perm']
    assert len(perms) == iTList[0].rank(), \
            f"perms({perms}) must be equal to input rank ({iTList[0].rank()})!!"
    oTList[0].shape = [iTList[0].shape[i] for i in perms]
    oTList[0].dtype = iTList[0].dtype

def reshape_sinf(iTList, oTList, op, **kwargs):
    allowzero = op.attrs.get('allowzero', 0)

    B = clone_by_shape_n_fill(iTList[1],data_maybe_missing=False) #B.data should exist

    assert B.dtype == DataType.INT64, f"Input Data-Type should be np.int64 {B}"
    assert iTList[0].check_shape(), f"Illegal Input Shape: {iTList[0].shape}"
    input_shape  = iTList[0].shape
    input_size   = iTList[0].nelems()
    target_shape = B.data

    minus_one_count = 0
    minus_one_index = None
    zeros_count     = 0
    zeros_index     = []
    for i,x in enumerate(target_shape):
        if x == -1:
            minus_one_count += 1
            minus_one_index = i
        elif x == 0:
            zeros_count += 1
            zeros_index.append(i)
        else:
            pass
    assert minus_one_count <= 1, f"Only one -1 is allowed in target shape {target_shape}"

    if allowzero == 1 and minus_one_count == 1 and zeros_count > 0:
        assert False, f"Cannot have -1 and zeros simultaneously with allowzero in target_shape({target_shape})"

    #copy dims from input_shape, if required
    output_shape = [x for x in target_shape]
    if allowzero == 0:
        for idx in zeros_index:
            assert idx < len(input_shape), f"Illegal index({idx}) for input_shape({input_shape}) with allowzero=0"
            output_shape[idx] = input_shape[idx]

    # Handle -1 inference
    if minus_one_count == 1:
        output_size = prod_ints([x for x in output_shape if x != -1])
        if not is_symbolic(input_size) and not is_symbolic(output_size):
            assert input_size >= output_size and input_size % output_size == 0, \
                f"Cannot infer -1: input size {input_size}/{output_size}"
        inferred_dim = input_size // output_size
        output_shape[minus_one_index] = inferred_dim #type: ignore

    # Final validation
    final_output_size = prod_ints(output_shape)
    if not is_symbolic(input_size) and not is_symbolic(final_output_size):
        assert input_size  == final_output_size, \
            f"in({input_size}) & out({final_output_size}) sizes are not equal!!"

    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype


def topk_sinf(iTList, oTList, op, **kwargs):
    if len(iTList) != 2:
        raise ValueError(f"topk_sinf error: needs 2 input tensors, got {len(iTList)} instead!!")

    X = iTList[0]
    K = clone_by_shape_n_fill(iTList[1])

    if not X.check_shape(): raise ValueError(f"Input tensor-X shape not defined: {X}")
    if not K.check_shape(): raise ValueError(f"Input tensor-K shape not defined: {K}")
    if K.dtype != DataType.INT64:
        raise ValueError(f"input tensor-K DataType should be INT64: {K}")

    XRank  = X.rank()
    _axis   : int = op.attrs.get('axis',   -1)
    _largest: int = op.attrs.get('largest', 1)
    _sorted : int = op.attrs.get('sorted',  1)

    if XRank < 1:
        raise ValueError("TopK expects input of rank >= 1: {X}")

    if _axis < 0:
        _axis = XRank + _axis

    if _axis < 0 or _axis >= XRank:
        raise ValueError(f"Axis {_axis} is out of bounds for X {X}")

    outshape = X.shape.copy()
    d_axis   = X.shape[_axis]
    k_value  = [x.item() for x in K.data]
    if len(k_value) != 1:
        raise ValueError("TopK expects K-tensor should be 1D with single scalar value")
    k_scalar_value = k_value[0]

    if k_scalar_value < 0:
        raise ValueError(f"TopK requires K value > 0: {k_scalar_value}")
    if k_scalar_value > d_axis:
        raise ValueError(f"TopK requires K value({k_scalar_value}) < dim(axis) = {d_axis}")
    outshape[_axis] = k_scalar_value

    oTList[0].shape = outshape
    oTList[1].shape = outshape
    oTList[0].dtype = X.dtype
    oTList[1].dtype = DataType.INT64



def argmax_sinf(iTList, oTList, op, **kwargs):
    keepdims = op.attrs.get('keepdims', 1)
    axis     = op.attrs.get('axis',     0)
    dataT    = iTList[0] #clone_by_shape()
    rank     = dataT.rank()
    if axis < 0: axis += rank
    assert (0 <= axis < rank), f"Arg axis {axis} out of range for rank {rank}"

    if keepdims:
        outShape = [i for i in dataT.shape]
        outShape[axis] = 1
    else:
        outShape = [d for i,d in enumerate(dataT.shape) if i != axis]

    oTList[0].shape = outShape
    oTList[0].dtype = DataType.INT64

def reduce_sinf(iTList, oTList, op, **kwargs):
    keepdims = op.attrs.get('keepdims', 1)
    noop     = op.attrs.get('noop_with_empty_axes', 0)
    dataT    = iTList[0]#clone_by_shape()
    axesT    = clone_by_shape_n_fill(iTList[1], data_maybe_missing=False) if len(iTList) == 2 else None
    rank     = dataT.rank()
    outShape = list(dataT.shape)
    if axesT is None:
        if noop:
            outShape = dataT.shape
            reduce_axes = None
        else:
            reduce_axes = [i for i in range(rank)]
    else:
        reduce_axes = [i for i in axesT.data]

    if reduce_axes:
        normalized_axes = []
        for a in reduce_axes:
            if a < 0: a += rank
            assert (0 <= a < rank), f"reduce axis {a} out of range for rank {rank}"
            normalized_axes.append(a)
        axes_set = sorted(set(normalized_axes))

        if keepdims:
            outShape = [1 if i in axes_set else d for i,d in enumerate(dataT.shape)]
        else:
            outShape = [d for i,d in enumerate(dataT.shape) if i not in axes_set]

    oTList[0].shape = outShape
    oTList[0].dtype = dataT.dtype


def slice_sinf(iTList, oTList, op, **kwargs):
    dataT   = iTList[0]
    startsT = clone_by_shape_n_fill(iTList[1], data_maybe_missing=False)
    endsT   = clone_by_shape_n_fill(iTList[2], data_maybe_missing=False)

    if len(iTList) >= 4:
        axesT = clone_by_shape_n_fill(iTList[3], data_maybe_missing=False)
    else:
        #ONNX Slice: when `axes` is omitted, default to [0, 1, ..., len(starts)-1]
        tdata0 = np.array([i for i in range(startsT.shape[0])], dtype=np.int64)
        axesT  = make_tensor(
                name=f'{op.name}__tmp_axesT__',
                data=tdata0,
                shape=list(tdata0.shape),
                dtype='int64',
                )

    if len(iTList) == 5:
        stepsT = clone_by_shape_n_fill(iTList[4], data_maybe_missing=False)
    else:
        tdata1 = np.array([1 for _ in range(len(axesT.data))])
        stepsT  = make_tensor(
                name=f'{op.name}__tmp_stepsT__',
                data=tdata1,
                shape=list(tdata1.shape),
                dtype='int64',
                )

    assert startsT.rank() == 1, f"Slice Error 0, {startsT.shape}, rank != 1"
    assert startsT.shape == endsT.shape, f"Slice Error 1, {startsT.shape} != {endsT.shape}"
    assert startsT.shape == axesT.shape, f"Slice Error 2, {startsT.shape} != {axesT.shape}"
    assert startsT.shape == stepsT.shape, f"Slice Error 3, {startsT.shape} != {stepsT.shape}"

    Y = oTList[0]
    out_shape = list(dataT.shape)
    rank = dataT.rank()

    n_axes = startsT.shape[0]
    for s in range(n_axes):
        axis = int(axesT.data[s])
        if axis < 0:
            axis += rank
        assert 0 <= axis < rank, f"Slice axis {axis} out of bounds for rank {rank}"

        dim = int(dataT.shape[axis])
        if is_symbolic(dim):
            start_v = int(startsT.data[s])
            end_v   = int(endsT.data[s])
            step_v  = int(stepsT.data[s])
            if start_v < 0 and end_v >= int(np.iinfo(np.int64).max) and step_v == 1:
                out_shape[axis] = -start_v
            continue
        dim   = int(dim)
        start = int(startsT.data[s])
        end   = int(endsT.data[s])
        step  = int(stepsT.data[s])

        if step == 0:
            raise ValueError(f"Slice step == 0 not supported (got {step})")

        if start < 0:
            start += dim
        if end < 0:
            end += dim

        start = max(0, min(dim, start))
        end   = max(0, min(dim, end  ))

        if end <= start:
            length = 0
        else:
            length = (end - start + step - 1) // step

        out_shape[axis] = length


    Y.shape = out_shape
    Y.dtype = dataT.dtype

    if not Y.check_shape():
        raise ValueError("SLICE SHAPE INF ERROR!!")


def concat_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs['axis']
    assert len(iTList) > 0, "empty input list in Concat!!"
    base_rank = iTList[0].rank()
    assert all(x.rank() == base_rank for x in iTList), "input tensors rank mismatch"
    if axis < 0: axis = base_rank + axis
    if axis < 0 or axis >= base_rank:
        raise ValueError(f"Axis {axis} is out of bounds for tensors with rank {base_rank}. "
                    f"Valid range is [-{base_rank}, {base_rank-1}].")

    for i, x in enumerate(iTList[1:], 1):
        for dim in range(x.rank()):
            if dim != axis and x.shape[dim] != iTList[0].shape[dim]:
                    raise ValueError(f"Incompatible shapes at dim {i}: {x.shape} vs {iTList[0].shape}. "
                                     f"All dimensions except the concat axis ({axis}) must match.")

    oshape = list(iTList[0].shape)
    oshape[axis]= sum(x.shape[axis] for x in iTList)
    oTList[0].shape = oshape
    outdtype = reduce(lambda acc, t: promote_types(acc, t.dtype), iTList[1:], iTList[0].dtype)
    oTList[0].dtype = outdtype


def trilu_sinf(iTList, oTList, op, **kwargs):
    X, Y = iTList[0], oTList[0]
    if not X.check_shape():
        raise ValueError(f"Input tensor shape not defined: {X}")
    Y.shape = X.shape
    Y.dtype = X.dtype

def squeeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    dataT = iTList[0]
    axesT = clone_by_shape_n_fill(iTList[1], data_maybe_missing=False) #Y.data must be present

    data_rank  = dataT.rank()
    data_idx   = [ d + data_rank if d < 0 else d for d in axesT.data]
    checkshape = [d >= 0 and d < data_rank for d in data_idx]
    assert all(checkshape), f"axes={axesT.data} out of bounds: [-{data_rank}, {data_rank-1}]"

    #validate: squeeze only removes dims of size 1
    for idx in data_idx:
        if dataT.shape[idx] != 1:
            raise ValueError(f"Cannot squeeze dim {idx} with size {dataT.shape[idx]} -- must be 1")

    outshape = [dim for i,dim in enumerate(dataT.shape) if i not in data_idx]

    oTList[0].shape = outshape
    oTList[0].dtype = dataT.dtype

def unsqueeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    Y = clone_by_shape_n_fill(iTList[1], data_maybe_missing=False) #Y.data must be present
    newshape = list(iTList[0].shape)
    for d in Y.data:
        newrank = len(newshape)
        if d < 0: d = newrank + d + 1
        if d < 0 or d > newrank:
            raise ValueError(f"Axis {d} out of bounds: [-{newrank+1}, {newrank}]")
        newshape.insert(d, 1)

    oTList[0].shape = list(newshape)
    oTList[0].dtype = iTList[0].dtype

