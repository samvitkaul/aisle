"""Graph fixtures for round-trip serde tests.

Re-hosted from ``tests/test_graph.py::TestGraph2Onnx`` so the JSON
round-trip property test has stable access to the 9 fixtures.
"""
from src.bten.tensor import make_tensor
from src.bten.op import make_op
from src.utils.sym import SymDim
from src.graph import WorkloadGraph


def _build(name, tensors, ops):
    G = WorkloadGraph(name)
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()
    return G


def make_linear_chain():
    tensors = [
        dict(name='x', dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[4, 16], op_in=['Add_0'],    op_out=['MatMul_0']),
        dict(name='b', dtype='float32', shape=[16],    op_in=['Add_0'],    op_out=[], is_param=True),
        dict(name='z', dtype='float32', shape=[4, 16], op_in=['Gelu_0'],   op_out=['Add_0']),
        dict(name='g', dtype='float32', shape=[4, 16], op_in=[],           op_out=['Gelu_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        dict(name='Add_0',    optype='Add',    inList=['y', 'b'], outList=['z']),
        dict(name='Gelu_0',   optype='Gelu',   inList=['z'],      outList=['g']),
    ]
    return _build('linear_chain', tensors, ops)


def make_const_param():
    tensors = [
        dict(name='x', dtype='float32', shape=[4, 8], op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[4, 16], op_in=[], op_out=['MatMul_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
    ]
    return _build('const_param', tensors, ops)


def make_rank0_const():
    tensors = [
        dict(name='x', dtype='float32', shape=[4], op_in=['Mul_0'], op_out=[]),
        dict(name='s', dtype='float32', shape=[],  op_in=['Mul_0'],
             op_out=[], is_const=True, data=2.5),
        dict(name='y', dtype='float32', shape=[4], op_in=[], op_out=['Mul_0']),
    ]
    ops = [
        dict(name='Mul_0', optype='Mul', inList=['x', 's'], outList=['y']),
    ]
    return _build('rank0', tensors, ops)


def make_check_test():
    return make_const_param()  # same shape, different name; spec lists distinctly


def make_filter_test():
    tensors = [
        dict(name='x', dtype='float32', shape=[4, 8], op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[4, 16], op_in=[], op_out=['MatMul_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y'],
             attrs={'alpha': 1.0}),
    ]
    return _build('filter_test', tensors, ops)


def make_sym_test():
    B = SymDim('B')
    tensors = [
        dict(name='x', dtype='float32', shape=[B, 8], op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[B, 16], op_in=[], op_out=['MatMul_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
    ]
    return _build('sym_test', tensors, ops)


def make_large_test():
    tensors = [
        dict(name='x', dtype='float32', shape=[1024, 4096, 4096], op_in=['Relu_0'], op_out=[]),
        dict(name='y', dtype='float32', shape=[1024, 4096, 4096], op_in=[], op_out=['Relu_0']),
    ]
    ops = [
        dict(name='Relu_0', optype='Relu', inList=['x'], outList=['y']),
    ]
    return _build('large_test', tensors, ops)


def make_large_param_test():
    tensors = [
        dict(name='x', dtype='float32', shape=[2, 4096], op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[4096, 16384],
             op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[2, 16384], op_in=[], op_out=['MatMul_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
    ]
    return _build('large_param_test', tensors, ops)


def make_sym_large_test():
    B = SymDim('B')
    tensors = [
        dict(name='x', dtype='float32', shape=[B, 4096], op_in=['MatMul_0'], op_out=[]),
        dict(name='w', dtype='float32', shape=[4096, 16384],
             op_in=['MatMul_0'], op_out=[], is_param=True),
        dict(name='y', dtype='float32', shape=[B, 16384], op_in=[], op_out=['MatMul_0']),
    ]
    ops = [
        dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
    ]
    return _build('sym_large_test', tensors, ops)


ALL_GRAPH_FIXTURES = [
    make_linear_chain,
    make_const_param,
    make_rank0_const,
    make_check_test,
    make_filter_test,
    make_sym_test,
    make_large_test,
    make_large_param_test,
    make_sym_large_test,
]
