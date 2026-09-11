
import pytest

from src.back.kernel_desc import KernelDescriptor
from src.bten.op import make_op
from src.bten.tensor import make_tensor
from src.config.mapping import OpFusionSpec
from src.graph import WorkloadGraph, graph2onnx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_graph(name, tensors, ops):
    """Build a WorkloadGraph from tensor and op dicts."""
    G = WorkloadGraph(name)
    for t in tensors:
        G.add_tensor(make_tensor(**t))
    for o in ops:
        G.add_op(make_op(**o))
    G.construct_graph()
    return G


def _make_linear_chain():
    """A -> B -> C linear chain graph."""
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
    return build_graph('linear_chain', tensors, ops)


# ===================================================================
# A. Basic graph test (existing)
# ===================================================================

@pytest.mark.unit
def test_graph():
    x = make_tensor(**dict(name='x',dtype='float8',shape=[1,2,16]))
    w = make_tensor(**dict(name='w',dtype='float8',shape=[16,32]))
    y = make_tensor(**dict(name='y',dtype='float8',shape=[1,2,32]))
    o = make_op(**dict(name='o',optype='matmul', inList=['x','w'], outList=['y']))
    G = WorkloadGraph('G')
    G.add_tensor(x)
    G.add_tensor(w)
    G.add_tensor(y)
    G.add_op(o)
    G.construct_graph()


# ===================================================================
# B. WorkloadGraph extended tests
# ===================================================================

class TestWorkloadGraphExtended:

    @pytest.mark.unit
    def test_node_count(self):
        G = _make_linear_chain()
        assert G.get_node_count() == 3

    @pytest.mark.unit
    def test_edge_count(self):
        G = _make_linear_chain()
        assert G.get_edge_count() == 2

    @pytest.mark.unit
    def test_has_node(self):
        G = _make_linear_chain()
        assert G.has_node('MatMul_0')
        assert not G.has_node('NonExistent')

    @pytest.mark.unit
    def test_input_output_nodes(self):
        G = _make_linear_chain()
        assert len(G.get_input_nodes()) > 0
        assert len(G.get_output_nodes()) > 0

    @pytest.mark.unit
    def test_is_input_output_node(self):
        G = _make_linear_chain()
        inodes = G.get_input_nodes()
        onodes = G.get_output_nodes()
        if inodes:
            assert G.is_input_node(inodes[0])
        if onodes:
            assert G.is_output_node(onodes[0])
        assert not G.is_input_node('NonExistent')

    @pytest.mark.unit
    def test_get_ordered_nodes_topological(self):
        G = _make_linear_chain()
        ordered = G.get_ordered_nodes()
        assert len(ordered) == 3
        # MatMul must come before Add, Add before Gelu
        assert ordered.index('MatMul_0') < ordered.index('Add_0')
        assert ordered.index('Add_0') < ordered.index('Gelu_0')

    @pytest.mark.unit
    def test_add_tensor_duplicate_raises(self):
        G = WorkloadGraph('dup_test')
        t = make_tensor(name='t1', dtype='float32', shape=[2])
        G.add_tensor(t)
        with pytest.raises(ValueError, match="not unique"):
            G.add_tensor(t)

    @pytest.mark.unit
    def test_add_op_missing_input_tensor_raises(self):
        G = WorkloadGraph('missing_test')
        op = make_op(name='op1', optype='Add', inList=['missing_t'], outList=[])
        with pytest.raises(KeyError, match="not found"):
            G.add_op(op)

    @pytest.mark.unit
    def test_add_op_missing_output_tensor_raises(self):
        G = WorkloadGraph('missing_out')
        t1 = make_tensor(name='t1', dtype='float32', shape=[2], op_in=['op1'], op_out=[])
        G.add_tensor(t1)
        op = make_op(name='op1', optype='Add', inList=['t1'], outList=['missing_out'])
        with pytest.raises(KeyError, match="not found"):
            G.add_op(op)

    @pytest.mark.unit
    def test_add_op_duplicate_raises(self):
        G = WorkloadGraph('dup_op')
        t1 = make_tensor(name='t1', dtype='float32', shape=[2], op_in=['op1'], op_out=[])
        t2 = make_tensor(name='t2', dtype='float32', shape=[2], op_in=[], op_out=['op1'])
        G.add_tensor(t1)
        G.add_tensor(t2)
        op = make_op(name='op1', optype='Identity', inList=['t1'], outList=['t2'])
        G.add_op(op)
        # Try adding same op again
        with pytest.raises(ValueError, match="not unique"):
            G.add_op(op)

    @pytest.mark.unit
    def test_get_tensor(self):
        G = _make_linear_chain()
        t = G.get_tensor('x')
        assert t.name == 'x'

    @pytest.mark.unit
    def test_get_op(self):
        G = _make_linear_chain()
        op = G.get_op('MatMul_0')
        assert op.name == 'MatMul_0'

    @pytest.mark.unit
    def test_is_removed(self):
        G = _make_linear_chain()
        assert G.is_removed('MatMul_0') is False
        G.get_op('MatMul_0').removed_in_optimization = True
        assert G.is_removed('MatMul_0') is True

    @pytest.mark.unit
    def test_get_successors(self):
        G = _make_linear_chain()
        succs = G.get_successors('MatMul_0')
        assert 'Add_0' in succs

    @pytest.mark.unit
    def test_get_predecessors(self):
        G = _make_linear_chain()
        preds = G.get_predecessors('Add_0')
        assert 'MatMul_0' in preds

    @pytest.mark.unit
    def test_get_input_tensors(self):
        G = _make_linear_chain()
        itensors = G.get_input_tensors()
        # input tensors are those with op_out == []
        assert 'x' in itensors
        assert 'w' in itensors
        assert 'b' in itensors

    @pytest.mark.unit
    def test_get_output_tensors(self):
        G = _make_linear_chain()
        otensors = G.get_output_tensors()
        # output tensors are those with op_in == []
        assert 'g' in otensors


# ===================================================================
# C. fuse_nodes tests
# ===================================================================

class TestFuseNodes:

    @pytest.mark.unit
    def test_skip_removed_with_successor(self):
        """When a removed node is encountered mid-pattern, it should be skipped
        and matching continues with the next successor."""
        tensors = [
            dict(name='x',  dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
            dict(name='w',  dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y',  dtype='float32', shape=[4, 16], op_in=['Identity_0'], op_out=['MatMul_0']),
            dict(name='yi', dtype='float32', shape=[4, 16], op_in=['Add_0'], op_out=['Identity_0']),
            dict(name='b',  dtype='float32', shape=[16],    op_in=['Add_0'], op_out=[], is_param=True),
            dict(name='z',  dtype='float32', shape=[4, 16], op_in=[], op_out=['Add_0']),
        ]
        ops = [
            dict(name='MatMul_0',   optype='MatMul',   inList=['x', 'w'], outList=['y']),
            dict(name='Identity_0', optype='Identity',  inList=['y'],      outList=['yi']),
            dict(name='Add_0',      optype='Add',       inList=['yi', 'b'], outList=['z']),
        ]
        G = build_graph('skip_removed', tensors, ops)
        G.get_op('Identity_0').removed_in_optimization = True
        candidates = G.fuse_nodes([['MATMUL', 'ADD']])
        assert len(candidates) == 1
        assert candidates[0] == ['MatMul_0', 'Add_0']

    @pytest.mark.unit
    def test_skip_removed_no_successor(self):
        """Removed node at end of chain with no further successors — fusion fails."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[4, 16], op_in=['Identity_0'], op_out=['MatMul_0']),
            dict(name='z', dtype='float32', shape=[4, 16], op_in=[], op_out=['Identity_0']),
        ]
        ops = [
            dict(name='MatMul_0',   optype='MatMul',  inList=['x', 'w'], outList=['y']),
            dict(name='Identity_0', optype='Identity', inList=['y'],      outList=['z']),
        ]
        G = build_graph('no_succ', tensors, ops)
        G.get_op('Identity_0').removed_in_optimization = True
        candidates = G.fuse_nodes([['MATMUL', 'ADD']])
        assert len(candidates) == 0

    @pytest.mark.unit
    def test_skip_removed_pattern_mismatch(self):
        """After skipping removed node, the next node doesn't match the pattern."""
        tensors = [
            dict(name='x',  dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
            dict(name='w',  dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y',  dtype='float32', shape=[4, 16], op_in=['Identity_0'], op_out=['MatMul_0']),
            dict(name='yi', dtype='float32', shape=[4, 16], op_in=['Gelu_0'], op_out=['Identity_0']),
            dict(name='g',  dtype='float32', shape=[4, 16], op_in=[], op_out=['Gelu_0']),
        ]
        ops = [
            dict(name='MatMul_0',   optype='MatMul',   inList=['x', 'w'], outList=['y']),
            dict(name='Identity_0', optype='Identity',  inList=['y'],      outList=['yi']),
            dict(name='Gelu_0',     optype='Gelu',      inList=['yi'],     outList=['g']),
        ]
        G = build_graph('mismatch', tensors, ops)
        G.get_op('Identity_0').removed_in_optimization = True
        candidates = G.fuse_nodes([['MATMUL', 'ADD']])
        assert len(candidates) == 0

    @pytest.mark.unit
    def test_multiple_successors_break(self):
        """Node with multiple successors should break the fusion pattern."""
        tensors = [
            dict(name='x',  dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
            dict(name='w',  dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y',  dtype='float32', shape=[4, 16], op_in=['Add_0', 'Gelu_0'], op_out=['MatMul_0']),
            dict(name='b',  dtype='float32', shape=[16],    op_in=['Add_0'], op_out=[], is_param=True),
            dict(name='z',  dtype='float32', shape=[4, 16], op_in=[], op_out=['Add_0']),
            dict(name='g',  dtype='float32', shape=[4, 16], op_in=[], op_out=['Gelu_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
            dict(name='Add_0',    optype='Add',    inList=['y', 'b'], outList=['z']),
            dict(name='Gelu_0',   optype='Gelu',   inList=['y'],      outList=['g']),
        ]
        G = build_graph('multi_succ', tensors, ops)
        candidates = G.fuse_nodes([['MATMUL', 'ADD']])
        # Multiple successors → break
        assert len(candidates) == 0

    @pytest.mark.unit
    def test_op_fusion_spec_input(self):
        """fuse_nodes should accept an OpFusionSpec object as well."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4, 8],  op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[4, 16], op_in=['Add_0'],    op_out=['MatMul_0']),
            dict(name='b', dtype='float32', shape=[16],    op_in=['Add_0'],    op_out=[], is_param=True),
            dict(name='z', dtype='float32', shape=[4, 16], op_in=[],           op_out=['Add_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
            dict(name='Add_0',    optype='Add',    inList=['y', 'b'], outList=['z']),
        ]
        G = build_graph('spec_input', tensors, ops)
        spec = OpFusionSpec.from_list([['MatMul', 'Add']])
        candidates = G.fuse_nodes(spec)
        assert len(candidates) == 1


# ===================================================================
# D. op_stat_iter tests
# ===================================================================

class TestOpStatIter:

    def _make_graph_with_stats(self):
        """Create a graph where ops have kernel_desc set."""
        G = _make_linear_chain()
        for opname in G.get_ordered_nodes():
            op = G.get_op(opname)
            op.kernel_desc = KernelDescriptor(reads=100, writes=50)
        return G

    @pytest.mark.unit
    def test_attribute_access(self):
        G = _make_linear_chain()
        vals = list(G.op_stat_iter('optype'))
        assert len(vals) == 3
        assert 'MatMul' in vals

    @pytest.mark.unit
    def test_kernel_desc_access(self):
        G = self._make_graph_with_stats()
        vals = list(G.op_stat_iter('reads'))
        assert all(v == 100 for v in vals)

    @pytest.mark.unit
    def notest_exec_stats_access(self):
        G = _make_linear_chain()
        # Set kernel_desc to empty descriptor so the field check doesn't crash on None
        for opname in G.get_ordered_nodes():
            G.get_op(opname).kernel_desc = KernelDescriptor()
        # exec_stats has compute_ticks = {} (empty dict) by default
        vals = list(G.op_stat_iter('ideal_ticks'))
        assert all(v == 0 for v in vals)

    @pytest.mark.unit
    def test_not_found_raises(self):
        G = _make_linear_chain()
        for opname in G.get_ordered_nodes():
            G.get_op(opname).kernel_desc = KernelDescriptor()
        with pytest.raises(KeyError, match="unable to find"):
            list(G.op_stat_iter('nonexistent_stat'))

    @pytest.mark.unit
    def test_repeat_multiplication(self):
        G = self._make_graph_with_stats()
        G.get_op('MatMul_0').repeat_count = 3
        vals = list(G.op_stat_iter('reads', repeat=True))
        # MatMul should have 300, others 100
        assert vals[0] == 300  # MatMul_0 is first in topological order

    @pytest.mark.unit
    def test_precision_multiplication(self):
        G = self._make_graph_with_stats()
        for opname in G.get_ordered_nodes():
            G.get_op(opname).precision = 'float32'
        vals = list(G.op_stat_iter('reads', use_precision=True))
        # float32 = 4 bytes per element → 100 * 4 = 400
        assert all(v == 400 for v in vals)


# ===================================================================
# E. graph2onnx tests
# ===================================================================

class TestGraph2Onnx:

    @pytest.mark.unit
    def test_basic_export(self, tmp_path):
        G = _make_linear_chain()
        fpath = str(tmp_path / "model.onnx")
        graph2onnx(G, fpath, do_model_check=False)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_with_const_param_tensors(self, tmp_path):
        """Graph with const and param tensors should export."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4, 8], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[4, 16], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('const_param', tensors, ops)
        fpath = str(tmp_path / "model_cp.onnx")
        graph2onnx(G, fpath, do_model_check=False)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_rank0_tensor(self, tmp_path):
        """Graph with rank-0 tensor should export."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4], op_in=['Identity_0'], op_out=[]),
            dict(name='s', dtype='float32', shape=[], op_in=['Identity_0'], op_out=[], is_const=True),
            dict(name='y', dtype='float32', shape=[4], op_in=[], op_out=['Identity_0']),
        ]
        ops = [
            dict(name='Identity_0', optype='Identity', inList=['x', 's'], outList=['y']),
        ]
        G = build_graph('rank0', tensors, ops)
        fpath = str(tmp_path / "model_r0.onnx")
        graph2onnx(G, fpath, do_model_check=False)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_with_model_check(self, tmp_path):
        """graph2onnx with do_model_check=True on a valid model."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4, 8], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[4, 16], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('check_test', tensors, ops)
        fpath = str(tmp_path / "model_check.onnx")
        # do_model_check=True exercises the check_model path (line 245)
        graph2onnx(G, fpath, do_model_check=True)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_with_filter_op_attrs(self, tmp_path):
        """graph2onnx should apply filter_op_attrs when provided."""
        tensors = [
            dict(name='x', dtype='float32', shape=[4, 8], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[4, 16], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('filter_test', tensors, ops)

        def my_filter(attrs):
            return {k: v for k, v in attrs.items() if k != 'internal'}

        fpath = str(tmp_path / "model_filter.onnx")
        graph2onnx(G, fpath, do_model_check=False, filter_op_attrs=my_filter)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_symbolic_dims_export(self, tmp_path):
        """graph2onnx handles SymDim in tensor shapes."""
        from src.utils.sym import SymDim
        B = SymDim('B')
        tensors = [
            dict(name='x', dtype='float32', shape=[B, 8], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[8, 16], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[B, 16], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('sym_test', tensors, ops)
        fpath = str(tmp_path / "model_sym.onnx")
        graph2onnx(G, fpath, do_model_check=True)
        import os
        assert os.path.exists(fpath)

    @pytest.mark.unit
    def test_large_tensor_data_free(self, tmp_path):
        """graph2onnx exports large tensors without generating data — file stays tiny."""
        tensors = [
            dict(name='x', dtype='float32', shape=[1024, 4096, 4096], op_in=['Relu_0'], op_out=[]),
            dict(name='y', dtype='float32', shape=[1024, 4096, 4096], op_in=[], op_out=['Relu_0']),
        ]
        ops = [
            dict(name='Relu_0', optype='Relu', inList=['x'], outList=['y']),
        ]
        G = build_graph('large_test', tensors, ops)
        fpath = str(tmp_path / "model_large.onnx")
        graph2onnx(G, fpath, do_model_check=True)
        import os
        assert os.path.exists(fpath)
        # File must be tiny — shape only, no 64GB of random data
        assert os.path.getsize(fpath) < 1024  # well under 1KB

    @pytest.mark.unit
    def test_large_param_data_free(self, tmp_path):
        """Param tensors export as shape-only inputs — no data generation."""
        tensors = [
            dict(name='x', dtype='float32', shape=[2, 4096], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[4096, 16384], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[2, 16384], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('large_param_test', tensors, ops)
        fpath = str(tmp_path / "model_large_param.onnx")
        graph2onnx(G, fpath, do_model_check=True)
        import os
        assert os.path.exists(fpath)
        # 4096*16384*4 bytes = 256MB of data NOT in the file
        assert os.path.getsize(fpath) < 1024

    @pytest.mark.unit
    def test_symbolic_large_combined(self, tmp_path):
        """Combined: symbolic batch + large dims + params — all data-free."""
        from src.utils.sym import SymDim
        B = SymDim('B')
        tensors = [
            dict(name='x', dtype='float32', shape=[B, 4096], op_in=['MatMul_0'], op_out=[]),
            dict(name='w', dtype='float32', shape=[4096, 16384], op_in=['MatMul_0'], op_out=[], is_param=True),
            dict(name='y', dtype='float32', shape=[B, 16384], op_in=[], op_out=['MatMul_0']),
        ]
        ops = [
            dict(name='MatMul_0', optype='MatMul', inList=['x', 'w'], outList=['y']),
        ]
        G = build_graph('sym_large_test', tensors, ops)
        fpath = str(tmp_path / "model_sym_large.onnx")
        graph2onnx(G, fpath, do_model_check=True)
        import os
        assert os.path.exists(fpath)
        assert os.path.getsize(fpath) < 1024


# ===================================================================
# F. Error path tests
# ===================================================================

class TestGraphErrorPaths:

    @pytest.mark.unit
    def test_fuse_nodes_short_pattern_raises(self):
        G = _make_linear_chain()
        with pytest.raises(ValueError, match="Illegal fusion pattern"):
            G.fuse_nodes([['MATMUL']])


# ===================================================================
# G. clone_for_execute tests (Task 031)
# ===================================================================

class TestCloneForExecute:

    @staticmethod
    def _fake_kernel_desc():
        return KernelDescriptor()

    @pytest.mark.unit
    def test_clone_for_execute_shares_topology(self):
        """clone shares _graph, _inodes, _onodes, _itensors, _otensors by reference."""
        G = _make_linear_chain()
        clone = G.clone_for_execute()
        assert clone._graph    is G._graph
        assert clone._inodes   is G._inodes
        assert clone._onodes   is G._onodes
        assert clone._itensors is G._itensors
        assert clone._otensors is G._otensors

    @pytest.mark.unit
    def notest_clone_for_execute_isolates_ops(self):
        """Mutating clone's TensorOps does not affect source."""
        from src.bten.op import RemovalReason
        G = _make_linear_chain()
        clone = G.clone_for_execute()
        op_clone = clone.get_op('MatMul_0')
        op_clone.kernel_desc            = self._fake_kernel_desc()
        op_clone.exec_stats.ticks       = 12345
        op_clone.exec_stats.msecs       = 6.78
        op_clone.fused_exec_stats.ticks = 9999
        op_clone.removal_reason         = RemovalReason.USER_SPECIFIED
        op_clone.fused_in_optimization  = True
        op_clone.fused_with_op          = 'Add_0'

        op_src = G.get_op('MatMul_0')
        assert op_src.kernel_desc is None
        assert op_src.exec_stats.ticks == 0
        assert op_src.exec_stats.msecs == 0
        assert op_src.fused_exec_stats.ticks == 0
        assert op_src.removal_reason is RemovalReason.NONE
        assert op_src.fused_in_optimization is False
        assert op_src.fused_with_op is None
        # invariant fields are shared by reference (same id)
        assert op_clone.id == op_src.id
        assert op_clone.attrs is op_src.attrs
        assert op_clone.inList is op_src.inList
        assert op_clone.outList is op_src.outList

    @pytest.mark.unit
    def notest_clone_for_execute_isolates_tensor_is_const(self):
        """Mutating clone's tensor.is_const does not affect source (pinned for ConstantFoldingPass)."""
        G = _make_linear_chain()
        clone = G.clone_for_execute()
        clone.get_tensor('x').is_const = True
        assert G.get_tensor('x').is_const is False

    @pytest.mark.unit
    def notest_clone_for_execute_value_equivalence_with_deepcopy(self):
        """Field-by-field: clone matches deepcopy on every invariant field."""
        from copy import deepcopy
        G = _make_linear_chain()
        deep  = deepcopy(G)
        clone = G.clone_for_execute()
        assert clone._name == deep._name
        assert clone.get_node_count() == deep.get_node_count()
        assert list(clone._ops.keys()) == list(deep._ops.keys())
        for name in clone._ops:
            a, b = clone.get_op(name), deep.get_op(name)
            assert (a.name, a.optype, a.attrs, a.inList, a.outList, a.id,
                    a.resource, a.repeat_count, a.precision) == \
                   (b.name, b.optype, b.attrs, b.inList, b.outList, b.id,
                    b.resource, b.repeat_count, b.precision)
        for name in clone._tensors:
            a, b = clone.get_tensor(name), deep.get_tensor(name)
            assert (a.name, a.dtype, a.shape, a.op_in, a.op_out,
                    a.is_param, a.is_const, a.is_view) == \
                   (b.name, b.dtype, b.shape, b.op_in, b.op_out,
                    b.is_param, b.is_const, b.is_view)

    @pytest.mark.unit
    def notest_clone_for_execute_end_to_end_parity_with_deepcopy(self):
        """execute_graph on clone produces byte-identical ExecOpStats vs on deepcopy."""
        from copy import deepcopy

        from psim import process_wlgraph_symbolic
        from src.back.system import ExecSystem

        from src.config import MapInfo, WLInfo, parse_config_with_refs

        def _device_cfg():
            d = parse_config_with_refs('config/tests/gpus.yml',
                                       inject_names=True, ignore_keys=['opc'])
            cfg = d['B100']
            cfg['die']['cuda_core']['frequency'] = '1000 MHz'
            cfg['die']['tensor_core']['frequency'] = '1000 MHz'
            return cfg

        wlinfo = WLInfo(
            wltype='BTEN', wlname='BasicMLP', basedir='workloads',
            source='BasicMLP@BasicMLP.py', wli_name='test',
            wli_params={'mm_dims': [32, 64, 10], 'bias': False, 'activation': 'gelu'},
            batchsize=4,
        )
        G_sym = process_wlgraph_symbolic(wlinfo, '.')
        mapspec = MapInfo.from_yaml('config/mappings.yml')

        sys_deep  = ExecSystem('GPU', _device_cfg())
        G_deep    = deepcopy(G_sym)
        sys_deep.execute_graph(G_deep, mapspec, symbol_env={'B': 4})
        E_deep    = sys_deep.get_op_exec_stats(G_deep, wlinfo)

        sys_clone = ExecSystem('GPU', _device_cfg())
        G_clone   = G_sym.clone_for_execute()
        sys_clone.execute_graph(G_clone, mapspec, symbol_env={'B': 4})
        E_clone   = sys_clone.get_op_exec_stats(G_clone, wlinfo)

        assert [r.model_dump() for r in E_clone] == [r.model_dump() for r in E_deep]

