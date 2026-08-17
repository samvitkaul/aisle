
from ..bten.tensor   import Tensor
from ..bten.op       import TensorOp
from ..utils.data_types import str2dt, get_bpe

import networkx as nx
from typing import Dict, List

class WorkloadGraph:
    def __init__(self, name):
        self._name     : str                 = name
        self._graph    : nx.MultiDiGraph     = nx.MultiDiGraph()
        self._tensors  : Dict[str, Tensor]   = {}
        self._ops      : Dict[str, TensorOp] = {}
        self._inodes   : List[str]           = []
        self._onodes   : List[str]           = []
        self._itensors : List[str]           = []
        self._otensors : List[str]           = []
        return

    def clone_for_execute(self) -> 'WorkloadGraph':
        """Fast structural clone for per-experiment isolation (Task 031).

        Shares topology (``_graph``, ``_inodes``, ``_onodes``,
        ``_itensors``, ``_otensors``) by reference — these are read-only
        after ``construct_graph()``. Allocates fresh ``TensorOp`` and
        ``Tensor`` objects via their ``clone_for_execute()`` methods,
        which reset mutation-prone fields while sharing invariant fields
        by reference. Designed as a faster drop-in for
        ``copy.deepcopy(wl_graph)`` inside ``execute_workload_on_system``.

        Does NOT preserve topology mutations — if a future pass adds or
        removes nodes, this method's invariants break.
        """
        clone = WorkloadGraph(self._name)
        clone._graph    = self._graph
        clone._inodes   = self._inodes
        clone._onodes   = self._onodes
        clone._itensors = self._itensors
        clone._otensors = self._otensors
        clone._tensors  = {name: t.clone_for_execute()
                           for name, t in self._tensors.items()}
        clone._ops      = {name: op.clone_for_execute()
                           for name, op in self._ops.items()}
        return clone

    def get_node_count(self)     : return self._graph.number_of_nodes()
    def get_edge_count(self)     : return self._graph.number_of_edges()
    def has_node(self, name)     : return self._graph.has_node(name)
    def get_input_nodes(self)    : return self._inodes
    def get_output_nodes(self)   : return self._onodes
    def get_input_tensors(self)  : return self._itensors
    def get_output_tensors(self) : return self._otensors
    def get_ordered_nodes(self)  : return list(nx.topological_sort(self._graph))

    def is_input_node(self, opname) : return opname in self._inodes
    def is_output_node(self, opname): return opname in self._onodes

    def add_tensor(self, tensor: Tensor):
        if tensor.name in self._tensors:
            raise ValueError(f"Tensor({tensor.name}) is not unique!!!")
        self._tensors[tensor.name] = tensor

    def add_op(self, op: TensorOp):
        for tensor_name in op.inList:
            if tensor_name not in self._tensors:
                raise KeyError(f"Input Tensor {tensor_name} for TensorOp {op.name} not found in WorkloadGraph")
        for tensor_name in op.outList:
            if tensor_name not in self._tensors:
                raise KeyError(f"Output Tensor {tensor_name} for TensorOp {op.name} not found in WorkloadGraph")
        if op.name in self._ops:
            raise ValueError(f"TensorOp({op.name}) is not unique!!!")
        self._ops[op.name] = op
        return

    def construct_graph(self):
        for op_count, (op_name, op_info) in enumerate(self._ops.items()):
            self._graph.add_node(op_name)
            for o in op_info.outList:
                for inode in self._tensors[o].op_in:
                    self._graph.add_edge(op_name, inode, name=o)
        self._itensors = sorted(set([tname for tname,tval in self._tensors.items() if tval.op_out == []]))
        self._otensors = sorted(set([tname for tname,tval in self._tensors.items() if tval.op_in == []]))
        self._inodes   = sorted(set([o for t in self._itensors for o in self._tensors[t].op_in]))
        self._onodes   = sorted(set([o for t in self._otensors for o in self._tensors[t].op_out]))
        return

    def get_tensor(self, tname): return self._tensors[tname]
    def get_op(self, opname): return self._ops[opname]
    def is_removed(self, opname): return self._ops[opname].removed_in_optimization
    def get_successors(self, opname): return list(self._graph.successors(opname))
    def get_predecessors(self, opname): return list(self._graph.predecessors(opname))

    def fuse_nodes(self, fusion_spec):
        from src.passes.op_fusion import find_fusion_candidates
        return find_fusion_candidates(self, fusion_spec)

    def op_stat_iter(self, statname, /, repeat=False, use_precision=False):
        from dataclasses import fields as _dc_fields
        for opname in self.get_ordered_nodes():
            op = self.get_op(opname)
            if hasattr(op, statname):
                val = getattr(op, statname)
            elif op.kernel_desc is not None and statname in {f.name for f in _dc_fields(op.kernel_desc)}:
                val = getattr(op.kernel_desc, statname)
            elif hasattr(op.exec_stats, statname):
                val = getattr(op.exec_stats, statname)
            else:
                raise KeyError(f"unable to find {statname} in {op}")

            if repeat:
                val = val * op.repeat_count

            if use_precision:
                val = val * get_bpe(str2dt(op.precision))

            yield val

    @classmethod
    def from_json(cls, path: str, **kwargs) -> 'WorkloadGraph':
        from .json2graph import json2graph
        return json2graph(path, **kwargs)

    def to_json(self, path: str, **kwargs) -> None:
        from .graph2json import graph2json
        graph2json(self, path, **kwargs)

    @classmethod
    def from_onnx(cls, path: str, **kwargs) -> 'WorkloadGraph':
        from .onnx2graph import onnx2graph
        return onnx2graph(path, **kwargs)
