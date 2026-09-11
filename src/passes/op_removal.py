
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from .base import PassConfig


class OpRemovalPass:
    name = "op_removal"
    depends_on: list[str] = []

    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph':
        if config.op_removal_spec is None:
            return G
        for opname in G.get_ordered_nodes():
            op = G.get_op(opname)
            optype = op.optype.upper()
            if config.op_removal_spec.check(optype):
                op.removed_in_optimization = True
        return G
