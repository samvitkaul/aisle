from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from .base import PassConfig


class DeadNodeEliminationPass:
    name = "dead_node_elimination"
    depends_on: list[str] = ["constant_folding"]

    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph':
        from src.bten.op import RemovalReason
        ordered = G.get_ordered_nodes()
        # Reverse topological order: process consumers before producers
        for opname in reversed(ordered):
            op = G.get_op(opname)
            if op.removed_in_optimization:
                continue  # already removed
            if G.is_output_node(opname):
                continue  # graph outputs are always live

            # Check: are ALL output tensors consumed only by removed ops
            # (or not consumed at all)?
            all_outputs_dead = True
            for tname in op.outList:
                tensor = G.get_tensor(tname)
                for consumer_op_name in tensor.op_in:
                    if not G.get_op(consumer_op_name).removed_in_optimization:
                        all_outputs_dead = False
                        break
                if not all_outputs_dead:
                    break

            if all_outputs_dead:
                op.removal_reason = RemovalReason.DEAD_ELIMINATED
        return G
