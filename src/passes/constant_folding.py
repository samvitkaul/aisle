from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from .base import PassConfig


class ConstantFoldingPass:
    name = "constant_folding"
    depends_on: list[str] = ["op_removal"]

    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph':
        from src.bten.op import RemovalReason
        for opname in G.get_ordered_nodes():
            op = G.get_op(opname)
            if op.removed_in_optimization:
                continue  # already removed by OpRemovalPass
            if not op.inList:
                continue  # no inputs — nothing to fold (source op)

            all_inputs_const = all(
                self._is_const_or_removed(G, tname)
                for tname in op.inList
            )
            if all_inputs_const:
                op.removal_reason = RemovalReason.CONSTANT_FOLDED
                # Promote output tensors to const so downstream ops
                # can cascade-fold
                for tname in op.outList:
                    G.get_tensor(tname).is_const = True
        return G

    @staticmethod
    def _is_const_or_removed(G: 'WorkloadGraph', tname: str) -> bool:
        tensor = G.get_tensor(tname)
        if tensor.is_const or tensor.is_param:
            return True
        # A tensor produced by no op (graph input) that isn't const/param
        # is an activation — not foldable
        if not tensor.op_out:
            return False
        # A tensor is foldable if it's already marked const (e.g., by a
        # previous cascade-fold step that promoted it). We don't infer
        # const-ness from the removal status of producers — a user-removed
        # Identity op may still pass through an activation.
        return False
