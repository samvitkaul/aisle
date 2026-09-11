
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from .base import PassConfig


class ResourceMappingPass:
    name = "resource_mapping"
    depends_on: list[str] = ["dead_node_elimination"]

    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph':
        # NOTE: this pass currently only annotates op.precision from the
        # output tensor dtype; rsrc_spec is not yet consumed in the body
        # (ResourceMap.op2pipe has no callers today). The precision
        # annotation must run unconditionally because get_compute_ticks
        # downstream requires op.precision to be non-None. Re-introduce a
        # `rsrc_spec is None` guard only when actual op->pipe mapping
        # logic is added here.
        for opname in G.get_ordered_nodes():
            op = G.get_op(opname)
            if op.removed_in_optimization:
                continue
            op.precision = f"{G.get_tensor(op.outList[0]).dtype}"
        return G
