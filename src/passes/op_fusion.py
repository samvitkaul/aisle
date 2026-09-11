
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from .base import PassConfig


def find_fusion_candidates(G: 'WorkloadGraph', fusion_spec) -> list:
    """Find op sequences matching fusion patterns in the graph."""
    # Normalize input: accept OpFusionSpec or raw list
    if hasattr(fusion_spec, 'op_sequences'):
        patterns = fusion_spec.op_sequences
    else:
        patterns = fusion_spec

    fusion_candidates = []
    already_matched_nodes_set = set()
    for pattern in patterns:
        pattern_len = len(pattern)
        if pattern_len <= 1:
            raise ValueError(f"Illegal fusion pattern specification {pattern}!!")
        for opname in G.get_ordered_nodes():
            if opname in already_matched_nodes_set:
                continue

            op = G.get_op(opname)
            if op.removed_in_optimization:
                continue

            if op.optype.upper() == pattern[0]:
                current_node          = opname
                matched_nodes_list    = [current_node]
                for i in range(1, pattern_len):
                    successors = G.get_successors(current_node)
                    if len(successors) == 1:
                        succ = successors[0]
                        # Fan-in aware matching
                        succ_preds = G.get_predecessors(succ)
                        extra_preds = [p for p in succ_preds if p not in matched_nodes_list]
                        all_extra_are_const = all(
                            G.get_op(p).optype.upper() in ('CONSTANT', 'IDENTITY')
                            or all(G.get_tensor(t).is_param or G.get_tensor(t).is_const
                                   for t in G.get_op(p).outList)
                            for p in extra_preds
                        )
                        if (G.get_op(succ).optype.upper() == pattern[i] and
                            succ not in already_matched_nodes_set and
                            (len(succ_preds) == 1 or all_extra_are_const)):
                            current_node = succ
                            matched_nodes_list.append(current_node)
                        elif G.is_removed(succ):
                            # Skip nodes that have been marked for removal during optimization
                            # and continue looking at their successors instead
                            current_node = succ
                            next_successors = G.get_successors(current_node)
                            if next_successors:
                                current_node = next_successors[0]
                                if (G.get_op(current_node).optype.upper() == pattern[i] and
                                    current_node not in already_matched_nodes_set):
                                    matched_nodes_list.append(current_node)
                                else:
                                    # Pattern doesn't match after skipping removed node
                                    break
                            else:
                                # No more successors after a removed node
                                break
                        else:
                            # pattern does not match break
                            break
                    else:
                        # multiple or no successors, break
                        break
                if len(matched_nodes_list) == pattern_len:
                    fusion_candidates.append(matched_nodes_list)
                    already_matched_nodes_set.update(matched_nodes_list)
    return fusion_candidates


class OpFusionPass:
    name = "op_fusion"
    depends_on: list[str] = ["resource_mapping"]

    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph':
        if config.op_fusion_spec is None:
            return G
        fusion_candidates = find_fusion_candidates(G, config.op_fusion_spec)
        for fusion_nodes in fusion_candidates:
            first_op_name = fusion_nodes[0]
            for i in range(1, len(fusion_nodes)):
                G.get_op(fusion_nodes[i]).fuse_op(first_op_name)
        return G
