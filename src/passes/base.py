
from __future__ import annotations
from typing import Protocol, Any, List, Optional, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError

if TYPE_CHECKING:
    from src.graph import WorkloadGraph
    from src.config.mapping import MapInfo


@runtime_checkable
class GraphPass(Protocol):
    name: str
    depends_on: List[str]
    def run(self, G: 'WorkloadGraph', config: 'PassConfig') -> 'WorkloadGraph': ...

@dataclass
class PassConfig:
    """Configuration passed to all optimization passes."""
    op_removal_spec: Optional[Any] = None
    op_fusion_spec:  Optional[Any] = None
    rsrc_spec:       Optional[Any] = None

    @staticmethod
    def from_mapinfo(mapinfo: 'MapInfo') -> 'PassConfig':
        return PassConfig(
            op_removal_spec=mapinfo.op_removal_spec,
            op_fusion_spec=mapinfo.op_fusion_spec,
            rsrc_spec=mapinfo.rsrc_spec,
        )

class PassPipeline:
    def __init__(self):
        self.passes: List[GraphPass] = []

    def add(self, p: GraphPass) -> 'PassPipeline':
        if any(existing.name == p.name for existing in self.passes):
            raise ValueError(
                f"PassPipeline: duplicate pass name {p.name!r}")
        self.passes.append(p)
        return self

    def validate(self) -> None:
        """Check every declared dependency resolves to a registered pass.
        Called eagerly by :meth:`run`; callers may invoke it themselves to
        surface configuration errors before the first graph is built."""
        names = {p.name for p in self.passes}
        for p in self.passes:
            for dep in getattr(p, 'depends_on', []) or []:
                if dep not in names:
                    raise ValueError(
                        f"PassPipeline: pass {p.name!r} depends on "
                        f"unknown pass {dep!r}")

    def _ordered_passes(self) -> List[GraphPass]:
        # Insertion-order index gives a stable tie-break across topo levels.
        order_idx = {p.name: i for i, p in enumerate(self.passes)}
        by_name = {p.name: p for p in self.passes}
        sorter: TopologicalSorter = TopologicalSorter()
        for p in self.passes:
            sorter.add(p.name, *(getattr(p, 'depends_on', []) or []))
        try:
            sorter.prepare()
        except CycleError as e:
            raise ValueError(
                f"PassPipeline: dependency cycle involving {list(e.args[1])!r}"
            ) from None
        result: List[GraphPass] = []
        while sorter.is_active():
            ready = sorted(sorter.get_ready(), key=order_idx.__getitem__)
            for name in ready:
                result.append(by_name[name])
                sorter.done(name)
        return result

    def run(self, G: 'WorkloadGraph', config: PassConfig) -> 'WorkloadGraph':
        self.validate()
        for p in self._ordered_passes():
            G = p.run(G, config)
        return G
