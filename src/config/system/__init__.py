"""Composite system mixin shared by Nvidia composites.

`Blade`, `Rack`, and `Cluster` (and the in-flight Qualcomm composites) all
implement the same six ExecSystem-contract methods on top of a single child
attribute (GPU, Blade, Rack ...) and a corresponding multiplicity attribute
(`num_gpu`, `num_blades`, `num_racks`). The previous code base duplicated
identical `reroot_clock`-style wrappers across three files; this mixin
parameterises them over the child / count attribute names.
"""

from typing import ClassVar


class CompositeSystemMixin:
    """Default ExecSystem-contract implementations for composite systems.

    Subclasses must set:
        _child_attr  : name of the leaf/sub-composite field (e.g. ``'gpu'``)
        _count_attr  : name of the multiplicity field      (e.g. ``'num_gpu'``)
    """

    _child_attr: ClassVar[str]
    _count_attr: ClassVar[str]

    # -- helpers ---------------------------------------------------------
    @property
    def _child(self):
        return getattr(self, self._child_attr)

    @property
    def _count(self) -> int:
        return getattr(self, self._count_attr)

    # -- ExecSystem contract --------------------------------------------
#    def default_compiler(self):
#        leaf = self._child.default_compiler()
#        # Wrap exactly once, at the lowest composite layer above a leaf
#        # device (the leaf's `default_compiler()` returns a per-device
#        # `DeviceCompiler`, not a `CompositeDeviceCompiler`).
#        if isinstance(self._child, CompositeSystemMixin):
#            return leaf
#        from src.back.device_compiler import CompositeDeviceCompiler
#        return CompositeDeviceCompiler(leaf)

    def peak_ops_per_cycle(self, instr: str, prec: str) -> tuple[str, float]:
        from src.config import reroot_clock
        return reroot_clock(self._child.peak_ops_per_cycle(instr, prec),
                            self._child.name,
                            f'{self.name}.{self._child_attr}')  # type: ignore[attr-defined]

    def peak_mem_bytes_per_cycle(self):
        from src.config import reroot_clock
        return reroot_clock(self._child.peak_mem_bytes_per_cycle(),
                            self._child.name,
                            f'{self.name}.{self._child_attr}')  # type: ignore[attr-defined]

    def get_mem_size(self, units: str = 'GB'):
        return self._child.get_mem_size(units) * self._count

    def system_capacity(self) -> int:
        # Leaves (e.g. GPU) lack `system_capacity`; treat them as 1.
        child_cap = getattr(self._child, 'system_capacity', lambda: 1)()
        return self._count * child_cap

    def is_distributed_capable(self) -> bool:
        return self.system_capacity() > 1

    def topology_extents(self) -> tuple[int, int, int]:
        """Return ``(gpus_per_blade, blades_per_rack, racks_per_cluster)``.

        Generic mixin walk: a composite at tier *k* takes its child's
        extents and fills the first ``1`` slot at index >= 1 with its own
        count. Base case (child is a leaf — i.e. NOT a
        :class:`CompositeSystemMixin`) returns ``(self._count, 1, 1)``.
        With the canonical Blade → Rack → Cluster nesting this yields
        ``(G, 1, 1)`` / ``(G, B, 1)`` / ``(G, B, R)`` respectively.

        Note: leaf systems (``GPU``, ``QualcommCard``) define their own
        ``topology_extents() → (1, 1, 1)`` so placement helpers can
        dispatch uniformly (A-20 / R-04-16); the mixin must therefore
        gate on ``isinstance(child, CompositeSystemMixin)`` rather than
        ``hasattr(child, 'topology_extents')``.
        """
        child = self._child
        if isinstance(child, CompositeSystemMixin):
            ext = list(child.topology_extents())
            for i in range(1, 3):
                if ext[i] == 1:
                    ext[i] = int(self._count)
                    return ext[0], ext[1], ext[2]
            raise RuntimeError(
                f'topology_extents: no tier slot left above '
                f'{type(child).__name__} for {type(self).__name__}')
        return int(self._count), 1, 1
