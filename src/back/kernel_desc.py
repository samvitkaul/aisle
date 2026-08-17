from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from src.utils.sym import SymExpr

# SymExpr | int — symbolic before resolve_sym(), concrete after.
SymInt = Union[int, SymExpr] #noqa: UP007


@dataclass
class KernelDescriptor:
    """Per-op work descriptor produced by a DeviceCompiler.

    Populated once per op by ``DeviceCompiler.lower(op, ins, outs)`` and
    consumed by ``ExecSystem.get_compute_ticks`` / ``get_mem_ticks`` to
    convert work units into device cycles via the device's rate model
    (``peak_ops_per_cycle`` / ``peak_mem_bytes_per_cycle``).

    Fields are SymExpr | int before symbol_env resolution and pure int
    after. Use ``resolve_sym(kd, env)`` to concretize a symbolic descriptor.

    Schema is expected to evolve holistically as new consumers appear
    (tiling, buffer plans, MLIR refs, vendor-supplied static profiles).
    No escape-hatch dict — when the schema needs to change, it changes.
    """
    # Compute work
    instrs:         dict[str, SymInt]       = field(default_factory=dict)
    # Memory work
    reads:          SymInt                  = 0
    writes:         SymInt                  = 0
    # PIM-internal weight reads: bytes serviced by the device's internal
    # (PIM) bandwidth tier rather than external DDR. Populated by
    # devices that model an internal-vs-external split
    # Devices without a PIM tier should leave this at 0; the rate model
    # falls back to the external `reads`/`writes` path only.
    #pim_reads:      SymInt                  = 0
    # I/O accounting (consumed by reporting; not used by rate model)
    in_param_count: SymInt                  = 0
    in_act_count:   SymInt                  = 0
    out_act_count:  SymInt                  = 0
