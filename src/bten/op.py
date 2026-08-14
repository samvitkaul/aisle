
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Any

from .registry import get_op_registry

#if TYPE_CHECKING:
#    from src.back.kernel_desc import KernelDescriptor

class RemovalReason(Enum):
    NONE = auto()
    USER_SPECIFIED = auto()
    CONSTANT_FOLDED = auto()
    DEAD_ELIMINATED = auto()

@dataclass
class ExecStats:
    pass


class TensorOp:
    op_counter = count(start=1, step=1)

    @classmethod
    def reset_counter(cls):
        cls.op_counter = count(start=1, step=1)

    @classmethod
    def advance_counter_past(cls, n: int):
        cls.op_counter = count(start=n + 1, step=1)

    def __init__(self, name: str, **kwargs):
        self.name    : str            = name
        self.optype  : str | None  = kwargs.get('optype')
        self.attrs   : dict[str, Any] = kwargs.get('attrs', {})
        self.inList  : list[str]      = kwargs.get('inList', [])
        self.outList : list[str]      = kwargs.get('outList', [])
        self.id      : int            = next(self.op_counter)

        #per-op filled by Device Compiler
        #self.kernel_desc: Optional['KernelDescriptor'] = None

        #stats from execution on system/device
        self.resource     : str | None = None
        self.repeat_count : int           = 1
        self.precision    : str | None = None

        #graph optimization related
        self.removal_reason        : RemovalReason = RemovalReason.NONE
        self.fused_in_optimization : bool          = False
        self.fused_with_op         : str | None = None

        #system execution related
        self.exec_stats       : ExecStats = ExecStats()
        self.fused_exec_stats : ExecStats = ExecStats()


    @property
    def removed_in_optimization(self) -> bool:
        return self.removal_reason is not RemovalReason.NONE

    @removed_in_optimization.setter
    def removed_in_optimization(self, value: bool):
        if value:
            self.removal_reason = RemovalReason.USER_SPECIFIED
        else:
            self.removal_reason = RemovalReason.NONE


    def __str__(self):
        s  = f"TensorOp({self.name}) optype={self.optype}, "
        s += f"attrs={self.attrs}, "
        s += f"inList={self.inList}, "
        s += f"outList={self.outList}"
        return s

    def get_info(self):
        assert self.optype is not None, f"TensorOp({self.name}) has no optype"
        return get_op_registry().get_op(self.optype)

    def __call__(self, inT, outT, **kwargs):
        opinfo = self.get_info()

        #do arity check
        in_range  = range(opinfo.min_input, opinfo.max_input+1)
        out_range = range(opinfo.min_output, opinfo.max_output+1)
        assert len(inT) in in_range,   f"#inputs for {self} operator should be in {in_range}, is {len(inT)}"
        assert len(outT) in out_range, f"#outputs for {self} operator should be in {out_range}, is {len(outT)}"

        #do shape inference
        shape_inf_func = opinfo.shape_inf_func
        shape_inf_func(inT, outT, self, **kwargs)


    def fuse_op(self, fused_with_op):
        self.fused_in_optimization = True
        self.fused_with_op         = fused_with_op

    def clone(self) -> 'TensorOp':
        new = object.__new__(TensorOp)
        new.name                  = self.name
        new.optype                = self.optype
        new.attrs                 = self.attrs
        new.inList                = self.inList
        new.outList               = self.outList
        new.id                    = self.id
        #new.kernel_desc           = None
        new.resource              = self.resource
        new.repeat_count          = self.repeat_count
        new.precision             = self.precision
        new.removal_reason        = RemovalReason.NONE
        new.fused_in_optimization = False
        new.fused_with_op         = None
        new.exec_stats            = ExecStats()
        new.fused_exec_stats      = ExecStats()
        return new

def make_op(**kwargs):
    return TensorOp(**kwargs)
