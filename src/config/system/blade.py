
from ..device.gpu import GPU
from ..device.memory import Memory
from .cpu import CPU
from .nvme import NVME
from .nic import NIC
from ..interconnect import _BladeFabricUnion
from . import CompositeSystemMixin

from pydantic import BaseModel, PositiveInt, model_validator
from typing import ClassVar, Optional
from loguru   import logger

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error
WARNING = logger.warning

class Blade(CompositeSystemMixin, BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    _child_attr: ClassVar[str] = 'gpu'
    _count_attr: ClassVar[str] = 'num_gpu'

    name             : str
    gpu              : GPU
    host             : CPU
    host_mem         : Memory
    nvme             : NVME
    nic              : NIC
    num_gpu          : PositiveInt
    num_host         : PositiveInt
    num_host_mem     : PositiveInt
    num_nvme         : PositiveInt
    num_nics         : PositiveInt
    # Task 039 N3: discriminated-union routes raw dicts via ``fabric_kind``.
    gpu_interconnect : Optional[_BladeFabricUnion] = None

    @model_validator(mode='after')
    def _require_fabric_for_distributed_blade(self) -> 'Blade':
        if self.num_gpu > 1 and self.gpu_interconnect is None:
            raise ValueError(
                f"Blade {self.name!r}: num_gpu={self.num_gpu} > 1 requires "
                f"gpu_interconnect (scaleup fabric); see config/nvidia/blade_fabric.yml "
                f"for reference NVSwitch fabrics.")
        return self

    def get_host_memory_size(self, units="GB"):
        return self.host_mem.get_size(units) * self.num_host_mem

    def get_host_memory_peak_bandwidth(self, units='GB/s'):
        return self.host_mem.peak_bandwidth(units) * self.num_host_mem

    def get_gpu_peak_bandwidth(self, interface: str, units: str='GB/s'):
        return self.gpu.peak_bandwidth(interface, units) * self.num_gpu


