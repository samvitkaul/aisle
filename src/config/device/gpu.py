

from loguru import logger
from pydantic import BaseModel, PositiveInt, computed_field, model_validator

from ..interconnect import NVLink, PCIe
from .cache import Cache, RegFile
from .compute_core import ComputeCore
from .memory import Memory

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error

class GPUDie(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name           : str
    process        : str
    die_size_sq_mm : float = 0.0
    transistors_B  : float
    architecture   : str = ""
    year           : str = ""

    #Die -> GPC -> TPC -> SM Hierarchy
    gpc_per_die         : PositiveInt
    tpc_per_gpc         : PositiveInt
    sm_per_tpc          : PositiveInt
    cuda_cores_per_sm   : PositiveInt
    tensor_cores_per_sm : PositiveInt

    #per SM
    cuda_core         : ComputeCore
    tensor_core       : ComputeCore
    l1_cache          : Cache
    regfile           : RegFile

    #per TPC (in Hopper, Blackwell, etc.)
    #sm-to-sm-fabric

    #per Die fabric
    l2_cache          : Cache

    def sm_count(self) -> PositiveInt:
        return self.gpc_per_die * self.tpc_per_gpc * self.sm_per_tpc

    @computed_field  # type: ignore[prop-decorator]
    @property
    def instr2core(self) -> dict[str, str]:
        _TBL = {}
        for instr in self.cuda_core.instructions:
            _TBL[instr] = 'cuda_core'
        for instr in self.tensor_core.instructions:
            if instr in _TBL:
                raise ValueError(
                    f"Instruction '{instr}' is defined in both 'cuda_core' and 'tensor_core' "
                    f"on die '{self.name}'. Instructions must be unique across cores. "
                    f"Fix the device YAML config to remove the duplicate."
                )
            _TBL[instr] = 'tensor_core'
        return _TBL

class GPU(BaseModel, extra='forbid', populate_by_name=True, frozen=True):
    name            : str
    form_factor     : str = ""
    die             : GPUDie
    memory          : Memory
    pcie            : PCIe
    nvlink          : NVLink

    die_count       : PositiveInt
    mem_count       : PositiveInt
    active_sm_count : PositiveInt | None = None

    @model_validator(mode='before')
    @classmethod
    def default_and_validate_x(cls, values):
        if isinstance(values, dict):
            die = values.get('die', {})
            if isinstance(die, GPUDie):
                tot_sm_count = values.get('die_count', 1) * die.sm_count()
            elif isinstance(die, dict):
                gpc = die.get('gpc_per_die', 1)
                tpc = die.get('tpc_per_gpc', 1)
                sm  = die.get('sm_per_tpc', 1)
                tot_sm_count = values.get('die_count', 1) * gpc * tpc * sm
            else:
                return values
            asc = values.get('active_sm_count')
            if asc is None:
                values['active_sm_count'] = tot_sm_count
            elif asc > tot_sm_count:
                raise ValueError(f'Constraint: active_sm_count({asc}) <= tot_sm_count({tot_sm_count})')
        return values

    def num_cores(self, core_type: str, active: bool = True) -> PositiveInt:
        ccount = self.active_sm_count if active else self.die_count * self.die.sm_count()
        if ccount is None:  # guaranteed by model_validator, but guard for -O safety
            raise RuntimeError("active_sm_count is None — model_validator should have set it")
        if core_type == 'cuda_core':
            core_count = ccount * self.die.cuda_cores_per_sm
        elif core_type == 'tensor_core':
            core_count = ccount * self.die.tensor_cores_per_sm
        else:
            raise ValueError(f'core_type({core_type}) can only be "cuda_core|tensor_core" ')

        return core_count

    def frequency(self, core_type: str, units = 'GHz'):
        if core_type == 'cuda_core':
            core = self.die.cuda_core
        elif core_type == 'tensor_core':
            core = self.die.tensor_core
        else:
            raise ValueError(f'core_type({core_type}) can only be "cuda_core|tensor_core" ')
        return core.frequency.get(units)

    def peak_ops_per_cycle(self, instr: str, prec: str, strict: bool = False) -> tuple[str, float]:
        core_type = self.die.instr2core[instr]
        if core_type == 'cuda_core':
            core = self.die.cuda_core
            fallback_core_type = 'tensor_core'
            fallback_core = self.die.tensor_core
        elif core_type == 'tensor_core':
            core = self.die.tensor_core
            fallback_core_type = 'cuda_core'
            fallback_core = self.die.cuda_core
        else:
            raise ValueError(f'core_type({core_type}) can only be "cuda_core|tensor_core" ')
        try:
            opc = core.peak_ops_per_cycle(instr, prec, strict)
        except (ValueError, KeyError):
            if instr in fallback_core.instructions:
                core_type = fallback_core_type
                core = fallback_core
                opc = core.peak_ops_per_cycle(instr, prec, strict)
            else:
                raise
        clkname = f'{self.name}.die.{core_type}.frequency'
        return clkname, self.num_cores(core_type) * opc

    def peak_bandwidth(self, interface: str, units = 'GB/s'):
        if interface.upper() not in ['MEM', 'PCIE', 'NVLINK']:
            raise ValueError(f"interface must be one of MEM, PCIE, NVLINK; got {interface!r}")
        bw = {
                'MEM'   : self.memory.peak_bandwidth(units) * self.mem_count,
                'PCIE'  : self.pcie.peak_bandwidth(units),
                'NVLINK': self.nvlink.peak_bandwidth(units),
                }[interface.upper()]

        return bw

    def get_mem_size(self, units='GB'):
        return self.memory.get_size(units) * self.mem_count

    def peak_mem_bytes_per_cycle(self):
        Clk = self.name + '.memory.frequency'
        BpC = self.memory.peak_bytes_per_cycle() * self.mem_count
        return Clk, BpC

    #def default_compiler(self):
    #    """Return the device's default DeviceCompiler.
#
#        Lazy-import avoids a config↔back import cycle (config is the lower
#        layer).
#        """
#        from src.back.device_compiler import DefaultDeviceCompiler
#        return DefaultDeviceCompiler()

    def topology_extents(self) -> tuple[int, int, int]:
        """Leaf-system identity: ``(num_gpu_per_blade, num_blades_per_rack,
        num_racks_per_cluster) = (1, 1, 1)``. Mirrors the composite contract
        defined by :class:`CompositeSystemMixin` so placement helpers can
        dispatch uniformly without a ``getattr`` fallback."""
        return (1, 1, 1)
