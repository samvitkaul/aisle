
from ..knob import KnobVal

from pydantic import BaseModel, PositiveInt, ConfigDict
from loguru   import logger

INFO    = logger.info
DEBUG   = logger.debug
ERROR   = logger.error
WARNING = logger.warning

class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    name               : str
    technology         : str
    data_bits          : int
    frequency          : KnobVal
    size               : KnobVal
    stacks             : PositiveInt = 1
    data_rate          : PositiveInt = 1
    transfer_rate_GTps : float #redundant, used for quick validation of spec

    #@model_validator(mode='after')
    #def validate_transfer_rate_GTps(self):
    #    GTps = 2 * self.stacks * self.data_rate * self.frequency.get(units="GHz")
    #    check = isclose(self.transfer_rate_GTps, GTps, rel_tol=1e-6, abs_tol=1e-6)
    #    if not check:
    #        raise AssertionError(
    #            f"[MemoryBlockModel({self.name!r})] transfer_rate_GTs ({self.transfer_rate_GTps}) "
    #            f"does not match calculated transfers per second ({GTps:.6f} GT/s)"
    #        )
    #    return self

    def with_frequency(self, freq: 'KnobVal') -> 'Memory':
        """Return a copy with the given frequency (frozen-safe)."""
        return self.model_copy(update={'frequency': freq})

    def get_size(self, /, units="GB"):
        return self.size.get(units=units)

    def get_frequency(self, /, units="MHz"):
        return self.frequency.get(units=units)

    def peak_bytes_per_cycle(self):
        Tpc = 2 * self.stacks * self.data_rate
        bw  = Tpc * self.data_bits / 8
        return bw

    def peak_bandwidth(self, /, units="GB/s"):
        freq_units = units[0] + 'Hz'
        Tps = 2 * self.stacks * self.data_rate * self.frequency.get(freq_units)
        bw  = Tps * self.data_bits / 8
        return bw

