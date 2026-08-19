from ...utils.data_types import DataType, str2dt, dt_fallbacks

from ..knob       import KnobVal
from .instruction import Instruction

from pydantic import BaseModel, ConfigDict, field_validator
from typing import Mapping, Any
from loguru import logger

INFO = logger.info
DEBUG = logger.debug
ERROR = logger.error
WARNING = logger.warning


class ComputeCore(BaseModel):
    """
    Represents a compute core (processing unit) with supported instructions.

    A compute core has a frequency and a set of supported instructions,
    each with precision-dependent throughput specifications.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    name: str = "Compute Core"
    frequency: KnobVal
    instructions: Mapping[str, Instruction]

    @field_validator("instructions", mode="before")
    @classmethod
    def coerce_instructions(cls, v: Mapping[str, Any]) -> Mapping[str, Instruction]:
        if v is None or not isinstance(v, Mapping):
            raise TypeError(f"instructions error: expected Mapping, got {type(v).__name__}")

        result = {}
        for key, value in v.items():
            # Skip the 'name' field that's injected at ISA level by inject_names=True
            if key == 'name':
                continue

            # Remaining keys are precision->throughput mappings
            # Build Instruction with separated name and throughput fields
            instruction_obj = Instruction.model_validate(value)
            result[key] = instruction_obj

        return result

    def get_instr(self, instr: str) -> Instruction:
        """Get an instruction by name."""
        try:
            return self.instructions[instr]
        except KeyError as e:
            raise KeyError(
                f"instruction {instr!r} not found in ComputeCore {self.name!r}; "
                f"known={sorted(self.instructions.keys())}"
            ) from e

    def handle_missing_precision(self, instr: Instruction, prec: str) -> DataType:
        """
        Find a compatible precision when the requested one is not available.

        Uses fallback chains defined in DataType to find equivalent precision.
        """
        dt = str2dt(prec)
        if dt in instr.throughput:
            return dt

        for alt in dt_fallbacks(dt):
            if alt in instr.throughput:
                return alt

        available = [str(dt) for dt in instr.throughput.keys()]
        raise ValueError(
            f"Missing precision {prec!r} for instruction={instr.name!r} on "
            f"core={self.name!r}. Available: {available}"
        )

    def peak_ops_per_cycle(self, instr: str, prec: str, strict: bool = False) -> float:
        """
        Get peak operations per cycle for instruction at given precision.

        If the exact precision isn't available, tries to find a compatible one
        using the DataType fallback chain.
        """
        xi = self.get_instr(instr)
        try:
            return xi.get_throughput(prec)
        except KeyError:
            if strict:
                return 0.0
            else:
                dt_up = self.handle_missing_precision(xi, prec)
                dt_up_str = str(dt_up)
                opc = xi.get_throughput(dt_up_str)
                WARNING(
                        f"Missing precision '{prec}' for instruction '{instr}'; "
                        f"using compatible precision '{dt_up_str}'"
                        )
                return opc

    def peak_flops(
        self,
        instr: str,
        prec: str,
        *,
        units: str = "TFLOPS",
        mul_factor: int = 1
    ) -> float:
        """
        Get peak floating-point operations per second.

        Args:
            instr: Instruction name
            prec: Precision (data type)
            units: Output unit ('TFLOPS', 'GFLOPS', 'MFLOPS', etc.)
            mul_factor: Multiplier for ops (e.g., 2 for MAC which does 2 ops)

        Returns:
            Peak FLOPS in requested units
        """
        ops_per_cycle = self.peak_ops_per_cycle(instr, prec) * float(mul_factor)
        flops = ops_per_cycle * self.frequency.get(units='Hz')
        res = KnobVal(f'{flops} FLOPS')
        return res.get(units=units)

