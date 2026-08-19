from ...utils.data_types import DataType, str2dt

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Mapping

class Instruction(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True, frozen=True)

    name: str = Field(description="Instruction name (e.g., 'add', 'mac', 'sqrt')")
    throughput: Mapping[DataType, float] = Field(alias="opc", description="Throughput (ops/cycle) by data type")

    @field_validator("throughput", mode="before")
    @classmethod
    def convert_throughput_keys(cls, v: Mapping) -> Mapping[DataType, float]:
        if v is None or not isinstance(v, Mapping):
            raise TypeError(f"throughput must be a mapping, got {type(v).__name__}")

        converted = {}
        for key, value in v.items():
            try:
                dt = str2dt(key)
                if not isinstance(value, (int, float)):
                    raise TypeError(f"throughput[{key}] must be numeric, got {type(value).__name__}")
                converted[dt] = float(value)
            except ValueError as e:
                raise ValueError(f"Invalid data type '{key}': {e}")
        return converted

    def has_precision(self, prec: str) -> bool:
        try:
            dt = str2dt(prec)
            return dt in self.throughput
        except ValueError:
            return False

    def get_throughput(self, prec: str) -> float:
        dt = str2dt(prec)
        if dt not in self.throughput:
            available = [str(dt) for dt in self.throughput.keys()]
            raise KeyError(
                f"Instruction '{self.name}' does not support precision '{prec}'. "
                f"Available: {available}"
            )
        return float(self.throughput[dt])
