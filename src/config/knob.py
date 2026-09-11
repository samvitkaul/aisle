import re
from enum import Enum

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class MetricPrefix(Enum):
    """Common SI metric prefixes represented as the power of 10"""
    EXA   =  18
    PETA  =  15
    TERA  =  12
    GIGA  =  9
    MEGA  =  6
    KILO  =  3
    HECTO =  2
    DECA  =  1
    UNIT  =  0
    MILLI = -3
    MICRO = -6
    NANO  = -9
    PICO  = -12
    FEMTO = -15

def convert(from_unit: MetricPrefix, to_unit: MetricPrefix) -> float:
    try:
        exp_from = int(from_unit.value)
        exp_to   = int(to_unit.value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("from_unit and to_unit must be enum members with integer .value") from exc

    return 10.0 ** (exp_from - exp_to)

class BaseUnit(Enum):
    BYTE    = "Byte"
    BIT     = "bit"
    FLOP    = "FLOP"
    OP      = "OP"
    SEC     = "SECOND"
    HERTZ   = "HERTZ"
    NULL    = "NULL"

def parse_unitstr(unit_str: str):
    if unit_str == "":
        raise ValueError("Empty unit string is not allowed")

    per_sec  = False
    us       = unit_str
    if us.endswith(("/s", "ps")):
        per_sec = True
        us      = us[:-2]
    elif us.lower().endswith("/sec"):
        per_sec = True
        us      = us[:-4]

    # prefix
    prefix_map = {
        "E": MetricPrefix.EXA,
        "P": MetricPrefix.PETA,
        "T": MetricPrefix.TERA,
        "G": MetricPrefix.GIGA,
        "M": MetricPrefix.MEGA,
        "K": MetricPrefix.KILO,
        "m": MetricPrefix.MILLI,
        "u": MetricPrefix.MICRO,
        "n": MetricPrefix.NANO,
        "p": MetricPrefix.PICO,
        "f": MetricPrefix.FEMTO,
    }

    prefix = MetricPrefix.UNIT
    tail = us
    if us[:1] in prefix_map:
        prefix = prefix_map[us[:1]]
        tail = us[1:]

    tail_orig = tail
    tail_l    = tail.lower()

    #base unit
    if "byte" in tail_l or "bytes" in tail_l or ("B" in tail_orig and "b" not in tail_orig):
        base_unit = BaseUnit.BYTE
    elif "bit" in tail_l or "bits" in tail_l or ("b" in tail_orig and "B" not in tail_orig):
        base_unit = BaseUnit.BIT
    elif "flop" in tail_l:
        base_unit = BaseUnit.FLOP
    elif "op" in tail_l:
        base_unit = BaseUnit.OP
    elif "hz" in tail_l or 'hertz' in tail_l:
        base_unit = BaseUnit.HERTZ
    else:
        base_unit = BaseUnit.NULL

    return prefix, base_unit, per_sec

def parse_valstr(s: str):
    m = re.match(r"^\s*([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*(\S.*)?$", s)
    if not m:
        raise ValueError(f"could not parse knob value from '{s}'")
    value_str, unit_part = m.group(1), (m.group(2) or "").strip()
    value = float(value_str)
    pref, base, per_second = parse_unitstr(unit_part)
    return value, pref, base, per_second

_PREFIX_TO_ABBR = {
    MetricPrefix.EXA:   "E",
    MetricPrefix.PETA:  "P",
    MetricPrefix.TERA:  "T",
    MetricPrefix.GIGA:  "G",
    MetricPrefix.MEGA:  "M",
    MetricPrefix.KILO:  "K",
    MetricPrefix.UNIT:  "",
    MetricPrefix.MILLI: "m",
    MetricPrefix.MICRO: "u",
    MetricPrefix.NANO:  "n",
    MetricPrefix.PICO:  "p",
    MetricPrefix.FEMTO: "f",
}

_BASE_TO_ABBR = {
    BaseUnit.BYTE:  "B",
    BaseUnit.BIT:   "b",
    BaseUnit.FLOP:  "FLOP",
    BaseUnit.OP:    "OP",
    BaseUnit.SEC:   "s",
    BaseUnit.HERTZ: "Hz",
    BaseUnit.NULL:  "",
}

class KnobVal:
    def __init__(self, value: str):
        _v, _p, _b, _s = parse_valstr(value)
        self.value      : float        = _v
        self.prefix     : MetricPrefix = _p
        self.base       : BaseUnit     = _b
        self.per_second : bool         = _s

    @classmethod
    def __get_pydantic_core_schema__(cls, src_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.no_info_plain_validator_function(cls),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.serialize(),
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    def serialize(self) -> str:
        v = self.value
        v_str = f"{v:g}"
        prefix_abbr = _PREFIX_TO_ABBR.get(self.prefix, "")
        base_abbr   = _BASE_TO_ABBR.get(self.base, "")
        suffix = "/s" if self.per_second else ""
        return f"{v_str} {prefix_abbr}{base_abbr}{suffix}".strip()

    def get(self, units="MHz"):
        _p, _b, _s = parse_unitstr(units)
        if self.base != _b or self.per_second != _s:
            raise ValueError(
                f"Unit mismatch: knob is {self.base.name}{'/s' if self.per_second else ''}, "
                f"requested {_b.name}{'/s' if _s else ''}"
            )
        mul_fac = convert(self.prefix, _p)
        return self.value * mul_fac

    def __repr__(self) -> str:
        v = self.value
        p = self.prefix.name.title()
        b = self.base.name.title()
        x =  f"(Knob: {v:.2f} {p}-{b}"
        if self.per_second:
            x += "/s"
        x += ')'
        return x

    def __str__(self) -> str:
        return self.__repr__()

