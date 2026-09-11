
from typing import Any

from pydantic import BaseModel

from .knob import BaseUnit, KnobVal
from .mapping import MapInfo as MapInfo
from .parser import parse_config_with_refs as parse_config_with_refs
from .serialize import OutFormat as OutFormat
from .serialize import dump_model as dump_model
from .workload import WLInfo as WLInfo


def _walk_clocks(model: BaseModel, prefix: str, out: dict) -> None:
    for field_name in type(model).model_fields:
        val = getattr(model, field_name)
        if isinstance(val, KnobVal) and val.base == BaseUnit.HERTZ:
            out[f'{prefix}.{field_name}'] = val
        elif isinstance(val, BaseModel):
            _walk_clocks(val, f'{prefix}.{field_name}', out)
        else:
            pass

def _update_clock(obj: Any, parts: list[str], value: Any):
    if not parts:
        if not isinstance(value, str):
            raise TypeError(f'Cannot descend into {type(obj).__name__} for {value}')
        return KnobVal(value)

    head, *tail = parts

    if isinstance(obj, BaseModel):
        old_child = getattr(obj, head)
        new_child = _update_clock(old_child, tail, value)
        return obj.model_copy(update={head: new_child})
    else:
        raise TypeError(f'Cannot descend into {type(obj).__name__} at {head!r}')

def set_clocks(model: BaseModel, clocks: dict[str, int], units='MHz') -> BaseModel:
    model_out = model
    for clkname, clkfreq in clocks.items():
        parts = clkname.split('.')
        if parts[0] != model.name:  # type: ignore[attr-defined]
            raise ValueError(f"{clkname}:: {parts[0]} != {model.name}")  # type: ignore[attr-defined]
        model_out = _update_clock(model_out, parts[1:], f'{clkfreq} {units}')
    return model_out

def find_clocks(model) -> dict[str, 'KnobVal']:
    clocks: dict[str, KnobVal] = {}
    _walk_clocks(model, model.name, clocks)
    return clocks


def reroot_clock(
    leaf_ret: tuple[str, float],
    leaf_name: str,
    new_prefix: str,
) -> tuple[str, float]:
    # Leaf rate methods build clock names rooted at the leaf's own ``self.name``;
    # ``find_clocks`` walks composites by pydantic field name. Re-prefix so the
    # returned key matches the registered key in the composite's clock registry.
    clk, val = leaf_ret
    leaf_prefix = f'{leaf_name}.'
    if not clk.startswith(leaf_prefix):
        raise ValueError(f"clock {clk!r} not rooted at {leaf_name!r}")
    return f'{new_prefix}.{clk[len(leaf_prefix):]}', val


def dedupe_identical_clocks(
    clks: dict[str, 'KnobVal'],
) -> tuple[dict[str, 'KnobVal'], dict[str, str]]:
    """Collapse repeated KnobVal entries with the same value into one rep.

    Highly-elaborated configs emit
    ~1k clock entries with only a handful of unique frequency values.
    For LCM / cost-dict purposes only one representative per unique value
    is needed.

    Returns:
        (deduped, alias_map)
          deduped   : dict mapping the *first* dotted-name observed per
                      unique value to its KnobVal.
          alias_map : dict mapping every original dotted-name to the
                      canonical (representative) dotted-name.  Includes
                      identity entries for the representatives themselves.

    Equality is based on the value in canonical Hz (for HERTZ knobs) so
    `'0.6 GHz'` and `'600 MHz'` collapse to the same group.  Non-HERTZ
    knobs are passed through unchanged (no dedup attempted).
    """
    canonical_by_key: dict[tuple, str] = {}
    deduped: dict[str, KnobVal] = {}
    alias_map: dict[str, str] = {}
    for clkname, kv in clks.items():
        # Use Hz value + per_second flag as the dedup key for HERTZ knobs;
        # fall through to identity for everything else.
        key: tuple
        if kv.base == BaseUnit.HERTZ:
            key = ("HERTZ", kv.get(units="Hz"), kv.per_second)
        else:
            key = ("OTHER", id(kv))
        if key not in canonical_by_key:
            canonical_by_key[key] = clkname
            deduped[clkname] = kv
        alias_map[clkname] = canonical_by_key[key]
    return deduped, alias_map
