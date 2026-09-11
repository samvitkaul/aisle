
import pytest

from src.config.knob import (
    BaseUnit,
    KnobVal,
    MetricPrefix,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "idx,rec",
    enumerate(
    [
        ("100.4 Gbps",  100.4,  MetricPrefix.GIGA, BaseUnit.BIT,   True),
        ("20.4 GBps",    20.4,  MetricPrefix.GIGA, BaseUnit.BYTE,  True),
        ("20.4 TBps",    20.4,  MetricPrefix.TERA, BaseUnit.BYTE,  True),
        ("20.4 TB/s",    20.4,  MetricPrefix.TERA, BaseUnit.BYTE,  True),
        ("20.4 POP/s",   20.4,  MetricPrefix.PETA, BaseUnit.OP,    True),
        ("20.4 GFLOP/s", 20.4,  MetricPrefix.GIGA, BaseUnit.FLOP,  True),
        ("2 MB/s",       2.0,   MetricPrefix.MEGA, BaseUnit.BYTE,  True),
        ("42 bytes",     42.0,  MetricPrefix.UNIT, BaseUnit.BYTE,  False),
        ("100.3 MHz",    100.3, MetricPrefix.MEGA, BaseUnit.HERTZ, False),
    ]),
)
def test_knob(idx, rec):
    s, exp_val, exp_pref, exp_base, exp_ps = rec
    k = KnobVal(s)
    assert pytest.approx(k.value, rel=1e-12) == exp_val
    assert k.prefix is exp_pref
    assert k.base is exp_base
    assert k.per_second is exp_ps
