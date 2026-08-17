"""Mechanism-level tests for the PyTorch->ONNX kwarg alias resolution
implemented in `src/front/functional.py::_canonicalize_kwargs` and the
registry-time alias validation in `src/bten/registry.py::register_ops`.

  * pure-alias path (no warning, key renamed)
  * canonical-passthrough path (DeprecationWarning, value preserved)
  * collision path (TypeError, no warning)
  * neither-supplied path (no warning, sinf default applies later)
  * register-time validation of bogus alias targets and collisions
"""
import warnings

import pytest

from src.bten.registry import (
    OpRegistryEntry,
    TensorOpRegistry,
    get_op_registry,
    register_ops,
)
from src.front.functional import get_op_attrs


# ---------------------------------------------------------------------------
# Per-module registration of a synthetic op so these tests are self-contained
# and don't depend on Softmax/TopK specifics.
# ---------------------------------------------------------------------------
def _noop_sinf(iTList, oTList, op, **kwargs):
    """Placeholder shape-inf func; never invoked from these tests."""
    return


@pytest.fixture(scope="module", autouse=True)
def _register_synth_op():
    """Register a synthetic op '_TestAxisOp' with attrs={'axis'} and
    aliases={'dim':'axis'} on the global registry. Done once per module."""
    optbl = [
        ['_TestAxisOp', 1, 1, 1, 1, _noop_sinf, {'axis'}, {'dim': 'axis'}],
    ]
    register_ops('test', optbl)
    yield


# ---------------------------------------------------------------------------
# get_op_attrs() level — exercises _canonicalize_kwargs in situ
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_alias_resolution_basic():
    """Passing only the PyTorch alias renames to canonical, no warning."""
    kwargs = {'dim': 3}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attrs = get_op_attrs('_TestAxisOp', kwargs)
    assert attrs == {'axis': 3}
    assert kwargs == {}, f"expected dim consumed, got leftover {kwargs!r}"
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], f"expected no DeprecationWarning, got {deprecations!r}"


@pytest.mark.unit
def test_alias_canonical_passthrough_warns():
    """Passing the ONNX canonical name preserves the value AND warns."""
    kwargs = {'axis': 3}
    with pytest.warns(DeprecationWarning) as recw:
        attrs = get_op_attrs('_TestAxisOp', kwargs)
    assert attrs == {'axis': 3}
    assert kwargs == {}
    assert len(recw) == 1
    msg = str(recw[0].message)
    assert "'dim'" in msg and "'axis'" in msg, f"warning text missing names: {msg!r}"


@pytest.mark.unit
def test_alias_collision_raises_typeerror():
    """Passing both keys raises TypeError naming both, with no warning."""
    kwargs = {'dim': 3, 'axis': 5}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError) as exc:
            get_op_attrs('_TestAxisOp', kwargs)
    msg = str(exc.value)
    assert "'dim'" in msg and "'axis'" in msg
    assert "_TestAxisOp" in msg
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], f"collision must not warn, got {deprecations!r}"


@pytest.mark.unit
def test_alias_default_when_neither_supplied():
    """Neither key supplied -> attrs empty, no warning. Sinf default applies."""
    kwargs = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attrs = get_op_attrs('_TestAxisOp', kwargs)
    assert attrs == {}
    assert kwargs == {}
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []


# ---------------------------------------------------------------------------
# Register-time validation — exercises register/register_ops directly on a
# scratch registry so we don't pollute the global one with broken rows.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_register_time_invalid_alias_target():
    """Alias value must name a key in declared attrs set."""
    reg = TensorOpRegistry()
    with pytest.raises(ValueError) as exc:
        reg.register(OpRegistryEntry(
            opname='_BadTarget', group='test',
            max_input=1, min_input=1,
            max_output=1, min_output=1, shape_inf_func=_noop_sinf,
            attrs=frozenset({'axis'}), aliases={'dim': 'nope'},
        ))
    msg = str(exc.value)
    assert "'nope'" in msg
    assert "_BadTarget" in msg


@pytest.mark.unit
def test_register_time_alias_canonical_namespace_collision():
    """Alias key must not collide with an existing canonical attr name."""
    reg = TensorOpRegistry()
    with pytest.raises(ValueError) as exc:
        reg.register(OpRegistryEntry(
            opname='_BadCollision', group='test',
            max_input=1, min_input=1,
            max_output=1, min_output=1, shape_inf_func=_noop_sinf,
            attrs=frozenset({'axis', 'dim'}), aliases={'dim': 'axis'},
        ))
    msg = str(exc.value)
    assert "'dim'" in msg
    assert "_BadCollision" in msg


@pytest.mark.unit
def test_register_ops_unsupported_arity():
    """register_ops raises for rows that are neither 6, 7, nor 8 long."""
    optbl = [
        ['_BadArity', 1, 1, 1, 1, _noop_sinf, {'axis'}, {'dim': 'axis'}, 'extra'],
    ]
    with pytest.raises(ValueError) as exc:
        register_ops('test', optbl)
    assert "_BadArity" in str(exc.value)


@pytest.mark.unit
def test_softmax_registry_has_alias():
    """End-to-end smoke: the real Softmax registration carries dim->axis."""
    opinfo = get_op_registry().get_op('Softmax')
    assert opinfo.aliases == {'dim': 'axis'}


@pytest.mark.unit
def test_topk_registry_has_alias():
    """End-to-end smoke: the real TopK registration carries dim->axis."""
    opinfo = get_op_registry().get_op('TopK')
    assert opinfo.aliases == {'dim': 'axis'}
