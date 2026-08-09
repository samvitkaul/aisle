

import pytest

from src.utils.sym import SymDim, SymExpr, is_divisible_by, is_symbolic, resolve_sym, sym


@pytest.mark.unit
def test_symdim_creation_repr_eq_hash():
    B = SymDim('B')
    S = SymDim('S')
    B2 = SymDim('B')

    assert repr(B) == 'B'
    assert repr(S) == 'S'
    assert B == B2
    assert B != S
    assert hash(B) == hash(B2)
    assert hash(B) != hash(S)

    #can be used as dict keys
    d = {B: 10, S: 20}
    assert d[B2] == 10

@pytest.mark.unit
def test_symdim_arithmetic():
    B = SymDim('B')
    S = SymDim('S')

    expr1 = B + 3
    assert isinstance(expr1, SymExpr)
    assert repr(expr1) == "(B + 3)"

    expr2 = 2 * B
    assert isinstance(expr2, SymExpr)
    assert repr(expr2) == "(2 * B)"

    expr3 = B * S
    assert isinstance(expr3, SymExpr)
    assert repr(expr3) == "(B * S)"

@pytest.mark.unit
def test_subs_full_env():
    B = SymDim('B')
    S = SymDim('S')

    assert B.subs({"B": 32}) == 32

    expr = B * S + 1
    result = expr.subs({"B": 4, "S": 128})
    assert result == 4 * 128 + 1
    assert isinstance(result, int)

@pytest.mark.unit
def test_subs_partial_env():
    B = SymDim('B')
    S = SymDim('S')

    expr = B * S + 1
    result = expr.subs({"S": 128})
    assert is_symbolic(result)

    result2 = result.subs({"B": 4})
    assert result2 == 4 * 128 + 1
    assert isinstance(result2, int)

@pytest.mark.unit
def test_resolve_sym():
    B = SymDim('B')
    data = {
            'A': B * 128 * 4,
            'B': B * 32 * 4,
            'C': 10,
            'D': B * 64,
            'E': B * 27,
            'I': { 'mov': B * 128},
            }
    env = {'B': 32}
    resolved = resolve_sym(data, env)
    assert resolved['A'] == 32 * 128 * 4
    assert resolved['B'] == 32 * 32 * 4
    assert resolved['C'] == 10
    assert resolved['D'] == 32 * 64
    assert resolved['E'] == 32 * 27
    assert resolved['I']['mov'] == 32 * 128
    assert isinstance(resolved['A'], int)
    assert isinstance(resolved['B'], int)
    assert isinstance(resolved['C'], int)
    assert isinstance(resolved['D'], int)
    assert isinstance(resolved['E'], int)
    assert isinstance(resolved['I']['mov'], int)

@pytest.mark.unit
def test_resolve_sym_cache_eq():
    B = SymDim('B')
    data = {
            'A': B * 128 * 4,
            'B': B * 32 * 4,
            'C': 10,
            'D': B * 64,
            'E': B * 27,
            'I': { 'mov': B * 128},
            }
    env = {'B': 32}
    assert resolve_sym(data, env) == resolve_sym(data, env, cache={})


@pytest.mark.unit
def test_is_divisible_by():
    assert is_divisible_by(8, 4) is True
    assert is_divisible_by(7, 4) is False
    assert is_divisible_by(sym('B'), 4) is None
    assert is_divisible_by(sym('B') * 4, 4) is True
    assert is_divisible_by(4 * sym('B'), 4) is True
    assert is_divisible_by(5 * sym('B'), 4) is None
    assert is_divisible_by(4 + sym('B'), 4) is None
    assert is_divisible_by(0, 4) is True

@pytest.mark.unit
def test_is_divisible_by_zero_divisor_raises():
    with pytest.raises(ValueError):
        is_divisible_by(9, 0)

@pytest.mark.unit
def test_is_divisible_by_neg_divisor_raises():
    with pytest.raises(ValueError):
        is_divisible_by(9, -1)

##################
# Deep SymExpr Trees

DEEP_N = 2000
def _deep_left_spine(n: int):
    B = sym('B')
    expr = B
    for _ in range(n):
        expr = expr + 1
    return expr

@pytest.mark.unit
def test_subs_deep_left_spine():
    expr = _deep_left_spine(DEEP_N)
    assert expr.subs({"B": 0}) == DEEP_N
    assert expr.subs({"B": 5}) == DEEP_N + 5

@pytest.mark.unit
def test_repr_deep_left_spine():
    expr = _deep_left_spine(DEEP_N)
    s = repr(expr)
    assert s.startswith('(')
    assert s.endswith(' + 1)')
    assert s.count(' + 1)') == DEEP_N

@pytest.mark.unit
def test_eq_deep_left_spine():
    a = _deep_left_spine(DEEP_N)
    b = _deep_left_spine(DEEP_N)
    assert a == b
    c = _deep_left_spine(DEEP_N -1) + 2 #different leaf at the root
    assert a != c

@pytest.mark.unit
def test_hash_deep_left_spine():
    a = _deep_left_spine(DEEP_N)
    b = _deep_left_spine(DEEP_N)
    assert hash(a) == hash(b)
    #usable as a dict key
    d = {a: 'x'}
    assert d[a] == 'x'

@pytest.mark.unit
def test_sum_over_many_sym_terms():
    B = sym('B')
    terms = [B * i for i in range(1, DEEP_N + 1)]
    s = sum(terms)
    expected = sum(range(1, DEEP_N + 1)) #B=1 -> 1+2+...+DEEP_N
    assert s.subs({'B': 1}) == expected
