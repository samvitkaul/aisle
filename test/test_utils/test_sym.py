

import pytest

from src.utils.sym import SymDim, SymExpr, sym, is_symbolic, is_divisible_by


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

