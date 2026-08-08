
"""Lightweight symbolic integral/fractional types for runtime configurable size manipulations"""

from __future__ import annotations

from typing import Union


class _PySymExpr:
    """
      Pure Python Fallback -- compound symbolic expression tree

      Keep in sync with src/utils/_sym_c.pyx -- algo changes MUST land in both files
      test/test_sym_c_parity.py guards this.
    """

    __slots__ = ('_hash', 'left', 'op', 'right')

    def __init__(self, op: str, left, right):
        self.op     = op
        self.left   = left
        self.right  = right
        self._hash : int | None = None

    # Arithmetic -----------------------------------------------

    def __add__(self, rhs):
        if isinstance(rhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('+', self, rhs)
        return NotImplemented

    def __sub__(self, rhs):
        if isinstance(rhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('-', self, rhs)
        return NotImplemented

    def __mul__(self, rhs):
        if isinstance(rhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('*', self, rhs)
        return NotImplemented

    def __radd__(self, lhs):
        if isinstance(lhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('+', lhs, self)
        return NotImplemented

    def __rsub__(self, lhs):
        if isinstance(lhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('-', lhs, self)
        return NotImplemented

    def __rmul__(self, lhs):
        if isinstance(lhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('*', lhs, self)
        return NotImplemented

    def __floordiv__(self, rhs):
        if isinstance(rhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('//', self, rhs)
        return NotImplemented

    def __mod__(self, rhs):
        if isinstance(rhs, (int, SymDim, _PySymExpr)):
            return _PySymExpr('%', self, rhs)
        return NotImplemented

    def __neg__(self):
        return _PySymExpr('*', -1, self)

    # Substitution -----------------------------------------------
    def subs(self,
             env : dict[str, int],
             cache: dict | None = None,
             ) -> int | _PySymExpr | SymDim:
        """
          Iterative post order walk, so deep left-spines (e.g. sum()
          over high fan in symbolic terms) don't blow up the Python
          recursion stack. When 'cache' is provided, it IS the memo -
          subtree hits across calls are automatic. Caller owns lifetime
          of cache-key nodes
        """
        if cache is not None and id(self) in cache:
            return cache[id(self)]

        memo : dict[int, int | _PySymExpr | SymDim] = \
                {} if cache is None else cache

        work : list[tuple[_PySymExpr, bool]] = [(self, False)]
        while work:
            node, expanded = work.pop()
            if id(node) in memo:
                continue
            if not expanded:
                work.append((node, True))
                for child in (node.left, node.right):
                    if isinstance(child, _PySymExpr) and id(child) not in memo:
                        work.append((child, False))
            else:
                left = memo[id(node.left)] if isinstance(node.left, _PySymExpr) else _subs_val(node.left, env)
                right = memo[id(node.right)] if isinstance(node.right, _PySymExpr) else _subs_val(node.right, env)
                if isinstance(left, int) and isinstance(right, int):
                    memo[id(node)] = _apply_op(node.op, left, right)
                elif left is node.left and right is node.right:
                    memo[id(node)] = node
                else:
                    memo[id(node)] = _PySymExpr(node.op, left, right)
        return memo[id(self)]

    # Comparison/Hashing -----------------------------------------------
    def __eq__(self, rhs):
        if not isinstance(rhs, _SYMEXPR_TYPES):
            return NotImplemented
        work: list[tuple[object, object]] = [(self, rhs)]
        while work:
            a, b = work.pop()
            if isinstance(a, _SYMEXPR_TYPES) and isinstance(b, _SYMEXPR_TYPES):
                if a is b:
                    continue
                if a.op != b.op:
                    return False
                work.append((a.left, b.left))
                work.append((a.right, b.right))
            elif isinstance(a, _SYMEXPR_TYPES) or isinstance(b, _SYMEXPR_TYPES):
                return False
            else:
                if a != b:
                    return False
        return True

    def __hash__(self):
        if self._hash is not None:
            return self._hash

        memo : dict[int, int] = {}
        work : list[tuple[_PySymExpr, bool]] = [(self, False)]
        while work:
            node, expanded = work.pop()
            if node._hash is not None:
                memo[id(node)] = node._hash
                continue
            if id(node) in memo:
                continue
            if not expanded:
                work.append((node, True))
                for child in (node.left, node.right):
                    if isinstance(child, _PySymExpr) and child._hash is None and id(child) not in memo:
                        work.append((child, False))
            else:
                lh = memo[id(node.left)] if isinstance(node.left, _PySymExpr) else _hash_val(node.left)
                rh = memo[id(node.right)] if isinstance(node.right, _PySymExpr) else _hash_val(node.right)
                h = hash((node.op, lh, rh))
                memo[id(node)] = h
                node._hash = h
        return self._hash

    def __repr__(self):
        memo : dict[int, int] = {}
        work : list[tuple[_PySymExpr, bool]] = [(self, False)]
        while work:
            node, expanded = work.pop()
            if id(node) in memo:
                continue
            if not expanded:
                work.append((node, True))
                for child in (node.left, node.right):
                    if isinstance(child, _PySymExpr) and id(child) not in memo:
                        work.append((child, False))
            else:
                ls = memo[id(node.left)] if isinstance(node.left, _PySymExpr) else _repr_val(node.left)
                rs = memo[id(node.right)] if isinstance(node.right, _PySymExpr) else _repr_val(node.right)
                memo[id(node)] = f'({ls} {node.op} {rs})'
        return memo[id(self)]

    def __bool__(self):
        raise TypeError("Cannot evaluate truth value of a symbolic expression")

    def __reduce__(self):
        return (_PySymExpr, (self.op, self.left, self.right))


#Public SymExpr binding: Prefer Cython when available...
try:
    from src.utils._sym_c import SymExpr as _SymExprC
    SymExpr = _SymExprC
    _USING_C_EXT = True
except ImportError:
    SymExpr = _PySymExpr
    _USING_C_EXT = False

_SYMEXPR_TYPES: tuple = (
        (_PySymExpr, SymExpr) if SymExpr is not _PySymExpr else (_PySymExpr,)
)


class SymDim:
    """
      A named symbolic dimension value, e.g. ``SymDim(B)``
    """

    __slots__ = ('name',)

    def __init__(self, name: str):
        self.name = name

    # Arithmetic -----------------------------------------------
    def __add__(self, rhs):
        if isinstance(rhs, (int, SymDim, SymExpr)):
            return SymExpr('+', self, rhs)
        return NotImplemented

    def __sub__(self, rhs):
        if isinstance(rhs, (int, SymDim, SymExpr)):
            return SymExpr('-', self, rhs)
        return NotImplemented

    def __mul__(self, rhs):
        if isinstance(rhs, (int, SymDim, SymExpr)):
            return SymExpr('*', self, rhs)
        return NotImplemented

    def __radd__(self, lhs):
        if isinstance(lhs, (int, SymDim, SymExpr)):
            return SymExpr('+', lhs, self)
        return NotImplemented

    def __rsub__(self, lhs):
        if isinstance(lhs, (int, SymDim, SymExpr)):
            return SymExpr('-', lhs, self)
        return NotImplemented

    def __rmul__(self, lhs):
        if isinstance(lhs, (int, SymDim, SymExpr)):
            return SymExpr('*', lhs, self)
        return NotImplemented

    def __floordiv__(self, rhs):
        if isinstance(rhs, (int, SymDim, SymExpr)):
            return SymExpr('//', self, rhs)
        return NotImplemented

    def __mod__(self, rhs):
        if isinstance(rhs, (int, SymDim, SymExpr)):
            return SymExpr('%', self, rhs)
        return NotImplemented

    def __neg__(self):
        return SymExpr('*', -1, self)

    # Substitution -----------------------------------------------
    def subs(self,
             env : dict[str, int],
             cache: dict | None = None,
             ) -> int | SymDim:
        return env.get(self.name, self)

    # Comparison/Hashing -----------------------------------------------
    def __eq__(self, rhs):
        if isinstance(rhs, SymDim):
            return self.name == rhs.name
        return NotImplemented

    def __hash__(self):
        return hash(('SymDim', self.name))

    def __repr__(self):
        return self.name

    def __bool__(self):
        raise TypeError("Cannot evaluate truth value of a symbolic dimension")

# Type Alias
Dim = Union[int, SymDim]

#Factory
def sym(name: str) -> SymDim:
    return SymDim(name)

#Helpers
_OPS = {
        '+':  lambda a, b: a + b,
        '-':  lambda a, b: a - b,
        '*':  lambda a, b: a * b,
        '//': lambda a, b: a // b,
        '%':  lambda a, b: a % b,
}

def _apply_op(op: str, left: int, right: int) -> int:
    return _OPS[op](left, right)

def _subs_val(v, env: dict[str, int], cache: dict | None = None):
    if isinstance(v, int):
        return v
    if isinstance(v, SymDim):
        return v.subs(env)
    if isinstance(v, SymExpr):
        return v.subs(env, cache=cache)
    return v #pragma: no cover

def _hash_val(v):
    return hash(v)

def _repr_val(v):
    return repr(v)

def is_symbolic(v) -> bool:
    return isinstance(v, (SymDim, SymExpr))

def is_divisible_by(expr, n: int) -> bool | None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f'is_divisible_by: n must be a positive int; got {type(n).__name__}={n!r}')
    if n < 0:
        raise ValueError(f'is_divisible_by: n must be a positive int; got {n}')
    if isinstance(expr, int):
        return expr % n == 0
    if isinstance(expr, SymExpr) and expr.op == '*':
        if isinstance(expr.right, int) and expr.right % n == 0:
            return True
        if isinstance(expr.left, int) and expr.left % n == 0:
            return True
    return None


###########################################################
#from dataclasses import fields, is_dataclass, replace

#def _subs_value(v, env: Dict[str, int], cache=None):
#    if isinstance(v, (SymDim, SymExpr)):
#        return v.subs(env) if cache is None else v.subs(env, cache=cache)
#    if isinstance(v, dict):
#        return {k: _subs_value(val, env, cache) for k, val in v.items()}
#    if isinstance(v, list):
#        return [_subs_value(item, env, cache) for item in v]
#    if is_dataclass(v) and not isinstance(v, type):
#        updates =  {f.name: _subs_value(getattr(v, f.name), env, cache) for f in fields(v)}
#        return replace(v, **updates)
#    return v

def resolve_sym(data, env: dict[str, int], cache=None):
    return _subs_val(data, env, cache)
