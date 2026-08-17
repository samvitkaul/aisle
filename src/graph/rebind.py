"""SymDim rebinder for ``WorkloadGraph``

A single-pass tensor walk that substitutes ``env`` into every shape
axis of every tensor. Returns a fresh graph via ``clone_for_execute``
Does NOT call any op's ``forward()`` or shape inference.
"""
from typing import Dict

from ..utils.sym import SymDim, SymExpr
from .graph import WorkloadGraph


def _subs_axis(axis, env: Dict[str, int]):
    if isinstance(axis, int):
        return axis
    if isinstance(axis, (SymDim, SymExpr)):
        return axis.subs(env)
    return axis


def rebind_symbolic_dims(G: WorkloadGraph,
                         env: Dict[str, int],
                         /) -> WorkloadGraph:
    """Return a fresh ``WorkloadGraph`` with each SymDim/SymExpr axis
    bound per ``env``. Partial binds are supported (unknown SymDim
    names remain symbolic in the returned graph).

    Raises ``ValueError`` if an ``env`` value is not an ``int`` or if
    rebinding produces a non-positive concrete axis.
    """
    for k, v in env.items():
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"rebind_symbolic_dims: env[{k!r}] must be int, got "
                f"{type(v).__name__}={v!r}"
            )
        if v <= 0:
            raise ValueError(
                f"rebind_symbolic_dims: env[{k!r}]={v} must be positive"
            )

    clone = G.clone_for_execute()

    for tname, t in clone._tensors.items():
        if t.shape is None:
            continue
        new_shape = []
        for i, axis in enumerate(t.shape):
            new_axis = _subs_axis(axis, env)
            if isinstance(new_axis, int) and new_axis <= 0:
                raise ValueError(
                    f"rebind_symbolic_dims: tensor {tname!r} axis {i} "
                    f"resolved to non-positive int {new_axis} under env={env!r}"
                )
            new_shape.append(new_axis)
        t.shape = new_shape

    return clone
