
from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

_ENABLED: bool = os.environ.get('AISLE_PROFILE') == '1'
_STATS: dict[str, list[float]] = defaultdict(list)

def is_enabled() -> bool:
    return _ENABLED

@contextmanager
def bracket(label: str) -> Iterator[None]:
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _STATS[label].append(time.perf_counter() - t0)


def get_stats() -> dict[str, tuple[int, float]]:
    """Snapshot: label -> (calls, tot_secs)"""
    return {k: (len(v), sum(v)) for k,v in _STATS.items()}

def get_samples() -> dict[str, list[float]]:
    """per-call sample lists for p99/mean computation"""
    return {k: list(v) for k,v in _STATS.items()}

def reset() -> None:
    _STATS.clear()
