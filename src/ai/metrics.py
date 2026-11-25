"""Tiny in-process operational metrics registry.

The system previously had rich logs but almost no quantitative feedback: it
was impossible to answer basic operational questions like "how many replies
were suppressed?", "how often did proactive sends lose the reservation?", or
"how many LLM attempts did one degraded minute produce?" without parsing logs.

This registry is intentionally dependency-free and low-overhead. Counters are
thread-safe, names are dot-separated, and snapshots can be logged or exposed
by future health/debug surfaces.
"""
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

_LOCK = threading.RLock()
_COUNTERS = defaultdict(int)          # name -> count
_GAUGES = {}                          # name -> float
_STARTED_AT = time.time()


def incr(name: str, amount: int = 1, **labels) -> None:
    """Increment a counter. Small cardinality labels are encoded into the key."""
    if labels:
        suffix = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        name = f"{name}{{{suffix}}}"
    with _LOCK:
        _COUNTERS[name] += int(amount)


def gauge(name: str, value: float) -> None:
    with _LOCK:
        _GAUGES[name] = float(value)


def counter(name: str) -> int:
    with _LOCK:
        return int(_COUNTERS.get(name, 0))


def snapshot() -> dict:
    """Return a copy of current counters/gauges plus process uptime."""
    with _LOCK:
        return {
            "uptime_s": round(time.time() - _STARTED_AT, 3),
            "counters": dict(_COUNTERS),
            "gauges": dict(_GAUGES),
        }


def reset() -> None:
    """Test hook — clear all counters/gauges."""
    with _LOCK:
        _COUNTERS.clear()
        _GAUGES.clear()


def compact_summary(max_items: int = 24) -> str:
    """One-line human-readable summary for periodic operational logs."""
    snap = snapshot()
    items = sorted(snap["counters"].items(), key=lambda kv: kv[1], reverse=True)[:max_items]
    body = " ".join(f"{k}={v}" for k, v in items)
    return body or "no metrics yet"
