"""Operational metrics registry tests."""
from src.ai import metrics


def test_metrics_counters_gauges_and_summary():
    metrics.reset()
    metrics.incr("demo.event", amount=2, kind="test")
    metrics.gauge("demo.depth", 3.5)

    snap = metrics.snapshot()
    assert snap["counters"]["demo.event{kind=test}"] == 2
    assert snap["gauges"]["demo.depth"] == 3.5
    assert "demo.event" in metrics.compact_summary()
    metrics.reset()
