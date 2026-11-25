"""Adaptive engagement persistence tests."""
import time

from src.ai import conversational_initiative as ci


def _isolate(monkeypatch):
    store = {}
    monkeypatch.setattr(ci.mem, "load_server_state", lambda key: store.get(key))
    monkeypatch.setattr(ci.mem, "save_server_state", lambda key, value: store.__setitem__(key, dict(value)))
    monkeypatch.setattr(ci, "_adaptive_cache", {})
    return store


def test_adaptive_profile_survives_restart(monkeypatch):
    store = _isolate(monkeypatch)
    analyzer = ci.UserEngagementAnalyzer()

    analyzer.record_bot_message_to_user("u1")
    time.sleep(0.002)
    analyzer.record_user_message("u1", "hello this is a reasonably long answer")

    key = ci._ADAPT_KEY_PREFIX + "u1"
    assert key in store
    assert store[key]["samples"] == 1
    assert store[key]["response_samples"] == 1
    assert store[key]["length_ewma"] > 10

    # Simulate restart/persona rotation: in-process windows are gone, but the
    # aggregate server-state profile still informs warmth and verbosity.
    monkeypatch.setattr(ci, "_adaptive_cache", {})
    ci._adaptive_profile("u1")  # off-loop hydration, as record_user_message does
    restarted = ci.UserEngagementAnalyzer()
    stats = restarted.get_user_stats("u1")
    assert stats["msg_count"] == 1
    assert stats["avg_msg_length"] > 10
    assert stats["avg_response_time"] > 0


def test_hot_path_lookup_never_does_storage_io(monkeypatch):
    monkeypatch.setattr(ci, "_adaptive_cache", {})

    def _boom(key):
        raise AssertionError("hot path performed storage IO")

    monkeypatch.setattr(ci.mem, "load_server_state", _boom)
    analyzer = ci.UserEngagementAnalyzer()
    assert analyzer.get_engagement_level("uncached-user") == "minimal"
    assert analyzer.get_avg_msg_length("uncached-user") == 0.0


def test_adaptive_profile_ewma_updates_without_raw_text(monkeypatch):
    store = _isolate(monkeypatch)
    analyzer = ci.UserEngagementAnalyzer()
    analyzer.record_user_message("u2", "short")
    analyzer.record_user_message("u2", "a much longer message with substance")
    ci._flush_adaptive("u2", force=True)

    blob = store[ci._ADAPT_KEY_PREFIX + "u2"]
    assert blob["samples"] == 2
    assert blob["length_ewma"] > len("short")
    assert "short" not in str(blob)
    assert "substance" not in str(blob)
