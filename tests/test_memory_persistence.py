"""JSON fallback memory persistence safety tests."""
import json
import os
import threading

from src.ai import memory as memory_mod


def _isolate(monkeypatch, tmp_path):
    path = tmp_path / "memory.json"
    monkeypatch.setattr(memory_mod, "_MEMORY_FILE", str(path))
    monkeypatch.setattr(memory_mod, "_memory_cache", None)
    return path


def test_save_keeps_primary_and_backup(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    memory_mod._save_memory({"users": {"1": {"facts": ["first"]}}})
    monkeypatch.setattr(memory_mod, "_memory_cache", None)
    memory_mod._save_memory({"users": {"1": {"facts": ["second"]}}})

    assert os.path.exists(path)
    assert os.path.exists(str(path) + ".bak")
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["users"]["1"]["facts"] == ["second"]
    with open(str(path) + ".bak", encoding="utf-8") as f:
        assert json.load(f)["users"]["1"]["facts"] == ["first"]


def test_load_recovers_backup_when_primary_missing(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    memory_mod._save_memory({"users": {"1": {"facts": ["remembered"]}}})
    os.remove(path)
    monkeypatch.setattr(memory_mod, "_memory_cache", None)

    loaded = memory_mod._load_memory()
    assert loaded["users"]["1"]["facts"] == ["remembered"]


def test_load_recovers_backup_when_primary_corrupt(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)
    memory_mod._save_memory({"server_state": {"x": 1}})
    with open(path, "w", encoding="utf-8") as f:
        f.write("{corrupt")
    monkeypatch.setattr(memory_mod, "_memory_cache", None)

    loaded = memory_mod._load_memory()
    assert loaded["server_state"]["x"] == 1


def test_concurrent_fact_writes_do_not_lose_facts(monkeypatch, tmp_path):
    path = _isolate(monkeypatch, tmp_path)

    def write_facts(start):
        for i in range(start, start + 4):
            memory_mod.update_user_memory("user", "tester", [f"fact-{i}"])

    threads = [threading.Thread(target=write_facts, args=(n,)) for n in (0, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    monkeypatch.setattr(memory_mod, "_memory_cache", None)
    loaded = memory_mod._load_memory()
    facts = set(loaded["users"]["user"]["facts"])
    assert {f"fact-{i}" for i in range(8)} <= facts
