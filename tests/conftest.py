"""Pytest sandbox for the AI persona simulation suite.

Guarantees:
* no real network — Groq/Discord/D1 are faked or disabled
* no real state — every module writing to data/ is pointed at a tmp dir
* no real secrets — env vars are scrubbed before any src module loads
* deterministic randomness — tests can seed random
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Scrub dangerous env BEFORE src imports ────────────────────────────────────
# config/.env holds REAL tokens — llm.py / d1_client load it at import time.
os.environ["GROQ_API_KEY"] = "fake-key-for-tests"
for _i in range(2, 20):
    os.environ.pop(f"GROQ_API_KEY_{_i}", None)
os.environ.pop("DISCORD_TOKEN", None)
os.environ.pop("DISCORD_TOKEN_ISLA", None)
os.environ.pop("DISCORD_TOKEN_ROWAN", None)
os.environ.pop("FISH_AUDIO_API_KEY", None)
os.environ.pop("CF_D1_ACCOUNT_ID", None)
os.environ.pop("CF_D1_DATABASE_ID", None)
os.environ.pop("CF_D1_API_TOKEN", None)
os.environ["TEXT_CHANNEL_IDS"] = "111111111"
os.environ["ACTION_WORKER_ENABLED"] = os.environ.get("ACTION_WORKER_ENABLED", "1")
os.environ["OWNER_IDS"] = "999000999"


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirect every persistent-state path into a tmp dir.

    Covers BOTH path styles in the codebase: cwd-relative "data/..." and
    repo-root-anchored constants (memory.py, persona/runtime.py).
    """
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Repo-anchored constants
    import src.ai.memory as mem_mod
    monkeypatch.setattr(mem_mod, "_MEMORY_FILE", str(data_dir / "memory.json"))

    import src.persona.runtime as prt
    monkeypatch.setattr(prt, "_DATA_DIR", str(data_dir))
    monkeypatch.setattr(prt, "_REGISTRY_FILE", str(data_dir / "persona_registry.json"))
    monkeypatch.setattr(prt, "_STATE_FILE", str(data_dir / "persona_state.json"))
    monkeypatch.setattr(prt, "_PENDING_DIR", str(data_dir / "pending"))
    monkeypatch.setattr(prt, "_DEFERRED_DIR", str(data_dir / "deferred"))

    import src.ai.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_COOLDOWN_FILE", str(data_dir / "groq_cooldowns.json"))

    # D1 must stay unreachable — force JSON fallback
    import src.ai.d1_client as d1c
    monkeypatch.setattr(d1c, "_ACCOUNT_ID", "")
    monkeypatch.setattr(d1c, "_DATABASE_ID", "")
    monkeypatch.setattr(d1c, "_API_TOKEN", "")
    try:
        import src.ai.d1_memory as d1m
        if hasattr(d1m, "_d1_available"):
            monkeypatch.setattr(d1m, "_d1_available", lambda: False)
    except Exception:
        pass

    # Web search (DuckDuckGo) must never hit the network
    import src.ai.search as search_mod
    monkeypatch.setattr(search_mod, "search", lambda q: "")

    return tmp_path


@pytest.fixture()
async def world(sandbox):
    """A fresh simulated Discord server + wired client per test."""
    from tests.sim.world import SimWorld
    w = await SimWorld.build()
    yield w
    await w.teardown()
