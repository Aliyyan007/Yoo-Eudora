"""Owner preferences — standing settings the owner sets once and the bot
honours forever (persists across restarts).

Examples: "always greet new members in #general", "prefer the main chat
for announcements". Stored flat in data/prefs.json — the event layer reads
them to steer behaviour (see core/events.py welcome_channel usage).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.action_engine.config.settings import settings
from src.action_engine.tools.context import ToolContext

STORE = Path("data/prefs.json")


def _load() -> dict:
    try:
        return json.loads(STORE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save(p: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(p, indent=1))


def get_pref(key: str, default=None):
    """Read a preference — used by event handlers/tools."""
    return _load().get(key, default)


async def set_preference(ctx: ToolContext, key: str, value: str) -> dict:
    """Store an owner preference, e.g. key=welcome_channel value=general.
    Owner-only."""
    if settings.owner_ids and not settings.is_owner(ctx.author_id):
        return {"error": "Only the owner can set preferences."}
    p = _load()
    p[key.strip().lower().replace(" ", "_")] = value.strip()
    _save(p)
    return {"ok": True, "set": {key: value},
            "note": "I'll use this from now on."}


async def get_preferences(ctx: ToolContext) -> dict:
    """List all stored owner preferences."""
    if settings.owner_ids and not settings.is_owner(ctx.author_id):
        return {"error": "Only the owner can view preferences."}
    return {"preferences": _load()}


async def delete_preference(ctx: ToolContext, key: str) -> dict:
    """Remove a stored preference. Owner-only."""
    if settings.owner_ids and not settings.is_owner(ctx.author_id):
        return {"error": "Only the owner can delete preferences."}
    p = _load()
    if key not in p:
        return {"error": f"No preference '{key}'.", "keys": list(p)}
    del p[key]
    _save(p)
    return {"ok": True, "deleted": key}
