"""Active-persona runtime — the single source of truth for "who is online
right now", plus the cross-persona plumbing that keeps accounts feeling like
separate people who share a server.

- active()/activate() — the currently-online persona profile. Only ONE is
  active per process at a time (the rotation supervisor owns switching).
- registry — data/persona_registry.json maps persona_id -> discord user_id,
  written each activation. Other personas' ids are how an active account
  recognises "@isla"/replies aimed at an OFFLINE persona.
- pending — data/pending/<pid>.json lists interactions aimed at an offline
  persona (mentions, replies to their messages). Consumed on activation so
  the returning persona can acknowledge naturally — once, never twice
  (handled message ids are recorded).
- state — data/persona_state.json records who's active + when the window
  started, so a restart resumes the same persona's remaining slot instead
  of restarting the clock.

All writes are atomic (tmp + replace) — rotation mid-write can't corrupt
state and a crash leaves the last good file.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from loguru import logger

from .profiles import DEFAULT_ID, PersonaProfile, get_profile

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_REGISTRY_FILE = os.path.join(_DATA_DIR, "persona_registry.json")
_STATE_FILE = os.path.join(_DATA_DIR, "persona_state.json")
_PENDING_DIR = os.path.join(_DATA_DIR, "pending")

# Pending items older than this are treated as context, not replied to —
# replying to a 6h-old message reads as weird.
_PENDING_REPLY_MAX_AGE_S = 45 * 60
# Never dump a huge backlog — keep the most recent few per channel.
_PENDING_REPLY_LIMIT = 3


def _atomic_write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────
#  Active persona
# ─────────────────────────────────────────────────────────────────────────

_active: PersonaProfile = get_profile(DEFAULT_ID)


def active() -> PersonaProfile:
    """The persona currently online."""
    return _active


def activate(profile: PersonaProfile) -> PersonaProfile:
    """Point every persona-dependent system at this profile."""
    global _active
    _active = profile
    # Rebuild prompt modules — consumers read prompts.X lazily at call time.
    try:
        from ..ai import prompts as _prompts
        _prompts.set_persona(profile)
    except Exception as e:
        logger.warning(f"[persona] prompts.set_persona failed: {e}")
    try:
        from ..action_engine.config import prompts as _ae_prompts
        _ae_prompts.set_persona(profile)
    except Exception as e:
        logger.warning(f"[persona] ae prompts.set_persona failed: {e}")
    try:
        from ..ai import d1_memory as _mem
        _mem.set_namespace(profile.id)
    except Exception as e:
        logger.warning(f"[persona] memory namespace failed: {e}")
    logger.info(f"[persona] active → {profile.full_name} ({profile.id})")
    return profile


# ─────────────────────────────────────────────────────────────────────────
#  Registry — persona_id -> discord user identity
# ─────────────────────────────────────────────────────────────────────────

def register_self(user_id: int, display_name: str = "") -> None:
    """Called each activation once the account is ready — records this
    persona's Discord user id so OTHER personas can spot messages aimed
    at this account while it's offline."""
    reg = _read_json(_REGISTRY_FILE, {})
    reg[_active.id] = {"user_id": int(user_id), "name": display_name or _active.name}
    _atomic_write(_REGISTRY_FILE, reg)


def other_persona_ids() -> Dict[int, str]:
    """user_id -> persona_id for every persona EXCEPT the active one."""
    reg = _read_json(_REGISTRY_FILE, {})
    return {
        int(v["user_id"]): pid
        for pid, v in reg.items()
        if pid != _active.id and v.get("user_id")
    }


def own_user_ids() -> List[int]:
    """ALL known persona user ids — used by engagement sweep so any
    persona's stale engagement messages count toward the channel wall."""
    reg = _read_json(_REGISTRY_FILE, {})
    return [int(v["user_id"]) for v in reg.values() if v.get("user_id")]


# ─────────────────────────────────────────────────────────────────────────
#  Rotation state — survives restarts so the window doesn't reset
# ─────────────────────────────────────────────────────────────────────────

def load_rotation_state() -> dict:
    return _read_json(_STATE_FILE, {})


def save_rotation_state(persona_id: str, activated_at: float, seq: int) -> None:
    _atomic_write(_STATE_FILE, {
        "active": persona_id, "activated_at": activated_at, "seq": seq,
    })


def clear_rotation_state() -> None:
    try:
        os.remove(_STATE_FILE)
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────
#  Pending interactions — messages aimed at an OFFLINE persona
# ─────────────────────────────────────────────────────────────────────────

def _pending_path(persona_id: str) -> str:
    return os.path.join(_PENDING_DIR, f"{persona_id}.json")


def add_pending(persona_id: str, *, guild_id: int, channel_id: int,
                message_id: int, author_id: int, author_name: str,
                text: str, kind: str) -> None:
    """Record an interaction aimed at an offline persona. kind: 'mention' |
    'reply' | 'name'. Bounded — capped at 50 entries, oldest dropped."""
    path = _pending_path(persona_id)
    items = _read_json(path, [])
    if not isinstance(items, list):
        items = []
    items.append({
        "guild_id": int(guild_id) if guild_id else 0,
        "channel_id": int(channel_id),
        "message_id": int(message_id),
        "author_id": int(author_id),
        "author_name": author_name,
        "text": (text or "")[:300],
        "kind": kind,
        "ts": time.time(),
        "handled": False,
    })
    _atomic_write(path, items[-50:])


def list_pending(persona_id: str) -> List[dict]:
    items = _read_json(_pending_path(persona_id), [])
    return items if isinstance(items, list) else []


def mark_pending_handled(persona_id: str, message_ids: List[int]) -> None:
    path = _pending_path(persona_id)
    items = _read_json(path, [])
    ids = {int(i) for i in message_ids}
    for it in items:
        if int(it.get("message_id", 0)) in ids:
            it["handled"] = True
    _atomic_write(path, items)


def fresh_pending(persona_id: str) -> List[dict]:
    """Unhandled items young enough to still warrant a reply."""
    now = time.time()
    return [
        it for it in list_pending(persona_id)
        if not it.get("handled") and now - it.get("ts", 0) < _PENDING_REPLY_MAX_AGE_S
    ]


def stale_pending(persona_id: str) -> List[dict]:
    """Unhandled but too old to reply to — context only."""
    now = time.time()
    return [
        it for it in list_pending(persona_id)
        if not it.get("handled") and now - it.get("ts", 0) >= _PENDING_REPLY_MAX_AGE_S
    ]


def pending_reply_cap() -> int:
    return _PENDING_REPLY_LIMIT


# ─────────────────────────────────────────────────────────────────────────
#  Deferred replies — directed messages dropped on daily-cap, answered
#  once the counter resets (or the same persona next rotates in)
# ─────────────────────────────────────────────────────────────────────────

_DEFERRED_DIR = os.path.join(_DATA_DIR, "deferred")
_DEFERRED_MAX_AGE_S = 20 * 3600   # a reply >20h late reads weird — expire
_DEFERRED_LIMIT_PER_CH = 3        # at most this many per channel


def _deferred_path(persona_id: str) -> str:
    return os.path.join(_DEFERRED_DIR, f"{persona_id}.json")


def add_deferred(persona_id: str, *, guild_id: int, channel_id: int,
                 message_id: int, author_id: int, author_name: str,
                 text: str) -> None:
    """Queue a directed message that hit the daily cap. Deduped by
    message_id; capped at 3 per channel / 30 total."""
    path = _deferred_path(persona_id)
    items = _read_json(path, [])
    if not isinstance(items, list):
        items = []
    mid = int(message_id)
    if any(int(i.get("message_id", 0)) == mid for i in items):
        return
    ch = int(channel_id)
    ch_items = [i for i in items if i.get("channel_id") == ch]
    if len(ch_items) >= _DEFERRED_LIMIT_PER_CH:
        items = [i for i in items if i.get("channel_id") != ch] + ch_items[1:]
    items.append({
        "guild_id": int(guild_id) if guild_id else 0,
        "channel_id": ch,
        "message_id": mid,
        "author_id": int(author_id),
        "author_name": author_name,
        "text": (text or "")[:300],
        "ts": time.time(),
    })
    _atomic_write(path, items[-30:])


def deferred_for(persona_id: str) -> List[dict]:
    """Fresh deferred items for this persona (unexpired)."""
    now = time.time()
    return [
        it for it in _read_json(_deferred_path(persona_id), [])
        if now - it.get("ts", 0) < _DEFERRED_MAX_AGE_S
    ]


def remove_deferred(persona_id: str, message_id: int) -> None:
    path = _deferred_path(persona_id)
    items = _read_json(path, [])
    items = [i for i in items if int(i.get("message_id", 0)) != int(message_id)]
    _atomic_write(path, items)
