"""Persistent persona life continuity.

A convincing person does not reinvent what they are doing every time someone
asks "wyd". This module gives each persona a stable, persona-appropriate
activity for the current date/time slot, plus a light weekly obligation. The
selected anchor is persisted in shared server state so restarts/redeploys do
not reset the life narrative.

The module is deliberately deterministic: the date+slot+persona seed chooses
the activity, while persistence records what was exposed and prevents rapid
back-to-back repeats when the deterministic pick collides with yesterday's.
"""
import datetime as _dt
import hashlib
import threading
import time
from typing import Dict, List, Tuple

from loguru import logger

from . import d1_memory as mem

_STATE_KEY = "persona_life_state"
_CACHE = None
_CACHE_AT = 0.0
_CACHE_TTL_S = 30.0
_LOCK = threading.RLock()

# activity strings are intentionally small/concrete. They double as custom
# status material, so they should be noun/gerund phrases rather than stories.
_ROUTINES: Dict[str, Dict[str, List[str]]] = {
    "eudora": {
        "morning": [
            "nursing coffee and avoiding emails",
            "sketching thumbnails before class",
            "digging through depop listings",
            "walking to campus half asleep",
            "sorting scans from her sketchbook",
        ],
        "afternoon": [
            "pretending to work on a crit piece",
            "in the studio making a mess",
            "editing a zine layout",
            "wandering camden with her camera",
            "collecting references for a print project",
        ],
        "evening": [
            "half-watching a film and sketching",
            "organising her depop parcels",
            "touching up a drawing",
            "making a late coffee and procrastinating",
            "sorting a playlist for the studio",
        ],
        "night": [
            "scrolling pinterest instead of sleeping",
            "redrawing the same panel again",
            "watching a film she's already seen",
            "listening to lo-fi with the lights off",
            "staring at a blank page like it owes her money",
        ],
        "late_night": [
            "half asleep scrolling pinterest",
            "listening to one song on repeat",
            "ignoring tomorrow's alarm",
            "sketching badly but refusing to stop",
        ],
    },
    "isla": {
        "morning": [
            "rating a flat white in her notes app",
            "backing up photos from yesterday",
            "walking through the northern quarter",
            "making a playlist for the day",
            "scrolling cheap flights she can't buy",
        ],
        "afternoon": [
            "editing portraits in lightroom",
            "out shooting street photos",
            "planning a café crawl",
            "sorting gig photos",
            "in the library pretending to edit",
        ],
        "evening": [
            "editing pics with a film on",
            "making a playlist for a mate",
            "at a tiny gig",
            "golden hour hunting with her camera",
            "backing up memory cards",
        ],
        "night": [
            "scrolling pinterest for trip ideas",
            "rewatching a comfort film",
            "editing one photo for way too long",
            "sending everyone songs they didn't ask for",
        ],
        "late_night": [
            "half asleep with headphones on",
            "looking at flights instead of sleeping",
            "editing pics past midnight",
            "watching travel vlogs in bed",
        ],
    },
    "rowan": {
        "morning": [
            "opening the repair shop",
            "diagnosing a dead pc",
            "drinking tea and ignoring windows updates",
            "sorting parts at work",
        ],
        "afternoon": [
            "rebuilding someone's overheating pc",
            "testing a secondhand gpu",
            "cable managing a customer build",
            "browsing parts he can't justify buying",
            "half-watching a stream at work",
        ],
        "evening": [
            "in a ranked queue regretting it",
            "playing guitar badly in his room",
            "browsing steam without buying anything",
            "watching a film with byte on the keyboard",
            "tweaking fan curves for no reason",
        ],
        "night": [
            "in ranked lobbies he should have left hours ago",
            "rehearsing the same riff",
            "watching hardware reviews",
            "playing an indie game instead of sleeping",
        ],
        "late_night": [
            "queueing one last game for the fifth time",
            "watching retro game videos",
            "tinkering with his rig at stupid o'clock",
            "half asleep in discord",
        ],
    },
}

_WEEKLY = {
    "eudora": "UAL crit later this week; sunday call with her mum",
    "isla": "a photo editing deadline and a mate's gig this week",
    "rowan": "repair-shop shifts on weekdays and band practice if everyone turns up",
}

_DEFAULT = {
    "morning": ["having a slow morning", "making coffee"],
    "afternoon": ["getting on with the day", "half busy rn"],
    "evening": ["winding down", "watching something"],
    "night": ["awake for no good reason", "scrolling in bed"],
    "late_night": ["should be asleep", "half asleep"],
}


def _slot(now=None) -> str:
    now = now or _dt.datetime.utcnow()
    h = now.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    if 22 <= h < 24 or 0 <= h < 2:
        return "night"
    return "late_night"


def _date_key(now=None) -> str:
    return (now or _dt.datetime.utcnow()).strftime("%Y-%m-%d")


def _stable_index(seed: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def _load_state() -> dict:
    global _CACHE, _CACHE_AT
    now = time.time()
    with _LOCK:
        if _CACHE is not None and now - _CACHE_AT < _CACHE_TTL_S:
            return _CACHE
        try:
            state = mem.load_server_state(_STATE_KEY) or {}
        except Exception as e:
            logger.debug(f"life-state load failed: {e}")
            state = {}
        if not isinstance(state, dict):
            state = {}
        _CACHE = state
        _CACHE_AT = now
        return _CACHE


def _save_state(state: dict) -> None:
    global _CACHE, _CACHE_AT
    with _LOCK:
        _CACHE = state
        _CACHE_AT = time.time()
    try:
        mem.save_server_state(_STATE_KEY, state)
    except Exception as e:
        logger.debug(f"life-state save failed: {e}")


def current_activity(persona_id: str, *, now=None) -> str:
    """Return the persona's stable activity for the current date/time slot."""
    pid = persona_id or "eudora"
    slot = _slot(now)
    day = _date_key(now)
    choices = (_ROUTINES.get(pid) or _DEFAULT).get(slot) or _DEFAULT["evening"]
    idx = _stable_index(f"{pid}|{day}|{slot}", len(choices))

    state = _load_state()
    pstate = state.setdefault(pid, {})
    key = f"{day}|{slot}"
    if pstate.get("slot_key") == key and pstate.get("activity"):
        return pstate["activity"]
    recent = pstate.get("recent_activities") or []

    # If today's deterministic pick is identical to the previous slot's
    # activity, rotate one step. The tiny variation matters: repeating the
    # same activity across afternoon/evening makes the persona feel static.
    if choices[idx] in recent[-2:] and len(choices) > 1:
        idx = (idx + 1) % len(choices)
    activity = choices[idx]

    if pstate.get("slot_key") != key or pstate.get("activity") != activity:
        pstate.update({
            "slot_key": key,
            "date": day,
            "slot": slot,
            "activity": activity,
            "updated_at": time.time(),
        })
        pstate["recent_activities"] = (recent + [activity])[-8:]
        _save_state(state)
    return activity


def weekly_anchor(persona_id: str) -> str:
    return _WEEKLY.get(persona_id or "", "")


def continuity_context(persona_id: str, *, now=None) -> str:
    """Compact prompt block pinning the persona's current life state."""
    pid = persona_id or "eudora"
    activity = current_activity(pid, now=now)
    week = weekly_anchor(pid)
    week_txt = f" this week: {week}." if week else ""
    return (
        f"[life continuity: right now you are {activity}.{week_txt} "
        "if asked wyd/what you're doing/how your day is, use THIS anchor — "
        "phrase it naturally but do not invent a conflicting job, class, trip, "
        "game, or activity. small details may evolve, the core activity must stay consistent.]\n"
    )


def status_activity(persona_id: str, *, now=None) -> str:
    """Short activity text suitable for Discord custom status."""
    activity = current_activity(persona_id, now=now)
    # Strip leading filler so statuses don't become sentences.
    for prefix in ("pretending to ", "half-watching ", "half asleep "):
        if activity.startswith(prefix):
            return activity[len(prefix):][:48]
    return activity[:48]


def snapshot() -> dict:
    """Debug/metrics snapshot of the persisted life state."""
    return dict(_load_state() or {})
