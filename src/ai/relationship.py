"""
Relationship progression — per-user social stats shared across personas.

Tracks how well the bot knows each user (message count, distinct active days,
facts learned, bot-suspicion strikes) and maps that onto a progression tier:

    stranger -> familiar -> acquaintance -> friend -> close_friend

The tier is injected into reply/voice prompts so the persona calibrates
familiarity — no teasing a stranger, no re-greeting a close friend.

Persistence goes through the shared server_state KV
(``mem.save_server_state(f"social:{user_id}", ...)``) which is deliberately
NON-namespaced: all three personas share one social graph, and it survives
redeploys (D1 when reachable, data/memory.json ``_server_state`` otherwise).

Writes are debounced — stats live in a module-level cache and only flush to
storage every _FLUSH_INTERVAL_S per user, so a chatty channel can't hammer D1.

Every public function is wrapped in try/except: this system must NEVER break
the reply path. Worst case, everyone reads as a stranger.
"""
import datetime
import time
from typing import Dict

from loguru import logger

from . import d1_memory as mem  # D1-backed, falls back to memory.json

# ── Tunables ──────────────────────────────────────────────────────────────────

_FLUSH_INTERVAL_S = 120      # min seconds between persisted writes per user
_DAYS_CAP = 60               # distinct-day list cap
_DECAY_AFTER_S = 21 * 86400  # >21 days silent -> drop one tier

_TIER_ORDER = ["stranger", "familiar", "acquaintance", "friend", "close_friend"]

# Behavioural cue per tier — injected into prompts so the model knows how
# familiar to act.
TIER_CUES = {
    "stranger": ("you basically just met — light curiosity, don't tease, "
                 "don't assume shared history, don't over-use their name"),
    "familiar": ("you've chatted a bit — light follow-ups fine, still "
                 "casual-acquaintance energy, not close friends"),
    "acquaintance": ("you know each other — reference things they've told "
                     "you (see memory), mild teasing ok"),
    "friend": ("you're friends — tease them, call back to things they've "
               "said before, be relaxed"),
    "close_friend": ("your people — fully relaxed, inside references, talk "
                     "like you've known them forever"),
}

# ≤8-word cues for the voice prompt (keeps the system prompt tight).
TIER_CUES_BRIEF = {
    "stranger": "just met — light curiosity, no teasing",
    "familiar": "chatted a bit — still casual",
    "acquaintance": "reference memory, mild teasing ok",
    "friend": "tease, call back, be relaxed",
    "close_friend": "fully relaxed, inside references",
}

# ── Stats cache ───────────────────────────────────────────────────────────────
# user_id -> stats dict; populated from storage on first touch, mutated in
# memory, flushed to server_state on the debounce schedule.
_stats: Dict[str, dict] = {}
_flush_at: Dict[str, float] = {}


def _key(user_id) -> str:
    return f"social:{user_id}"


def _blank() -> dict:
    return {"msg_count": 0, "first_seen": 0.0, "last_seen": 0.0,
            "days": [], "suspicion": 0}


def _stats_for(user_id) -> dict:
    """Return the cached stats dict for a user, merging in whatever is already
    persisted on first touch."""
    uid = str(user_id)
    cached = _stats.get(uid)
    if cached is not None:
        return cached
    merged = _blank()
    try:
        stored = mem.load_server_state(_key(uid))
    except Exception:
        stored = None
    if isinstance(stored, dict):
        for k in ("msg_count", "suspicion"):
            try:
                merged[k] = int(stored.get(k) or 0)
            except Exception:
                pass
        for k in ("first_seen", "last_seen"):
            try:
                merged[k] = float(stored.get(k) or 0.0)
            except Exception:
                pass
        days = stored.get("days")
        if isinstance(days, list):
            merged["days"] = [str(d) for d in days][-_DAYS_CAP:]
    _stats[uid] = merged
    return merged


def _flush(uid: str):
    """Persist the cached stats for a user. Best-effort — storage errors are
    swallowed so a dead D1 never breaks message handling."""
    try:
        st = _stats.get(uid)
        if st is None:
            return
        blob = dict(st)
        blob["days"] = list(st.get("days") or [])
        mem.save_server_state(_key(uid), blob)
        _flush_at[uid] = time.time()
    except Exception as e:
        logger.debug(f"relationship flush failed for {uid}: {e}")


# ── Recording ─────────────────────────────────────────────────────────────────

def record_message(user_id):
    """Count a message (or voice turn) from this user. Flushes to storage at
    most once per _FLUSH_INTERVAL_S per user."""
    try:
        uid = str(user_id)
        st = _stats_for(uid)
        now = time.time()
        st["msg_count"] = int(st.get("msg_count") or 0) + 1
        st["last_seen"] = now
        if not st.get("first_seen"):
            st["first_seen"] = now
        today = datetime.date.today().isoformat()
        days = st.setdefault("days", [])
        if today not in days:
            days.append(today)
            del days[:-_DAYS_CAP]
        if now - _flush_at.get(uid, 0) > _FLUSH_INTERVAL_S:
            _flush(uid)
    except Exception as e:
        logger.debug(f"relationship record_message failed: {e}")


def record_suspicion(user_id):
    """Bump the user's bot-suspicion counter (they accused us of being a bot).
    Always flushed — rare event, worth persisting immediately."""
    try:
        uid = str(user_id)
        st = _stats_for(uid)
        st["suspicion"] = int(st.get("suspicion") or 0) + 1
        _flush(uid)
    except Exception as e:
        logger.debug(f"relationship record_suspicion failed: {e}")


def get_stats(user_id) -> dict:
    """Return a copy of the user's merged stats dict."""
    try:
        st = _stats_for(user_id)
        out = dict(st)
        out["days"] = list(st.get("days") or [])
        return out
    except Exception:
        return _blank()


# ── Scoring / tiers ───────────────────────────────────────────────────────────

def _profile(user_id) -> dict:
    """Fetch the user's memory profile ({} on any failure)."""
    try:
        p = mem.get_user_profile(str(user_id))
        return p if isinstance(p, dict) else {}
    except Exception:
        return {}


def _facts_count(user_id) -> int:
    """Facts + hobbies known about a user — both memory paths return lists."""
    prof = _profile(user_id)
    try:
        return len(prof.get("facts") or []) + len(prof.get("hobbies") or [])
    except Exception:
        return 0


def compute_score(stats: dict, facts_count: int) -> float:
    """Relationship score from activity stats + known facts.

    msg_count contributes up to 3.0, distinct days up to 3.0, facts up to
    1.5; each suspicion strike costs 0.4.
    """
    stats = stats or {}
    try:
        msgs = min(int(stats.get("msg_count") or 0), 300) * 0.01
    except Exception:
        msgs = 0.0
    days = min(len(stats.get("days") or []), 20) * 0.15
    try:
        facts = min(int(facts_count or 0), 15) * 0.1
    except Exception:
        facts = 0.0
    try:
        susp = int(stats.get("suspicion") or 0) * 0.4
    except Exception:
        susp = 0.0
    return msgs + days + facts - susp


def _band_for(score: float) -> str:
    if score < 0.5:
        return "stranger"
    if score < 1.5:
        return "familiar"
    if score < 3.0:
        return "acquaintance"
    if score < 5.0:
        return "friend"
    return "close_friend"


def tier_for(user_id, owner: bool = False) -> str:
    """Resolve the user's relationship tier.

    - owner and users with a declared relationship get a floor (declared
      "best friend"/"close" floors at close_friend; owner/"friend"/"bestie"
      floor at friend)
    - close_friend additionally requires >=5 distinct active days
    - >21 days of silence decays the result one tier (never below stranger)
    """
    try:
        stats = get_stats(user_id)
        prof = _profile(user_id)
        try:
            facts = len(prof.get("facts") or []) + len(prof.get("hobbies") or [])
        except Exception:
            facts = 0
        tier = _band_for(compute_score(stats, facts))

        # close_friend needs real history, not just a busy afternoon
        if tier == "close_friend" and len(stats.get("days") or []) < 5:
            tier = "friend"

        # Floors — owner and declared relationships can't be strangers
        rel = str(prof.get("relationship") or "").strip().lower()
        floor = None
        if rel in ("best friend", "close"):
            floor = "close_friend"
        elif rel in ("friend", "bestie"):
            floor = "friend"
        if owner and floor is None:
            floor = "friend"
        if floor and _TIER_ORDER.index(tier) < _TIER_ORDER.index(floor):
            tier = floor

        # Decay: >3 weeks silent drops one tier, never below stranger
        try:
            last_seen = float(stats.get("last_seen") or 0.0)
        except Exception:
            last_seen = 0.0
        if last_seen and (time.time() - last_seen) > _DECAY_AFTER_S:
            tier = _TIER_ORDER[max(0, _TIER_ORDER.index(tier) - 1)]

        return tier
    except Exception:
        return "stranger"


# ── Prompt text ───────────────────────────────────────────────────────────────

def describe(user_id, owner: bool = False) -> str:
    """One-line history summary + behavioural cue for the text reply prompt."""
    try:
        stats = get_stats(user_id)
        tier = tier_for(user_id, owner=owner)
        cue = TIER_CUES.get(tier, TIER_CUES["stranger"])
        try:
            msg = int(stats.get("msg_count") or 0)
        except Exception:
            msg = 0
        if msg <= 0:
            if tier == "stranger":
                return ("stranger — you basically just met. CUE: light "
                        "curiosity, no teasing, don't assume shared history, "
                        "don't over-use their name.")
            return f"{tier} — {cue}"
        days = stats.get("days") or []
        try:
            first = float(stats.get("first_seen") or 0.0)
        except Exception:
            first = 0.0
        age_days = int((time.time() - first) / 86400) if first else 0
        return (f"{tier} — ~{msg} msgs across {len(days)} days, "
                f"first met {age_days} days ago. CUE: {cue}")
    except Exception:
        return ""


def describe_brief(user_id, owner: bool = False) -> str:
    """Compact 'tier — cue' line for the voice prompt."""
    try:
        tier = tier_for(user_id, owner=owner)
        return f"{tier} — {TIER_CUES_BRIEF.get(tier, TIER_CUES_BRIEF['stranger'])}"
    except Exception:
        return ""
