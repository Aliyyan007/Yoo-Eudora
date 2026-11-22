"""
Human timing engine — one place that models how a real person's Discord
actions land in time.

Every "bot tell" the audits found comes down to latency structure: a human
who reacts first had to notice the message, read it, decide, and move a
finger; a human who sends a GIF first had to open the picker, scroll and
pick. This module models those component latencies with the right
distributions (gaussian bodies, log-normal-ish tails, occasional long
"got distracted" outliers) and per-persona pacing, so no two actions land
on the same metronome.

Latency model notes (grounded in conversational-timing research):
- Turn-taking in human chat centres on ~200ms-2s for direct turn exchange,
  but *noticed* reactions in an async channel stretch much wider — a
  reaction is a low-effort ack that lands whenever the scroller reaches it.
- Typing: inter-keystroke flight times for casual mobile/desktop typing
  average 8-14 chars/sec burst pace, but real composition includes
  hesitation pauses (~15-25% of messages) where the writer stops to think,
  re-reads, or gets distracted. We fold pauses into total duration since
  Discord's typing indicator stays lit anyway.
- Long tail: ~5% of actions get a "distracted" delay (tabbed away, checked
  another channel) — the single most humanizing timing feature.
"""
import random
import time
from typing import Optional

# Per-persona pace multipliers — >1 means slower/lazier, <1 snappier.
# Deliberately subtle: the three share one server, so wildly different
# clocks would itself be a fingerprint.
_PACE = {
    "eudora": 1.0,
    "isla": 0.9,     # excitable — reaches for her phone fast
    "rowan": 1.25,   # half-watching a stream; everything's a beat late
}
_DEFAULT_PACE = 1.0

# Mood modifiers on action latency.
_MOOD_PACE = {
    "hyped": 0.8, "excited": 0.8, "giddy": 0.85, "playful": 0.9,
    "annoyed": 1.1, "angry": 0.9,   # angry types fast; annoyed stalls
    "sad": 1.3, "lonely": 1.2,
    "tired": 1.4,
}


def _pace(persona_id: Optional[str], mood: Optional[str]) -> float:
    p = _PACE.get(persona_id or "", _DEFAULT_PACE)
    m = _MOOD_PACE.get(mood or "", 1.0)
    return p * m


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _read_seconds(text_len: int, cps: float = 28.0) -> float:
    """Time to actually read a message — ~28 chars/sec skim pace with a
    floor, since even 'ok' takes a beat to register."""
    if text_len <= 0:
        return 0.0
    return _clamp(text_len / cps * random.uniform(0.8, 1.3), 0.15, 12.0)


def distraction_tail(p: float = 0.05) -> float:
    """Usually 0. Occasionally the human alt-tabbed, checked another app,
    answered the door — returns a big extra delay. The single most
    humanizing thing a bot's clock can do."""
    if random.random() >= p:
        return 0.0
    return random.uniform(15.0, 120.0)


# ── Reactions ────────────────────────────────────────────────────────────────

def reaction_delay(text_len: int = 0, directed: bool = False,
                   mood: str = "chill", persona_id: Optional[str] = None) -> float:
    """Seconds from message arrival to the emoji landing, when the reaction
    rides along with a reply (the person is already 'there' for this convo).

    Model: read time + decide/tap time. Reactions are low-effort — a human
    taps a reaction mid-read or right after, fast but never instant.
    """
    decide = _clamp(random.gauss(1.4, 0.6), 0.5, 4.0)
    # If they're about to type anyway the react often comes first — but a
    # funny message earns a fast react while a paragraph earns a slow one.
    total = (_read_seconds(text_len) * 0.6 + decide) * _pace(persona_id, mood)
    total += distraction_tail(0.04)
    return _clamp(total, 1.0, 30.0)


def standalone_react_delay(text_len: int = 0, mood: str = "chill",
                           persona_id: Optional[str] = None) -> float:
    """Seconds to react to a message the persona does NOT reply to — they're
    scrolling, see something funny, leave a 💀 and move on. Slower and
    wider than the reply-path reaction."""
    total = (_read_seconds(text_len) + random.uniform(3.0, 14.0)) \
        * _pace(persona_id, mood)
    total += distraction_tail(0.07)
    return _clamp(total, 2.0, 60.0)


def counter_react_delay(persona_id: Optional[str] = None) -> float:
    """Someone reacted to the persona's own message — the counter-react
    comes whenever they next glance at the channel."""
    total = random.uniform(4.0, 18.0) * _PACE.get(persona_id or "", _DEFAULT_PACE)
    total += distraction_tail(0.10)
    return _clamp(total, 3.0, 120.0)


def react_to_typing_gap(persona_id: Optional[str] = None,
                        mood: str = "chill") -> float:
    """Gap between the reaction landing and 'is typing…' appearing.

    A human who reacts AND replies usually does: react (done, message
    acknowledged) → compose thoughts → start typing. The gap is short but
    nonzero; occasionally they start typing right away.
    """
    if random.random() < 0.2:
        return random.uniform(0.3, 1.0)   # went straight to typing
    return _clamp(random.gauss(3.0, 1.2), 1.2, 8.0) * _pace(persona_id, mood)


# ── Media picking ────────────────────────────────────────────────────────────

def gif_browse_delay(persona_id: Optional[str] = None,
                     mood: str = "chill") -> float:
    """Time a human spends in the GIF picker: open, type a query, scroll,
    pick — and sometimes the first search was wrong and they re-search.
    This is THE tell for 'send me a gif': instant results read as a bot.
    """
    open_pick = random.uniform(0.8, 2.0)          # open picker + type query
    scroll = random.uniform(2.5, 7.0)              # scroll results
    total = open_pick + scroll
    if random.random() < 0.25:
        total += random.uniform(2.0, 6.0)          # re-search, better query
    total *= _pace(persona_id, mood)
    total += distraction_tail(0.06)
    return _clamp(total, 4.0, 45.0)


def sticker_pick_delay(persona_id: Optional[str] = None) -> float:
    """Sticker panel is faster than GIF search — still a real beat."""
    total = random.uniform(1.8, 5.5) * _PACE.get(persona_id or "", _DEFAULT_PACE)
    return _clamp(total, 1.2, 20.0)


# ── Typing ───────────────────────────────────────────────────────────────────

def typing_seconds(text: str, persona_id: Optional[str] = None,
                   mood: str = "chill") -> float:
    """How long 'is typing…' should show before a message of this length
    lands. Composition speed ~6-10 chars/sec with hesitation pauses folded
    in — faster than naive CPS for short texts (people burst-type), slower
    on long ones (they re-read what they wrote).
    """
    n = len(text or "")
    if n <= 0:
        return 0.0
    cps = random.uniform(6.0, 9.5)
    t = n / cps
    # Hesitation pause mid-compose — re-read, reword, check something.
    if n > 30 and random.random() < 0.22:
        t += random.uniform(1.0, 3.5)
    # Very short messages still need a human beat on the keys.
    t *= _pace(persona_id, mood)
    return _clamp(t, 1.2, 14.0)


def inter_burst_delay(persona_id: Optional[str] = None) -> float:
    """Gap between burst messages — hit send, think of the next line,
    start typing again."""
    return random.uniform(1.5, 4.5) * _PACE.get(persona_id or "", _DEFAULT_PACE)


# ── Helpers for call sites ───────────────────────────────────────────────────

def remaining(target_ts: float) -> float:
    """Seconds still owed until an absolute target time (clamped >= 0)."""
    return max(0.0, target_ts - time.time())


def target_from(message_ts: float, delay: float, cap: float = 30.0) -> float:
    """Absolute send/react time given when the triggering message arrived
    and how long a human would take — capped so a slow generation pipeline
    doesn't stretch the visible latency absurdly."""
    return message_ts + min(delay, cap)
