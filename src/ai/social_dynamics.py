"""Ambient social-dynamics policy.

This module turns raw channel/user signals into one calibrated probability for
whether a human-like persona would join an ambient message. It is deliberately
separate from the large client hot path so the social trade-offs are explicit
and unit-testable:

- busy multi-party rooms need less intervention
- direct questions/substance deserve more
- users who repeatedly engage get warmth; low-investment users get space
- channels that ignored recent engagement should not be peppered again
- a zero-willingness engine must mean near-silence, not a hidden probability
  floor
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AmbientSignals:
    text: str
    time_chance: float
    engagement_probability: float
    active_users: int
    user_engagement_level: str = "medium"
    consecutive_engagement_fails: int = 0
    recent_engagement: bool = False
    is_new_participant: bool = False
    is_returning_user: bool = False
    recently_joined: bool = False
    mentions_other_name: bool = False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def ambient_reply_chance(s: AmbientSignals) -> float:
    """Return calibrated probability [0.01, 0.82] for an ambient reply.

    The baseline intentionally starts low. Hooks (question/substance/being
    part of an active exchange) must earn the reply; this is what separates a
    socially aware participant from a reply bot.
    """
    # Time of day sets the person's general availability, but the willingness
    # engine is the dominant term. At zero willingness this remains selective
    # instead of inheriting a large time-of-day floor.
    chance = 0.035 + (s.time_chance * 0.42) + (s.engagement_probability * 0.42)

    text = (s.text or "").strip()
    if "?" in text:
        chance += 0.13
    if len(text) > 30:
        chance += 0.07
    elif len(text) < 12 and "?" not in text:
        chance -= 0.06

    # Room shape: a one-on-one-ish quiet channel welcomes participation; a
    # crowded fast conversation does not need a fourth voice on every line.
    if s.active_users <= 1:
        chance += 0.07
    elif s.active_users == 2:
        chance += 0.04
    elif s.active_users >= 5:
        chance -= 0.13
    elif s.active_users >= 3:
        chance -= 0.07

    # User warmth: people invest in people who invest back. Low/minimal
    # engagement does not mean hostility — it means shorter, rarer replies.
    level = (s.user_engagement_level or "medium").lower()
    if level == "high":
        chance += 0.06
    elif level == "low":
        chance -= 0.055
    elif level == "minimal":
        chance -= 0.095

    # Someone entering/returning to a quiet room is a strong social opening.
    if s.is_new_participant:
        chance += 0.20
        if s.is_returning_user:
            chance += 0.07
    if s.recently_joined:
        chance += 0.11

    # If the bot explicitly opened a topic, a following human message probably
    # answers it. This is the highest ambient hook.
    if s.recent_engagement:
        chance += 0.30

    # Failed engagement cycles are negative feedback. Learn from them: each
    # ignored opener lowers the urge to post into the same cold room.
    if s.consecutive_engagement_fails == 1:
        chance -= 0.09
    elif s.consecutive_engagement_fails >= 2:
        chance -= 0.18

    # A message plainly aimed at another named person should almost never be
    # joined ambiently; direct-address logic handles the rare exception before
    # this policy is consulted.
    if s.mentions_other_name:
        chance *= 0.18

    return clamp(chance, 0.01, 0.82)
