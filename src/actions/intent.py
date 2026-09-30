"""Cheap action-intent prefilter for the side-channel worker.

The vendored router (`router.py`) already has a zero-token heuristic pass and
an LLM arbiter — but we don't want to burn an arbiter call on every directed
message. This regex is a loose first gate: if it doesn't fire, no action
check happens at all.
"""
import re

# Action verbs / targets — intentionally loose (false positives are fine,
# the classifier decides next). Kept separate from commands.py's COMMAND_PATTERNS
# which handles the regex-detectable built-ins (bump/vc/mention/memory).
_ACTION_HINT_RE = re.compile(
    r"\b("
    r"send|dm|message|msg|ping|mention|tag|tell|say|speak|"
    r"react|delete|remove|edit|change|update|rename|"
    r"schedule|remind|reminder|"
    r"gif|sticker|emoji|emote|"
    r"status|nickname|nick|profile|bio|avatar|"
    r"channel|members?|roles?|server|invite|"
    r"slash|bump|join|leave|move|mute|deafen|kick|ban|"
    r"clean|cleanup|purge|"
    r"search|find|look\s*up|list|show|get|fetch|read|check"
    r")\b",
    re.I,
)

# Phrases that look like actions but are almost always conversational —
# cheap rescue before burning a router call ("i'll send you the file",
# "delete this app", "check my horoscope")
_ACTION_SOFT_EXCLUDE_RE = re.compile(
    r"^\s*(i|im|i'm|we|they|he|she)\s+(will|'ll|am|'m|did|just|can|want)",
    re.I,
)


def looks_like_action(text: str) -> bool:
    """True when the text plausibly asks the bot to DO something.
    Loose gate — the router (heuristic → LLM arbiter) decides for real."""
    if not text or len(text) > 600:
        return False
    if not _ACTION_HINT_RE.search(text):
        return False
    # "i'll send it to you" — first-person intent, not a request to the bot
    if _ACTION_SOFT_EXCLUDE_RE.match(text) and "you" not in text.lower()[:14]:
        return False
    return True
