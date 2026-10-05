"""Intent router — decides whether a request is conversation or action.

The user-facing win: casual chat never drags 56 tool schemas through the LLM
(~1500 wasted prompt tokens per call) and never hallucinates tool calls.
Action requests get a *filtered* tool subset so they resolve faster.

Two stages:
  1. Heuristic — regex cues map to tool categories instantly (0 tokens).
  2. Arbiter  — a ~15-token LLM call for genuinely ambiguous utterances.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.action_engine.config.settings import settings
from src.action_engine.core.groq_pool import get_pool
from src.action_engine.utils.logger import logger

ROUTE_CHAT = "chat"
ROUTE_ACTION = "action"

# keyword cue -> tool categories to include for the action
_CUES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"\b(bump|bumpall|bump all|disboard|d bump)\b", re.I), {"bump"}),
    (re.compile(r"\bslash\b|/\w+\s+command", re.I), {"slash"}),
    (re.compile(r"\b(join|come (to|into)|hop in|vc|voice|stage|leave|mute|deafen|unmute|move to)\b", re.I), {"voice"}),
    (re.compile(r"\b(send|say|tell|message|dm|dm him|dm her|whisper|write|post|reply|type|text|greet|welcome|announce|spam|spamming)\b", re.I), {"messaging"}),
    (re.compile(r"\b(delete|edit|remove|undo|take back|edit that)\b", re.I), {"messaging"}),
    (re.compile(r"\b(react|reaction|emoji)\b", re.I), {"reactions"}),
    (re.compile(r"\b(gif|gifs|meme|tenor|klipy)\b", re.I), {"media"}),
    (re.compile(r"\b(sticker|stickers)\b", re.I), {"media"}),
    (re.compile(r"\b(ping|mention|tag|summon|call)\b", re.I), {"messaging", "members", "roles"}),
    (re.compile(r"\b(who|member|members|user|users|online|joined|left|profile|bio|avatar|count)\b", re.I), {"members"}),
    (re.compile(r"\b(channel|channels|category|topic|nsfw)\b", re.I), {"channels"}),
    (re.compile(r"\b(role|roles)\b", re.I), {"roles"}),
    (re.compile(r"\b(nickname|status|custom status|bio|about me|presence)\b", re.I), {"profile"}),
    (re.compile(r"\b(read|what did|last message|recent messages|check the|look at|fetch)\b", re.I), {"messages"}),
    # timed actions: "after 5 sec", "every X seconds", "keep sending",
    # "until i say stop", "stop it", "schedule a reminder"
    (re.compile(r"\b(every|after|in \d|seconds?|minutes?|hours?|interval|repeat|keep (sending|posting|doing)|until i|schedule|scheduled|remind|timer|stop (it|that|sending|the)|cancel)\b", re.I), {"scheduling", "messaging"}),
    (re.compile(r"\b(prefer|preference|remember that|from now on|always greet|always welcome|default channel)\b", re.I), {"prefs"}),
]

# words that almost always mean the person wants the bot to DO something
_ACTION_VERBS = re.compile(
    r"\b(send|say|tell|ping|mention|tag|dm|delete|edit|react|join|leave|"
    r"move|mute|deafen|bump|post|write|read|check|search|find|list|show|"
    r"fetch|get|count|use|invoke|play|sticker|gif|change|set|update|"
    r"make|create|spawn|click|press|spam|announce|welcome|greet|give|"
    r"assign|schedule|remind|repeat|stop|cancel)\b",
    re.I,
)

# Fuzzy verb set — catches typos ("jioin", "delte", "pingg") that exact
# regex misses. Only for words >=4 chars so "hi"/"go" can't false-match.
_VERB_WORDS = (
    "send", "tell", "ping", "mention", "tag", "delete", "edit", "react",
    "join", "leave", "move", "mute", "deafen", "bump", "post", "write",
    "read", "check", "search", "find", "list", "show", "fetch", "count",
    "invoke", "play", "sticker", "gif", "change", "update", "make",
    "spam", "announce", "welcome", "greet", "give", "assign",
)
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")

# single-word -> categories, for fuzzy matching of typo'd cues
_FUZZY_CUES: dict[str, set] = {
    "ping": {"messaging", "members", "roles"},
    "mention": {"messaging", "members", "roles"},
    "message": {"messaging"}, "send": {"messaging"},
    "join": {"voice"}, "voice": {"voice"}, "leave": {"voice"},
    "mute": {"voice"}, "deafen": {"voice"},
    "channel": {"channels"}, "channels": {"channels"},
    "member": {"members"}, "members": {"members"},
    "delete": {"messaging"}, "react": {"reactions"},
    "gif": {"media"}, "gifs": {"media"}, "sticker": {"media"},
    "bump": {"bump"}, "profile": {"members"}, "avatar": {"members"},
    "welcome": {"messaging"}, "greet": {"messaging"},
    "dm": {"messaging"}, "role": {"roles"},
    "schedule": {"scheduling"}, "remind": {"scheduling"},
    "repeat": {"scheduling"}, "stop": {"scheduling"},
    "cancel": {"scheduling"}, "prefer": {"prefs"},
}


def _fuzzy(t: str) -> tuple[bool, set]:
    """Returns (has_action_verb, categories) from near-miss words only —
    exact matches are already handled by the regexes above."""
    from rapidfuzz import fuzz
    verb_hit, cats = False, set()
    for w in _WORD_RE.findall(t):
        wl = w.lower()
        for v in _VERB_WORDS:
            if wl != v and fuzz.ratio(wl, v) >= 87:
                verb_hit = True
                break
        for kw, kcats in _FUZZY_CUES.items():
            if wl != kw and len(kw) >= 3 and fuzz.ratio(wl, kw) >= 87:
                cats |= kcats
    return verb_hit, cats


@dataclass
class Route:
    kind: str = ROUTE_CHAT
    categories: set = field(default_factory=set)
    via: str = "heuristic"


def classify(text: str) -> Route | None:
    """Fast deterministic classification. Returns None when ambiguous
    (caller may run the LLM arbiter or default to chat)."""
    t = (text or "").strip()
    if not t:
        return Route(ROUTE_CHAT, via="empty")

    cats: set = set()
    for pat, c in _CUES:
        if pat.search(t):
            cats |= c

    verb_exact = bool(_ACTION_VERBS.search(t))
    verb_fuzzy, fuzzy_cats = _fuzzy(t)
    cats |= fuzzy_cats

    # "join(ed) the server" is a member-lookup context, not a voice action —
    # prune the voice category that the bare "join" cue pulled in.
    if re.search(r"\bjoin(ing|ed)?\b.*\b(server|guild|discord)\b|\b(joined|newest|latest|recent)\b.*\b(join|member|joiner)\b", t, re.I):
        cats.discard("voice")

    # action verb present (exact or near-typo) AND we mapped at least one
    # category -> action
    if cats and (verb_exact or verb_fuzzy):
        return Route(ROUTE_ACTION, cats, via="heuristic")

    # question/chat forms — no action intent
    if not cats:
        return Route(ROUTE_CHAT, via="heuristic")

    return None  # keywords matched but no action verb — ambiguous


_ARB_PROMPT = (
    "You route short Discord utterances to a system. Reply with exactly one "
    "word: ACTION if the speaker wants the agent to perform a Discord task "
    "(send a message, ping someone, join/leave voice, bump, send gif/sticker, "
    "delete/edit, react, read messages, change profile, look up info), or "
    "CHAT if it's conversation, a question, or talk directed at anyone. "
    "Reply with only ACTION or CHAT."
)


async def classify_llm(text: str) -> Route:
    """LLM arbiter for ambiguous utterances (~30 tokens in, ~3 out)."""
    heuristic = classify(text)
    if heuristic is not None:
        return heuristic
    pool = get_pool()
    try:
        resp = await pool.chat(
            # Router arbiter — small/cheap model, configurable via
            # ROUTER_MODEL (defaults to 3.3-70b; scout isn't on all tiers).
            model=settings.router_model,
            messages=[
                {"role": "system", "content": _ARB_PROMPT},
                {"role": "user", "content": text[:400]},
            ],
            tools=None, tool_choice="none",
            temperature=0.0, max_tokens=8,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        if "ACTION" in verdict:
            # collect categories from cues anyway for the filtered set
            cats: set = set()
            for pat, c in _CUES:
                if pat.search(text):
                    cats |= c
            return Route(ROUTE_ACTION, cats, via="arbiter")
        return Route(ROUTE_CHAT, via="arbiter")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"router arbiter failed, defaulting to chat: {e}")
        return Route(ROUTE_CHAT, via="fallback")
