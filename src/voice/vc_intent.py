"""Voice intent scorers + irritation tracker — the algorithmic "social brain"
of the voice pipeline.

All detectors are pure regex/heuristic scorers — zero LLM calls, zero
latency, and every threshold is fuzzy/probabilistic rather than a hard rule.

Used by:
- pipeline._build_turn_hints  → per-turn prompt directive + TTS prosody boost
- manager._check_current_vc   → stay-duration / dry-VC leave scoring
- discord_client              → VC-transfer request flow
"""
from __future__ import annotations

import re
import time
from typing import Optional


# ── Farewell detection ──────────────────────────────────────────────────────
# Spoken farewells people actually say — matched loosely, scored not gated.
_FAREWELL_PHRASES = [
    (r"\bi(?:'m| am)?\s*(?:gonna|gotta|going to|about to)\s*(?:go|head out|dip|leave|sleep)\b", 0.9),
    (r"\bi'?m\s+feeling\s+sleepy\b", 0.85),
    (r"\bgoing to (?:bed|sleep)\b", 0.85),
    (r"\bheading (?:out|off|to bed)\b", 0.85),
    (r"\bi'?m\s+(?:out|off)\b", 0.8),
    (r"\bgotta run\b", 0.85),
    (r"\bgtg\b", 0.9),
    (r"\bgood ?night\b", 0.9),
    (r"\bnight night\b", 0.9),
    (r"\bsee (?:you|ya) (?:later|around|soon)\b", 0.9),
    (r"\bsee (?:you|ya)\b", 0.7),
    (r"\btalk (?:to )?you later\b", 0.85),
    (r"\bcatch (?:you|ya) later\b", 0.85),
    (r"\btake care\b", 0.75),
    (r"\bpeace out\b", 0.85),
    (r"\bbye ?bye\b", 0.9),
    (r"\bbye\b", 0.7),
    (r"\blater\b", 0.35),   # weak alone — 'see you later'/'catch you later' carry the phrase
    (r"\bi'?m\s+leaving\b", 0.8),
    (r"\bi'?ll?\s+(?:be|see you)\s+(?:back|later)\b", 0.6),
]


def farewell_score(text: str) -> float:
    """0-1 confidence that this utterance is a goodbye. Short, clean
    farewells score highest; longer sentences mentioning 'go' score less."""
    t = text.lower().strip()
    if len(t) < 2:
        return 0.0
    score = 0.0
    for pat, w in _FAREWELL_PHRASES:
        if re.search(pat, t):
            score = max(score, w)
    if score == 0.0:
        return 0.0
    # Brevity boost — "i'm gonna go" alone is a real goodbye; buried inside a
    # long sentence it's usually not the point. A 1-3 word utterance that is
    # essentially just the farewell ("later", "bye", "gtg") still counts.
    words = len(t.split())
    if words <= 3 and score >= 0.35:
        score = max(score, 0.85)
    elif words <= 8:
        score += 0.15
    elif words <= 15:
        score += 0.05
    else:
        score -= 0.15
    # Questions aren't farewells ("should i go?")
    if t.endswith("?"):
        score -= 0.2
    return max(0.0, min(1.0, score))


# ── Abuse / insult severity ─────────────────────────────────────────────────
# Reuses the text-path detector (same vocabulary) but keeps a running
# per-user intensity score — that's what makes escalation algorithmic.
def abuse_severity(text: str) -> float:
    """0-10 severity of insult/abuse in one utterance. 0 = clean."""
    try:
        from ..ai.abuse_handler import detect_abuse
        level, severity = detect_abuse(text)
        if level == "none":
            return 0.0
        return float(severity)
    except Exception:
        return 0.0


# ── "Join my/their VC" detection ────────────────────────────────────────────
_VC_INVITE_PATTERNS = [
    r"\bjoin\s+(?:my|our|the other|their)\s+(?:vc|voice|call|channel)\b",
    r"\bcome\s+(?:to|in|join)\s+(?:my|our|the)\s+(?:vc|voice|call|channel)\b",
    r"\bhop\s+(?:in|into|over to)\s+(?:my|our|the)\s+(?:vc|voice|call)\b",
    r"\b(?:join|come to)\s+(?:us|the other)\s+(?:vc|channel|call)\b",
    r"\bmove\s+to\s+(?:my|our|their|the)\s+(?:vc|voice|channel)\b",
    r"\bswitch\s+(?:to|over)\s+(?:my|our|the other)\s+(?:vc|channel)\b",
]


def vc_invite_score(text: str) -> float:
    """0-1 confidence the speaker is asking the bot to move/join a VC."""
    t = text.lower().strip()
    for pat in _VC_INVITE_PATTERNS:
        if re.search(pat, t):
            return 0.8
    return 0.0


# ── "Leave the VC" spoken command ───────────────────────────────────────────
# Imperative forms aimed AT the bot — distinct from farewells ("i'm leaving"
# = goodbye; "leave the vc" = a command the bot must obey).
_VC_LEAVE_CMD_PATTERNS = [
    r"\bleave\s+(?:the|this|our|my|their)?\s*(?:vc|voice|call|channel)\b",
    r"\bget\s+out\s+of\s+(?:the|this|our|my|their)?\s*(?:vc|voice|call|channel)\b",
    r"\b(?:eudora|bot)\s+leave\b",
    r"\bleave\s+(?:this|the)\s+(?:channel|call)\b",
    r"\bdisconnect\b",
    r"\bhop\s+off\b",
    r"\bgo\s+join\s+(?:their|their\s+other|another|the other)\s+(?:vc|channel)\b",
    r"\bfuck\s+off\s+(?:the\s+)?vc\b",
    r"\bbuzz\s+off\b",
]


def leave_vc_score(text: str) -> float:
    """0-1 confidence the speaker is TELLING the bot to leave the VC.
    'i'm leaving' is a farewell, not a command — this only catches imperatives."""
    t = text.lower().strip()
    # "i'm leaving" / "i gotta go" = farewell, handled elsewhere.
    # "let's leave" is a suggestion, "did/should you leave" is a question —
    # neither is a command.
    if re.search(r"\bi(?:'m| am)?\s+(?:leaving|gonna go|gotta go|heading)", t):
        return 0.0
    if re.search(r"\b(?:let'?s|did|should|can|could|will|would|do|are|wanna)\s*(?:you\s+)?leave\b", t):
        return 0.0
    for pat in _VC_LEAVE_CMD_PATTERNS:
        if re.search(pat, t):
            return 0.85
    # "leave!" / "you leave" alone — very short imperatives only
    if re.search(r"\b(?:you\s+)?leave\b", t) and len(t.split()) <= 3:
        return 0.6
    return 0.0


# ── Song request detection ──────────────────────────────────────────────────
_SONG_PATTERNS = [
    r"\bsing\s+(?:me\s+)?(?:a\s+)?song\b",
    r"\bsing\s+(?:something|for\s+(?:me|us)|about)\b",
    r"\bcan\s+you\s+sing\b",
    r"\bdo\s+you\s+sing\b",
    r"\b(?:a\s+)?song\s+(?:about|for|from)\b",
    r"\bmake\s+(?:up\s+)?a\s+song\b",
    r"\bwrite\s+(?:me\s+)?a\s+song\b",
    r"\brap\s+(?:for\s+(?:me|us)|something|about)\b",
    r"\bserenade\b",
    r"\bplay\s+(?:me\s+)?(?:a\s+)?song\b",
    r"\bmusic\b.*\bsing\b|\bsing\b.*\bmusic\b",
]


def song_request_score(text: str) -> float:
    """0-1 confidence the speaker wants the bot to sing."""
    t = text.lower().strip()
    for pat in _SONG_PATTERNS:
        if re.search(pat, t):
            return 0.85
    # bare "sing" as an imperative to the bot
    if re.search(r"\bsing\b", t) and len(t.split()) <= 5:
        return 0.7
    return 0.0


# ── Yes / no vote detection (for "can I go?" permission asks) ───────────────
_VOTE_YES = [
    r"\byeah\b", r"\byes\b", r"\byes+\b", r"\byup\b", r"\byeh\b", r"\bye\b",
    r"\bsure\b", r"\bof course\b", r"\bgo ahead\b", r"\bgo for it\b",
    r"\ballow it\b", r"\bfine\b", r"\bok(?:ay)?\b", r"\bdo it\b",
    r"\bi don'?t mind\b", r"\bgo on\b", r"\boff you go\b", r"\bsafe\b",
]
_VOTE_NO = [
    r"\bno\b", r"\bnah\b", r"\bnope\b", r"\bdon'?t\b", r"\bstay\b",
    r"\bplease stay\b", r"\bdon'?t go\b", r"\bdont go\b", r"\bwait\b",
    r"\bno way\b", r"\bhell no\b", r"\bnaw\b",
]


def vote(text: str) -> Optional[str]:
    """'yes' / 'no' / None — decisive short answers only."""
    t = text.lower().strip()
    words = len(t.split())
    if words > 12:
        return None  # long speech isn't a vote
    # 'no' wins on conflict — dissent beats assent for permission asks
    for pat in _VOTE_NO:
        if re.search(pat, t):
            return "no"
    for pat in _VOTE_YES:
        if re.search(pat, t):
            return "yes"
    return None


# ── Irritation tracker — per-user, decaying, tiered ─────────────────────────
_IRR_DECAY_PER_MIN = 4.0       # score points cooled per minute of peace
_IRR_APOLOGY_DROP = 15.0       # apology/calm words knock this off instantly
_IRR_MAX = 100.0

_APOLOGY_PAT = re.compile(
    r"\b(?:sorry|my bad|didn'?t mean|no offence|jk|joking|just kidding|"
    r"chill|calm down|apologis|apologiz|love you|all good)\b"
)

# Tier boundaries → (name, directive for the LLM prompt, vol_db, speed_mult)
_IRR_TIERS = [
    (70, "furious",
     "They keep insulting you nonstop — you're FURIOUS now. Swear back, tell them off "
     "(\"fuck off\", \"shut the fuck up\", \"piss off\"), keep it 1-2 sentences, real "
     "angry energy. You might threaten to leave.",
     2.5, 1.15),
    (45, "heated",
     "They've been insulting you — you're properly angry. Push back, be sharp, some "
     "profanity is fine (\"mate, fuck off\", \"what's your problem\"). Still short.",
     1.5, 1.08),
    (20, "irritated",
     "That was rude — you're a bit annoyed. Be shorter and colder with them, maybe "
     "call it out (\"okay what's your problem\", \"allow it\").",
     0.5, 1.03),
    (0, "calm", "", 0.0, 1.0),
]


class IrritationTracker:
    """Per-user irritation score (0-100) with time decay + apology credit.

    This is what makes anger algorithmic: a single 'shut up' bumps the score
    a little (bot stays cool, asks why), sustained abuse escalates through
    the tiers, and it decays back down when things calm down.
    """

    def __init__(self):
        self._scores: dict[int, float] = {}       # user_id -> score
        self._last_update: dict[int, float] = {}  # user_id -> ts

    def _decayed(self, user_id: int) -> float:
        now = time.time()
        score = self._scores.get(user_id, 0.0)
        last = self._last_update.get(user_id, now)
        if last:
            score -= (now - last) / 60.0 * _IRR_DECAY_PER_MIN
        self._last_update[user_id] = now
        return max(0.0, score)

    def feed(self, user_id: int, text: str) -> float:
        """Feed a transcript — returns the (updated) irritation score."""
        score = self._decayed(user_id)
        sev = abuse_severity(text)
        if sev > 0:
            # severity 0-10 → +6 to +30 points; repeated abuse compounds
            score = min(_IRR_MAX, score + sev * 3.0)
        elif _APOLOGY_PAT.search(text.lower()):
            score = max(0.0, score - _IRR_APOLOGY_DROP)
        self._scores[user_id] = score
        self._last_update[user_id] = time.time()
        return score

    def tier(self, user_id: int) -> tuple:
        """(name, directive, vol_db, speed_mult) for the current score."""
        score = self._decayed(user_id)
        self._scores[user_id] = score
        for threshold, name, directive, vol, spd in _IRR_TIERS:
            if score >= threshold:
                return name, directive, vol, spd, score
        return _IRR_TIERS[-1][0], "", 0.0, 1.0, score

    def leave_urge(self, user_id: int) -> float:
        """0-1 extra urge to leave the VC when heated — feeds the
        stay-duration scoring in the manager."""
        score = self._decayed(user_id)
        if score >= 70:
            return 0.8
        if score >= 45:
            return 0.35
        return 0.0

    def clear(self, user_id: int) -> None:
        self._scores.pop(user_id, None)
        self._last_update.pop(user_id, None)

    def clear_all(self) -> None:
        self._scores.clear()
        self._last_update.clear()
