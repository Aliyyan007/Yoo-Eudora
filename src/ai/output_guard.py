"""
Output guard — the last checkpoint between generated text and the wire.

is_degenerate catches garbage LLM output (markdown theorem soup, ellipsis
storms, oversized rambles) that must never reach a channel — the caller
suppresses the send when it fires.
"""
import re

# Discord mention tokens render as real names/channels for the reader —
# to a text heuristic they're digit-noise that trips the letter-ratio
# check. Normalize each to a letter token before scoring.
_MENTION_TOKEN = re.compile(
    r"<@!?\d+>|<@&\d+>|<#\d+>|<a?:\w{2,}:\d+>|@everyone|@here")


def is_degenerate(text: str, max_len: int = 400) -> bool:
    """True for output that should never be sent — empty/oversized text,
    markdown-garbled model soup ("## Theorem (a). We're **…..**"), ellipsis
    storms, punctuation runs no human types, or mostly non-letter noise."""
    t0 = (text or "").strip()
    if len(_MENTION_TOKEN.findall(t0)) > 4:
        return True  # mention spam — no human drops 5+ links in one line
    t = _MENTION_TOKEN.sub(" mention ", t0)
    if not t:
        return True
    if len(t) > max_len:
        return True
    if "##" in t or "**" in t or "\n\n" in t or "theorem" in t.lower():
        return True
    if t.count("…") + t.count("...") >= 4:
        return True
    # Any run of >= 6 consecutive chars that are neither alnum nor whitespace
    run = 0
    for c in t:
        if not c.isalnum() and not c.isspace():
            run += 1
            if run >= 6:
                return True
        else:
            run = 0
    # Longer strings that are mostly symbols/digits, not letters
    if len(t) >= 20:
        non_space = [c for c in t if not c.isspace()]
        letters = sum(1 for c in non_space if c.isalpha())
        if letters < 0.4 * len(non_space):
            return True
    return False


# Naming an AI provider/model or discussing your own internals is the
# single most damning bot tell — "probably OpenAI GPT-4", "i use the groq
# api", "my system prompt". The prompt rule tells the model not to; this
# stops it when it does anyway. A provider name is a leak even inside a
# question ("probably GPT-4?" still outs you) — only bare "what api" /
# "which model" style confusion echoes pass.
_PROVIDER_LEAK = re.compile(
    r"\b(?:openai|gpt[-\s]?[0-9o]+|gpt\b|groq|deepseek|anthropic|claude|"
    r"gemini|mistral|qwen|copilot|llm\b|my\s+(?:system\s+)?prompt|"
    r"system\s+prompt|digitalocean|vercel|netlify|heroku|railway|"
    r"my\s+deployment|my\s+hosting|hosting\s+platform)\b",
    re.IGNORECASE)
# Everyday words that are ALSO infra terms — only a leak when they sit next
# to ops context ("the", "my", "a", "on") — 'i saw a llama' or 'the aws
# outage' as a news mention must not silence a reply.
_OPS_WORD = re.compile(
    r"\b(?:the|my|an?|your|this|that|on|in|its)\s+"
    r"(?:llama|qwen|aws|vps|droplet|ec2|kubernetes|docker|container)s?\b",
    re.IGNORECASE)
_API_ADMISSION = re.compile(
    r"\b(?:api\s*keys?|the\s+\w+\s+api|(?:use[sd]?|using)\s+an?\s+api|"
    r"on\s+render\b|deployed\s+on|hosted?\s+on\s+\w+)\b",
    re.IGNORECASE)
# Earnest humanity claims are the biggest bot tell of all — "nope, just a
# real human here 😊" outs the account harder than naming a provider.
# Real people mock accusations; they don't swear they're human.
_DENIAL_TELL = re.compile(
    r"\bi'?m\s+(?:just\s+)?a?\s*real\s+(?:human|person)\b|"
    r"\bjust\s+a\s+real\s+(?:human|person)\b",
    re.IGNORECASE)
_SELF_AI_TELL = re.compile(
    r"\bas\s+(?:an?\s+)?(?:ai|artificial\s+intelligence|language\s+model|chatbot)\b|"
    r"\bi(?:'m| am)\s+(?:just\s+)?(?:an?\s+)?(?:ai|artificial\s+intelligence|language\s+model|chatbot)\b",
    re.IGNORECASE)


def leaks_internals(text: str) -> bool:
    """True when a generated reply names an AI provider/model or admits to
    using an API. Confused echoes of a probe ('what api?', 'wdym') pass —
    the bare word in a question isn't an admission."""
    t = (text or "").strip()
    if not t:
        return False
    if _PROVIDER_LEAK.search(t):
        return True
    if _OPS_WORD.search(t):
        return True
    if _API_ADMISSION.search(t) and not t.endswith("?"):
        return True
    if _DENIAL_TELL.search(t):
        return True
    if _SELF_AI_TELL.search(t):
        return True
    return False


# ── Persona output policy ────────────────────────────────────────────────────
# Prompt instructions are probabilistic; these are deterministic boundaries for
# the highest-visibility identity contradictions. Keep them narrow: broad
# censorship makes output stilted, but a Rowan message full of hearts or Isla
# suddenly saying "bruv" is an immediate persona break.

_PERSONA_BANNED = {
    "eudora": re.compile(r"\b(?:aww|omg|yass|so\s+cute)\b", re.IGNORECASE),
    "isla": re.compile(r"\b(?:bruv|mate|roadman|meh)\b", re.IGNORECASE),
    "rowan": re.compile(r"\b(?:aww|omg|hehe|yass|bestie|so\s+cute)\b", re.IGNORECASE),
}

# Emoji in the broad Unicode pictograph blocks. Discord custom emoji are
# counted separately below.
_UNICODE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_CUSTOM_EMOJI = re.compile(r"<a?:\w{2,}:\d+>")
_HEART_EMOJI = re.compile("[\u2764\U0001F493-\U0001F49F\U0001F5A4\U0001F90D\U0001F90E\U0001FA75-\U0001FA77]")

_PERSONA_MAX_EMOJI = {
    "eudora": 1,
    "rowan": 1,
    # Isla is intentionally unrestricted beyond the global degenerate-output
    # checks — emoji abundance is part of her voice.
}
_ROWAN_ALLOWED_EMOJI = {"💀"}


def emoji_count(text: str) -> int:
    """Best-effort count of unicode/custom emoji in a message."""
    return len(_UNICODE_EMOJI.findall(text or "")) + len(_CUSTOM_EMOJI.findall(text or ""))


def persona_violations(text: str, persona_id: str = "") -> list:
    """Deterministic persona-boundary violations in generated output."""
    t = (text or "").strip()
    if not t or not persona_id:
        return []
    pid = persona_id.lower()
    violations = []

    banned = _PERSONA_BANNED.get(pid)
    if banned and banned.search(t):
        violations.append("persona-banned-vocabulary")

    count = emoji_count(t)
    max_emoji = _PERSONA_MAX_EMOJI.get(pid)
    if max_emoji is not None and count > max_emoji:
        violations.append("persona-emoji-overuse")

    if pid == "eudora" and _HEART_EMOJI.search(t):
        violations.append("eudora-heart-emoji")
    if pid == "isla" and t.lower() in {"k", "k.", "kk"}:
        violations.append("isla-dry-k")
    if pid == "rowan":
        for e in _UNICODE_EMOJI.findall(t):
            if e not in _ROWAN_ALLOWED_EMOJI:
                violations.append("rowan-disallowed-emoji")
                break

    return violations


def validate_outbound(text: str, persona_id: str = "", surface: str = "reply") -> tuple:
    """Unified final validation for every human-visible text surface.

    Returns (ok, reasons). `surface` tunes length limits while keeping the
    non-negotiable checks identical across replies, proactive messages,
    statuses, voice lines, and search follow-ups.
    """
    t = (text or "").strip()
    if not t:
        return False, ["empty"]
    max_len = {
        "status": 100,
        "voice": 500,
        "proactive": 240,
        "search": 400,
    }.get(surface, 400)
    if is_degenerate(t, max_len=max_len):
        return False, ["degenerate"]
    if leaks_internals(t):
        return False, ["internals-leak"]
    violations = persona_violations(t, persona_id)
    if violations:
        return False, violations
    return True, []
