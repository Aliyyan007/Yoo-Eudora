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
    r"gemini|llama|mistral|qwen|copilot|llm\b|my\s+(?:system\s+)?prompt|"
    r"system\s+prompt|digitalocean|vercel|netlify|heroku|railway|gcp|azure|"
    r"aws\b|vps\b|droplet|ec2\b|kubernetes|docker|container|"
    r"my\s+deployment|my\s+hosting|hosting\s+platform)\b",
    re.IGNORECASE)
_API_ADMISSION = re.compile(
    r"\b(?:api\s*keys?|the\s+\w+\s+api|(?:use[sd]?|using)\s+an?\s+api|"
    r"on\s+render\b|deployed\s+on|hosted?\s+on\s+\w+)\b",
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
    if _API_ADMISSION.search(t) and not t.endswith("?"):
        return True
    return False
