"""
Output guard — the last checkpoint between generated text and the wire.

is_degenerate catches garbage LLM output (markdown theorem soup, ellipsis
storms, oversized rambles) that must never reach a channel — the caller
suppresses the send when it fires.
"""
import re


def is_degenerate(text: str, max_len: int = 400) -> bool:
    """True for output that should never be sent — empty/oversized text,
    markdown-garbled model soup ("## Theorem (a). We're **…..**"), ellipsis
    storms, punctuation runs no human types, or mostly non-letter noise."""
    t = (text or "").strip()
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
