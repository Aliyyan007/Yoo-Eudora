"""Call-name resolution — what a real friend would call someone.

Discord display names are decorations ("Mr. Alien", "👑 Sarah ✨",
"[GF] Tom | he/him"). A bot that reads them aloud verbatim sounds like a
receptionist, not a person. This module resolves the name the persona
should actually SAY:

1. The learned real name (from memory — "my name is Aliyyan", "call me
   Jeff") always wins.
2. Otherwise the display name, cleaned: titles stripped, emoji/clan tags
   removed, first name-ish word picked.
"""
import re
from typing import Optional

# Honorifics people front-load into display names — never spoken.
_NAME_TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "lord", "mx", "prof",
    "capt", "captain", "king", "queen", "its", "im", "iam", "the",
}

# Filler words in decorated handles — skipped when picking the name word.
# "its ya boy Jay" → "Jay", "Lil Peep" → "Peep", "xX_Alien_Xx" → "Alien"
# (the xX fragments are too short to reach this list).
_NAME_FILLERS = {
    "ya", "yah", "boi", "boy", "girl", "lad", "lil", "big", "da", "real",
    "official", "just", "not", "ur", "only", "aka", "aka",
}


def clean_display_name(name: str) -> str:
    """'Mr. Alien' → 'Alien', '👑Sarah✨' → 'Sarah', '[GF] Tom' → 'Tom'.

    Conservative: if nothing name-like survives, return the original so the
    caller still has *something* to say.
    """
    if not name:
        return name
    n = name.strip()
    # [CLAN] / (tag) / {x} prefixes
    n = re.sub(r"^[\[\(\{][^\]\)\}]{1,12}[\]\)\}]\s*", "", n)
    # "| he/him" / "｜" suffixes
    n = n.split("|")[0].split("｜")[0]
    # Emoji/symbols → spaces; underscores split too (xX_Alien_Xx → xX Alien Xx)
    n = re.sub(r"[^\w\s'\-]|_", " ", n)
    words = [w for w in n.split() if w]
    # Strip leading honorifics
    while words and words[0].lower().rstrip(".") in _NAME_TITLES:
        words.pop(0)
    # Keep words that contain at least one letter, drop filler decorations
    words = [w for w in words if any(ch.isalpha() for ch in w)
             and w.lower().rstrip(".") not in _NAME_FILLERS]
    if not words:
        return name.strip()
    # First name-ish word (≥3 chars, not a repeated decoration like 'xxx')
    for w in words:
        w2 = w.strip("-'\"")
        if len(w2) >= 3 and len(set(w2.lower())) > 1:
            return w2
    # Fallback: first word that's at least 2 chars, else the raw name
    for w in words:
        w2 = w.strip("_-'\"")
        if len(w2) >= 2:
            return w2
    return name.strip()


def resolve_call_name(
    user_id: str,
    display_name: str,
    profile: Optional[dict] = None,
    get_profile=None,
) -> str:
    """Pick the name a friend would use.

    profile: pre-fetched profile dict (avoids a second memory hit when the
    caller already has it). get_profile: callable(user_id)->dict, used only
    when profile is None — pass a LOCAL/fast lookup (JSON memory), never a
    blocking remote call from a hot path.
    """
    prof = profile
    if prof is None and get_profile is not None:
        try:
            prof = get_profile(str(user_id)) or {}
        except Exception:
            prof = {}
    real = (prof or {}).get("real_name", "") or ""
    if real and ("." in real or "_" in real or any(c.isdigit() for c in real)):
        # Handle-shaped junk stored as a "real name" ("thomas.codez",
        # "rishab06027") — fall back to the cleaned display name.
        real = ""
    if real:
        # "Aliyyan Khan" → "Aliyyan"; must look like an actual name
        cand = re.sub(r"[^\w'\- ]", "", str(real)).strip().split()
        if cand:
            first = cand[0].strip("_-'\"")
            if len(first) >= 2 and any(c.isalpha() for c in first):
                return first
    return clean_display_name(display_name)
