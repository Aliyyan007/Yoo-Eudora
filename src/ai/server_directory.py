"""Server facts directory — the ground truth for channels and the owner.

The reply engine kept hallucinating channel names ("post it in #art-uploads"
in a server with no such channel) and guessing who owns the place. This
module gives the prompt the list of channels that ACTUALLY exist, and a
post-filter that rewrites or drops invented #mentions before a reply goes
out.
"""
import re
import time

import discord
from loguru import logger
from rapidfuzz import fuzz

from . import d1_memory as mem


NO_CHANNEL_FALLBACK = "not sure which channel that'd be tbh, check the channel list"

_CACHE_TTL_S = 600
_channel_cache: dict = {}  # guild_id -> (timestamp, [(channel, normalized_name)])

# Unicode hyphen lookalikes Discord channel names love to use
_HYPHEN_VARIANTS = "‐‑‒–—"


def normalize_channel_name(name: str) -> str:
    """'💬┃general' -> 'general', 'art‑hub' -> 'art-hub'."""
    if not name:
        return ""
    n = name.lower()
    for h in _HYPHEN_VARIANTS:
        n = n.replace(h, "-")
    n = re.sub(r"[^a-z0-9-]", " ", n)
    tokens = [re.sub(r"-{2,}", "-", t).strip("-") for t in n.split()]
    return "-".join(t for t in tokens if t)


def invalidate(guild_id=None):
    """Drop the cached channel listing — one guild, or all when no id given."""
    if guild_id is None:
        _channel_cache.clear()
    else:
        _channel_cache.pop(guild_id, None)


def get_visible_channels(guild) -> list:
    """[(channel, normalized_name)] for every text/forum channel the bot can
    read, in guild order. Cached 10 min per guild."""
    if guild is None:
        return []
    ent = _channel_cache.get(guild.id)
    if ent and time.time() - ent[0] < _CACHE_TTL_S:
        return ent[1]
    forum_cls = getattr(discord, "ForumChannel", None)
    me = getattr(guild, "me", None)
    out = []
    for ch in getattr(guild, "channels", []) or []:
        if not isinstance(ch, discord.TextChannel) and not (
                forum_cls is not None and isinstance(ch, forum_cls)):
            continue
        try:
            if me is None or not ch.permissions_for(me).read_messages:
                continue
        except Exception:
            continue
        norm = normalize_channel_name(getattr(ch, "name", "") or "")
        if norm:
            out.append((ch, norm))
    _channel_cache[guild.id] = (time.time(), out)
    return out


def get_owner_name(guild) -> str:
    """Owner display name from the member cache — '' when not cached.
    Never makes an API call."""
    try:
        m = guild.get_member(guild.owner_id) or guild.owner
        return getattr(m, "display_name", "") or ""
    except Exception:
        return ""


_FACT_HINTS = re.compile(
    r"#|channel|where (?:can|do|should|to|would) (?:i|we|u|you)|post|share|"
    r"upload|partner|promo|advertis|owner|admin|\bmods?\b|staff|\brules?\b|"
    r"\broles?\b|verify|\bserver\b",
    re.IGNORECASE,
)


def wants_server_facts(text: str) -> bool:
    """True when the trigger message asks about channels/server/owner."""
    return bool(text) and bool(_FACT_HINTS.search(text))


def build_server_facts(guild, max_channels: int = 45) -> str:
    """The [SERVER FACTS] block prepended to the reply transcript."""
    owner = get_owner_name(guild) or "unknown — do not guess"
    chan_bits = []
    for ch, norm in get_visible_channels(guild)[:max_channels]:
        bit = f"#{norm}"
        try:
            topic = mem.get_channel_topic(str(ch.id)) or ""
        except Exception:
            topic = ""
        if topic:
            bit += f" (topic: {topic[:50]})"
        chan_bits.append(bit)
    return "\n".join([
        "[SERVER FACTS — these are the ONLY channels that exist here. If you "
        "mention a channel, write it as #name exactly as listed. Never invent "
        "channel names, roles or people; if you don't know, say you're not sure.]",
        f"server: {getattr(guild, 'name', '')}",
        f"owner: {owner}",
        "channels: " + ", ".join(chan_bits),
    ])


_EXISTING_MENTION = re.compile(r"<#(\d+)>")
_PLAIN_TOKEN = re.compile(r"(?<![<\w&])#([^\s#<>,.!?;:()\[\]\"']{2,40})")
_UNRESOLVED = "\x00"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _resolve_channel(token_norm: str, by_norm: dict, visible: list):
    """Exact -> fuzzy (>=85) -> containment (token len >= 4)."""
    ch = by_norm.get(token_norm)
    if ch is not None:
        return ch
    best_ch, best_score = None, 0.0
    for c, cn in visible:
        score = fuzz.ratio(token_norm, cn)
        if score > best_score:
            best_score, best_ch = score, c
    if best_score >= 85:
        return best_ch
    if len(token_norm) >= 4:
        for c, cn in visible:
            if token_norm in cn or cn in token_norm:
                return c
    return None


def ground_channel_mentions(text: str, guild) -> str:
    """Rewrite #name tokens to real <#id> mentions, then drop every sentence
    still referencing a channel that doesn't exist. '' when nothing survives
    (caller decides the fallback)."""
    if guild is None or not text:
        return text

    visible = get_visible_channels(guild)
    id_set = {ch.id for ch, _ in visible}
    by_norm: dict = {}
    for ch, norm in visible:
        by_norm.setdefault(norm, ch)

    events = [
        (m.start(), m.end(), "existing", m.group(1))
        for m in _EXISTING_MENTION.finditer(text)
    ]
    events += [
        (m.start(), m.end(), "plain", m.group(1))
        for m in _PLAIN_TOKEN.finditer(text)
    ]
    events.sort()

    parts = []
    pos = 0
    for start, end, kind, payload in events:
        parts.append(text[pos:start])
        if kind == "existing":
            if int(payload) in id_set:
                parts.append(text[start:end])
            else:
                parts.append(_UNRESOLVED)
        else:
            token_norm = normalize_channel_name(payload)
            if not any(c.isalpha() for c in token_norm):
                parts.append(text[start:end])  # '#2024' — not a channel ref
            else:
                ch = _resolve_channel(token_norm, by_norm, visible)
                if ch is not None:
                    parts.append(f"<#{ch.id}>")
                else:
                    parts.append(_UNRESOLVED)
        pos = end
    parts.append(text[pos:])
    new_text = "".join(parts)

    if _UNRESOLVED in new_text:
        kept = [s for s in _SENTENCE_SPLIT.split(new_text)
                if _UNRESOLVED not in s]
        new_text = " ".join(kept)
        logger.info(f"[facts] dropped hallucinated channel refs: '{text[:80]}' -> '{new_text[:80]}'")

    return new_text.strip()
