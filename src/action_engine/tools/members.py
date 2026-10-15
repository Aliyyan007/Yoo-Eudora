"""Member tools: details, fuzzy search, joins/leaves, online presence, pings.

IMPORTANT: `guild.members` is only the gateway-cached subset (the client
never chunks guilds), so lookups must fall back to server-side member
search (`query_members` / `query_recent_members`) when the cache misses —
otherwise real members resolve to fuzzy garbage like 'raus'.
"""
from __future__ import annotations

import re
from typing import Any

import discord
from rapidfuzz import fuzz

from src.action_engine.tools.context import ToolContext
from src.action_engine.utils.logger import logger

_STATUS_MAP = {
    discord.Status.online: "online",
    discord.Status.idle: "idle",
    discord.Status.dnd: "do_not_disturb",
    discord.Status.offline: "offline",
    discord.Status.invisible: "invisible",
}


def _status_name(status) -> str:
    return _STATUS_MAP.get(status, str(status))


def _member_summary(m: discord.Member) -> dict[str, Any]:
    return {
        "id": m.id,
        "display_name": m.display_name,
        "username": str(m),
        "nick": m.nick,
        "mention": m.mention,           # <@id>
        "bot": m.bot,
        "status": _status_name(m.status),
        "joined_at": m.joined_at.isoformat() if m.joined_at else None,
    }


def _member_detail(m: discord.Member) -> dict[str, Any]:
    info = _member_summary(m)
    info["avatar_url"] = str(m.display_avatar.url) if m.display_avatar else None
    info["banner"] = str(m.banner.url) if getattr(m, "banner", None) else None
    info["created_at"] = m.created_at.isoformat() if m.created_at else None
    info["roles"] = [r.name for r in m.roles if r.name != "@everyone"]
    info["top_role"] = m.top_role.name if m.top_role and m.top_role.name != "@everyone" else None
    info["activities"] = [
        (a.name if hasattr(a, "name") else str(a)) for a in m.activities
    ]
    info["is_in_voice"] = bool(m.voice)
    # Public profile bio (user profile) — best-effort, may be None
    bio = None
    try:
        bio = getattr(m, "public_flags", None)
    except Exception:
        pass
    info["public_flags"] = str(bio) if bio is not None else None
    return info


async def _maybe_fetch_profile(m: discord.Member) -> str | None:
    """Best-effort fetch of a member's profile bio.

    On discord.py-self 2.2 the real API is `await m.profile()` ->
    MemberProfile with `.bio`/`.guild_bio`/`.display_bio`.
    """
    try:
        prof = await m.profile()
        bio = (
            getattr(prof, "bio", None)
            or getattr(prof, "guild_bio", None)
            or getattr(prof, "display_bio", None)
            or getattr(prof, "about_me", None)
        )
        if bio:
            return bio
    except Exception as e:  # noqa: BLE001
        logger.debug(f"profile fetch failed for {m}: {e}")
    return None


async def list_members(ctx: ToolContext, limit: int = 100) -> list[dict]:
    """List guild members (capped)."""
    guild = ctx.require_guild()
    members = guild.members[:limit]
    return [_member_summary(m) for m in members]


async def identify_bots(ctx: ToolContext) -> list[dict]:
    """List all bot accounts in the guild (useful for finding bump bots, moderation bots, etc.)."""
    guild = ctx.require_guild()
    bots = [m for m in guild.members if m.bot]
    return [
        {
            "id": m.id,
            "name": m.display_name,
            "username": str(m),
            "mention": m.mention,
            "roles": [r.name for r in m.roles if r.name != "@everyone"],
        }
        for m in bots
    ]


def _try_direct_member(ctx: ToolContext, query: str):
    """If query is a numeric ID or <@id> mention, return the member directly."""
    m = re.match(r"<@!?(\d+)>", query.strip())
    uid = int(m.group(1)) if m else None
    if not uid and query.strip().isdigit():
        uid = int(query.strip())
    if uid:
        guild = ctx.require_guild()
        return guild.get_member(uid)
    return None


def _name_fields(m) -> list[str]:
    """Every name a member answers to: username, global display name,
    effective display name, guild nickname."""
    return [
        getattr(m, "name", "") or "",               # username
        getattr(m, "global_name", "") or "",        # account display name
        getattr(m, "display_name", "") or "",       # effective (nick/global/name)
        getattr(m, "nick", "") or "",               # guild nickname
    ]


def _exact_member(candidates, query: str):
    """Case-insensitive EXACT match across all name fields — username first,
    then global name, display name, nick. Always wins over fuzzy so a real
    exact-name target can never be stolen by a fuzzy near-miss."""
    q = (query or "").strip().casefold()
    if not q:
        return None
    for idx in range(4):
        for m in candidates:
            f = _name_fields(m)[idx]
            if f and f.casefold() == q:
                return m
    return None


async def _server_member_search(guild, query: str, limit: int = 10) -> list:
    """Gateway member search — hits Discord's member index, not just the
    (sparse) local cache. Safe no-op on failure."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        return await guild.query_members(
            q, limit=max(5, min(100, limit)), cache=True)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"server member search failed for '{q}': {e}")
        return []


def _name_score(query: str, target: str) -> float:
    """token_set + partial — but partial ONLY in the legitimate direction
    (query ⊆ target, e.g. 'ris' → 'rishab'). The reverse (a short member
    name inside a long query — 'raus' ⊂ 'sweetbutdangeROUS28') is how the
    wrong person used to get pinged."""
    if not target:
        return 0.0
    score = float(fuzz.token_set_ratio(query, target))
    if len(query.strip()) <= len(target.strip()):
        score = max(score, float(fuzz.partial_ratio(query, target)))
    return score


def _best_member_score(query: str, m) -> tuple[float, str]:
    best, field = 0.0, ""
    for f in _name_fields(m):
        s = _name_score(query, f)
        if s > best:
            best, field = s, f
    return best, field


def _member_pool(ctx, guild, fetched: list) -> dict:
    """Union of cached + server-fetched members, keyed by id."""
    pool = {m.id: m for m in guild.members}
    for m in fetched:
        pool[m.id] = m
    return pool


async def _find_member(ctx: ToolContext, query: str):
    """Full confidence-ranked resolution:
    1. ID/mention
    2. exact name match (cache)
    3. server-side search → exact match
    4. directional fuzzy — only if confident AND clearly ahead of #2
    Returns (member_or_None, ranked_list)."""
    guild = ctx.require_guild()
    m = _try_direct_member(ctx, query)
    if m:
        return m, [(100.0, m)]

    pool = _member_pool(ctx, guild, [])
    hit = _exact_member(pool.values(), query)
    if hit is not None:
        return hit, [(100.0, hit)]

    # Cache missed — ask the server (gateway OP 8 member search).
    fetched = await _server_member_search(guild, query)
    if fetched:
        pool = _member_pool(ctx, guild, fetched)
        hit = _exact_member(fetched, query)
        if hit is not None:
            return hit, [(100.0, hit)]

    # idx tiebreak — Member objects aren't orderable; equal scores would
    # crash sorted() on the second element.
    scored = []
    for i, m in enumerate(pool.values()):
        s, _field = _best_member_score(query, m)
        scored.append((s, i, m))
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    ranked = [(s, m) for s, _, m in scored if s >= 40.0]
    return None, ranked


async def search_members(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    """Search members by display name / username / nick. Accepts IDs.
    Merges the sparse local cache with a server-side member search so
    names that were never cached still resolve."""
    guild = ctx.require_guild()
    m = _try_direct_member(ctx, query)
    if m:
        return [_member_summary(m)]

    fetched = await _server_member_search(guild, query, limit=max(limit, 5))
    pool = _member_pool(ctx, guild, fetched)

    scored = []
    for i, mm in enumerate(pool.values()):
        s, field = _best_member_score(query, mm)
        if s >= 40.0:
            scored.append((s, i, field, mm))   # idx tiebreak: Members unorderable
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [{**_member_summary(mm), "score": s, "matched_on": field}
            for s, _, field, mm in scored[:limit]]


async def get_member_details(ctx: ToolContext, query: str) -> dict | list[dict]:
    """Get full details for a member by name or ID. Includes bio if available."""
    hit, ranked = await _find_member(ctx, query)
    if hit is not None:
        detail = _member_detail(hit)
        detail["bio"] = await _maybe_fetch_profile(hit)
        return detail
    if not ranked:
        return {"error": f"No member matching '{query}'."}
    if len(ranked) == 1 or ranked[0][0] >= 85:
        m = ranked[0][1]
        detail = _member_detail(m)
        detail["bio"] = await _maybe_fetch_profile(m)
        return detail
    return [{**_member_summary(m), "score": s} for s, m in ranked[:3]]


async def resolve_member(ctx: ToolContext, query: str) -> dict:
    """Resolve a name or ID to a single member. FAILS SAFE: a weak or
    ambiguous fuzzy guess returns an error rather than pinging the wrong
    person (the 'raus' bug)."""
    hit, ranked = await _find_member(ctx, query)
    if hit is not None:
        return {"id": hit.id, "name": hit.display_name, "mention": hit.mention}
    # Only accept a fuzzy winner when it's clearly ahead of the runner-up —
    # a 2-point gap on 50-70 scores is a coin flip, not a match.
    if ranked and ranked[0][0] >= 70.0 and (
            len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 5.0):
        m = ranked[0][1]
        return {"id": m.id, "name": m.display_name, "mention": m.mention}
    if ranked:
        opts = ", ".join(f"{m.display_name} (@{m.name})" for _, m in ranked[:3])
        return {"error": f"Ambiguous '{query}' — possible: {opts}. "
                         "Ask which one (or use their exact username)."}
    return {"error": f"No member matching '{query}'."}


async def get_member_mention(ctx: ToolContext, query: str) -> dict:
    """Return the @user mention string (<@id>) for a fuzzy name."""
    res = await resolve_member(ctx, query)
    if "error" in res:
        return res
    return {"mention": res["mention"], "name": res["name"], "id": res["id"]}


async def get_member_count(ctx: ToolContext) -> dict:
    """Total member count + online/idle/dnd/offline breakdown."""
    guild = ctx.require_guild()
    counts = {"online": 0, "idle": 0, "do_not_disturb": 0, "offline": 0, "invisible": 0}
    for m in guild.members:
        counts[_status_name(m.status)] = counts.get(_status_name(m.status), 0) + 1
    return {
        "total": guild.member_count,
        "by_status": counts,
    }


async def get_online_members(ctx: ToolContext) -> list[dict]:
    """List members currently online (online/idle/dnd)."""
    guild = ctx.require_guild()
    online = [
        _member_summary(m)
        for m in guild.members
        if m.status in (discord.Status.online, discord.Status.idle, discord.Status.dnd)
    ]
    return online


async def get_recent_joins(ctx: ToolContext, limit: int = 10) -> list[dict]:
    """Members who joined most recently (sorted by joined_at desc).

    The local member cache is sparse, so this first asks the server for
    the genuinely-recent joins (query_recent_members) and merges with the
    cache — otherwise "who just joined" silently answered from a stale
    handful of cached members."""
    guild = ctx.require_guild()
    pool = {m.id: m for m in guild.members if getattr(m, "joined_at", None)}
    try:
        recents = await guild.query_recent_members(
            limit=max(5, min(50, limit)), cache=True)
        for m in recents:
            if getattr(m, "joined_at", None):
                pool[m.id] = m
        logger.info(f"[members] recent-joins: server returned "
                    f"{len(recents or [])} member(s)")
    except Exception as e:  # noqa: BLE001
        logger.info(f"[members] query_recent_members failed "
                    f"({type(e).__name__}: {e}) — cache fallback")
    # The bot's own live-observed joins (gateway member_join events seen
    # since boot) — the freshest join signal there is.
    try:
        seen = getattr(ctx.bot, "_recent_joins", None) or {}
        for uid in seen:
            m = pool.get(uid) or guild.get_member(uid)
            if m is not None and getattr(m, "joined_at", None):
                pool[m.id] = m
    except Exception:
        pass
    joined = sorted(pool.values(), key=lambda m: m.joined_at, reverse=True)
    return [_member_summary(m) for m in joined[:limit]]


async def get_recent_leaves(ctx: ToolContext, limit: int = 10) -> list[dict]:
    """Members who recently left (from the bot's in-memory leave log).

    The bot records leaves in `ctx.bot.engager_leaves` (see core/events.py).
    If none recorded yet, returns an empty list.
    """
    leaves: list = getattr(ctx.bot, "engager_leaves", [])
    return leaves[-limit:][::-1] if leaves else []
