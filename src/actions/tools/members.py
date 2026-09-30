"""Member tools: details, fuzzy search, joins/leaves, online presence, pings."""
from __future__ import annotations

import re
from typing import Any

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search
from src.actions.utils.logger import logger

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


async def search_members(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    """Fuzzy-search members by display name OR username. Also accepts IDs."""
    guild = ctx.require_guild()
    # Direct ID resolution.
    m = _try_direct_member(ctx, query)
    if m:
        return [_member_summary(m)]
    members = guild.members

    # search by display_name first
    by_disp = fuzzy_search(query, members, key=lambda m: m.display_name, limit=limit)
    by_user = fuzzy_search(query, members, key=lambda m: str(m), limit=limit)

    # merge & dedupe by id, keep best score
    seen: dict[int, dict] = {}
    for r in by_disp + by_user:
        mid = r.item.id
        if mid not in seen or r.score > seen[mid]["score"]:
            seen[mid] = {**_member_summary(r.item), "score": r.score}
    ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
    return ranked[:limit]


async def get_member_details(ctx: ToolContext, query: str) -> dict | list[dict]:
    """Get full details for a member by fuzzy name or ID. Includes bio if available."""
    guild = ctx.require_guild()
    # Direct ID resolution.
    m = _try_direct_member(ctx, query)
    if m:
        detail = _member_detail(m)
        detail["bio"] = await _maybe_fetch_profile(m)
        return detail
    results = fuzzy_search(query, guild.members, key=lambda m: m.display_name, limit=3)
    if not results:
        return {"error": f"No member matching '{query}'."}
    if len(results) == 1 or results[0].score >= 85:
        m = results[0].item
        detail = _member_detail(m)
        detail["bio"] = await _maybe_fetch_profile(m)
        return detail
    return [{**_member_summary(r.item), "score": r.score} for r in results]


async def resolve_member(ctx: ToolContext, query: str) -> dict:
    """Resolve a fuzzy name or ID to a single member (best match)."""
    # Direct ID resolution first.
    m = _try_direct_member(ctx, query)
    if m:
        return {"id": m.id, "name": m.display_name, "mention": m.mention}
    guild = ctx.require_guild()
    results = fuzzy_search(query, guild.members, key=lambda m: m.display_name, limit=1)
    if not results:
        return {"error": f"No member matching '{query}'."}
    m = results[0].item
    return {"id": m.id, "name": m.display_name, "mention": m.mention}


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
    """Members who joined most recently (sorted by joined_at desc)."""
    guild = ctx.require_guild()
    joined = [m for m in guild.members if m.joined_at]
    joined.sort(key=lambda m: m.joined_at, reverse=True)
    return [_member_summary(m) for m in joined[:limit]]


async def get_recent_leaves(ctx: ToolContext, limit: int = 10) -> list[dict]:
    """Members who recently left (from the bot's in-memory leave log).

    The bot records leaves in `ctx.bot.engager_leaves` (see core/events.py).
    If none recorded yet, returns an empty list.
    """
    leaves: list = getattr(ctx.bot, "engager_leaves", [])
    return leaves[-limit:][::-1] if leaves else []
