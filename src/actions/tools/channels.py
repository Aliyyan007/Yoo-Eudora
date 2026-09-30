"""Channel tools: discovery, fuzzy search, details, mentions, links.

All functions are async and take a :class:`ToolContext` as first argument.
They return JSON-serialisable dicts/lists so the agent can reason about them.
"""
from __future__ import annotations

import re
from typing import Any

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search

# discord.py-self channel type names
_TYPE_MAP = {
    discord.TextChannel: "text",
    discord.VoiceChannel: "voice",
    discord.StageChannel: "stage",
    discord.Thread: "thread",
    discord.ForumChannel: "forum",
    discord.CategoryChannel: "category",
    discord.DMChannel: "dm",
    discord.GroupChannel: "group",
}


def _channel_type(ch: discord.abc.GuildChannel) -> str:
    for cls, name in _TYPE_MAP.items():
        if isinstance(ch, cls):
            return name
    return "unknown"


def _channel_summary(ch: discord.abc.GuildChannel) -> dict[str, Any]:
    return {
        "id": ch.id,
        "name": ch.name,
        "type": _channel_type(ch),
        "category": ch.category.name if getattr(ch, "category", None) else None,
        "position": getattr(ch, "position", None),
        "mention": ch.mention,           # <#id>
        "jump_url": f"https://discord.com/channels/{ch.guild.id}/{ch.id}",
    }


def _channel_detail(ch: discord.abc.GuildChannel) -> dict[str, Any]:
    info = _channel_summary(ch)
    # text-ish channels have topic / nsfw / slowmode
    info["topic"] = getattr(ch, "topic", None)
    info["nsfw"] = getattr(ch, "nsfw", None)
    info["slowmode_delay"] = getattr(ch, "slowmode_delay", None)
    # Permission overwrites (who can post, who's muted, etc.)
    overwrites = getattr(ch, "overwrites", None)
    if overwrites:
        ow_list = []
        for target, ow in overwrites.items():
            name = target.name if hasattr(target, "name") else str(target)
            # Summarize key permissions instead of dumping all.
            perms = {}
            for attr in ("send_messages", "view_channel", "read_message_history",
                         "connect", "speak", "add_reactions", "attach_files",
                         "manage_messages"):
                val = getattr(ow, attr, None)
                if val is not None:  # True=allow, False=deny, None=inherit
                    perms[attr] = "allow" if val else "deny"
            if perms:
                ow_list.append({"target": name, "permissions": perms})
        if ow_list:
            info["permission_overwrites"] = ow_list
    # voice / stage
    if isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
        info["bitrate"] = getattr(ch, "bitrate", None)
        info["user_limit"] = getattr(ch, "user_limit", None)
        info["rtc_region"] = getattr(ch, "rtc_region", None)
        info["video_quality_mode"] = str(getattr(ch, "video_quality_mode", ""))
        info["member_count"] = len(getattr(ch, "members", []))
    if isinstance(ch, discord.StageChannel):
        info["stage_topic"] = getattr(ch, "topic", None)
    # threads
    if isinstance(ch, discord.Thread):
        info["parent_id"] = ch.parent_id
        info["archived"] = ch.archived
        info["locked"] = ch.locked
        info["auto_archive_duration"] = getattr(ch, "auto_archive_duration", None)
        info["member_count"] = ch.member_count
    # forum
    if isinstance(ch, discord.ForumChannel):
        info["active_threads"] = len(getattr(ch, "threads", []))
    return info


async def list_channels(ctx: ToolContext, type_filter: str | None = None) -> list[dict]:
    """List all channels in the guild, optionally filtered by type."""
    guild = ctx.require_guild()
    channels = guild.channels
    if type_filter:
        type_filter = type_filter.lower()
        channels = [c for c in channels if _channel_type(c) == type_filter]
    return [_channel_summary(c) for c in channels]


async def search_channels(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    """Fuzzy-search channel names. Returns ranked matches with scores."""
    guild = ctx.require_guild()
    results = fuzzy_search(query, guild.channels, key=lambda c: c.name, limit=limit)
    return [
        {**_channel_summary(r.item), "score": r.score}
        for r in results
    ]


async def get_channel_details(ctx: ToolContext, query: str) -> dict | list[dict]:
    """Get full details of a channel by name (fuzzy) or ID. Returns best match or list."""
    guild = ctx.require_guild()
    # Direct ID resolution (if query is a numeric ID or <#id> mention).
    ch = _try_direct_channel(ctx, query)
    if ch:
        return _channel_detail(ch)
    results = fuzzy_search(query, guild.channels, key=lambda c: c.name, limit=3)
    if not results:
        return {"error": f"No channel matching '{query}'."}
    if len(results) == 1 or results[0].score >= 85:
        return _channel_detail(results[0].item)
    # ambiguous — return top candidates
    return [
        {**_channel_summary(r.item), "score": r.score}
        for r in results
    ]


def _try_direct_channel(ctx: ToolContext, query: str):
    """If query is a numeric ID or <#id> mention, return the channel directly."""
    # Strip <#id> format
    m = re.match(r"<#(\d+)>", query.strip())
    cid = int(m.group(1)) if m else None
    if not cid and query.strip().isdigit():
        cid = int(query.strip())
    if cid:
        guild = ctx.require_guild()
        return guild.get_channel(cid)
    return None


async def resolve_channel(ctx: ToolContext, query: str) -> dict:
    """Resolve a fuzzy name or ID to a single channel (best match)."""
    # Direct ID resolution first.
    ch = _try_direct_channel(ctx, query)
    if ch:
        return {"id": ch.id, "name": ch.name, "mention": ch.mention, "type": _channel_type(ch)}
    guild = ctx.require_guild()
    results = fuzzy_search(query, guild.channels, key=lambda c: c.name, limit=1)
    if not results:
        return {"error": f"No channel matching '{query}'."}
    ch = results[0].item
    return {"id": ch.id, "name": ch.name, "mention": ch.mention, "type": _channel_type(ch)}


async def get_channel_mention(ctx: ToolContext, query: str) -> dict:
    """Return the #channel mention string (<#id>) for a fuzzy name."""
    res = await resolve_channel(ctx, query)
    if "error" in res:
        return res
    return {"mention": res["mention"], "name": res["name"], "id": res["id"]}


async def get_channel_link(ctx: ToolContext, query: str) -> dict:
    """Return a clickable jump link for a channel."""
    res = await resolve_channel(ctx, query)
    if "error" in res:
        return res
    guild = ctx.require_guild()
    return {
        "name": res["name"],
        "link": f"https://discord.com/channels/{guild.id}/{res['id']}",
    }
