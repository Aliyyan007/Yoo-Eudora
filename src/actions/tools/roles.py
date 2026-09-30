"""Role tools: search roles, get role mentions for pinging.

In Discord, a role ping is `<@&role_id>`. For self-bots, this works
when included in the message content — no `allowed_mentions` needed
(user accounts don't have the bot allowed_mentions restriction).
"""
from __future__ import annotations

import re

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search


async def search_roles(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    """Fuzzy-search roles by name. Returns ranked matches."""
    guild = ctx.require_guild()
    results = fuzzy_search(query, guild.roles, key=lambda r: r.name, limit=limit)
    return [
        {
            "id": r.item.id,
            "name": r.item.name,
            "mention": r.item.mention,
            "color": str(r.item.color),
            "member_count": len(r.item.members),
            "score": r.score,
        }
        for r in results
    ]


async def get_role_mention(ctx: ToolContext, query: str) -> dict:
    """Get the <@&role_id> mention string for a fuzzy role name."""
    guild = ctx.require_guild()
    results = fuzzy_search(query, guild.roles, key=lambda r: r.name, limit=1)
    if not results:
        return {"error": f"No role matching '{query}'."}
    role = results[0].item
    return {"mention": role.mention, "name": role.name, "id": role.id}


def resolve_role_sync(ctx: ToolContext, query: str) -> dict | None:
    """Synchronous role resolution. Returns {id, name, mention} or None."""
    guild = ctx.require_guild()
    # Direct ID or <@&id> format
    m = re.match(r"<@&(\d+)>", query.strip())
    rid = int(m.group(1)) if m else None
    if not rid and query.strip().isdigit():
        rid = int(query.strip())
    if rid:
        role = guild.get_role(rid)
        if role:
            return {"id": role.id, "name": role.name, "mention": role.mention}
    results = fuzzy_search(query, guild.roles, key=lambda r: r.name, limit=1)
    if not results:
        return None
    role = results[0].item
    return {"id": role.id, "name": role.name, "mention": role.mention}
