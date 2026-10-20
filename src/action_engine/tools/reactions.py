"""Reaction tools: add emoji reactions to messages."""
from __future__ import annotations

import discord

from src.action_engine.tools.context import ToolContext


def _resolve_emoji_payload(ctx: ToolContext, emoji: str) -> str:
    """Resolve an emoji name to a usable payload (custom emoji ID or unicode).

    Guild-emoji name resolution runs FIRST — a 1-2 char custom emoji name
    (":)", "ok") used to be swallowed by the unicode shortcut and never
    resolved. Plain unicode emoji fall through untouched.
    """
    emoji = str(emoji or "")
    if not emoji:
        return emoji
    if emoji.startswith("<"):
        return emoji  # already a <:name:id>/<a:name:id> payload
    try:
        guild = ctx.require_guild()
    except Exception:  # noqa: BLE001
        guild = None
    if guild is not None:
        match = next((e for e in guild.emojis if e.name.lower() == emoji.lower()), None)
        if match:
            return str(match)
        from src.action_engine.utils.fuzzy import best_match
        r = best_match(emoji, guild.emojis, key=lambda e: e.name, score_cutoff=50)
        if r:
            return str(r.item)
    return emoji  # unicode emoji or unresolvable — pass through


async def react_to_message(
    ctx: ToolContext, message_link: str, emoji: str
) -> dict:
    """Add an emoji reaction to a message identified by its jump link.

    `emoji` can be a unicode emoji ("🔥", "👍") or a custom guild emoji name
    ("pepe") which is resolved from the guild's emoji list.
    """
    payload = _resolve_emoji_payload(ctx, emoji)

    # Single fetch → react. The old path fetched the message twice AND ran
    # a Groq-vision serialisation purely for error-checking.
    try:
        p = [x for x in message_link.split("/") if x.isdigit()]
        cid, mid = int(p[-2]), int(p[-1])
        ch = ctx.bot.get_channel(cid) or await ctx.bot.fetch_channel(cid)
        msg = await ch.fetch_message(mid)
        await msg.add_reaction(payload)
    except discord.HTTPException as e:
        return {"error": f"Reaction failed: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Could not fetch message: {e}"}
    return {"ok": True, "emoji": payload, "message_link": message_link}


async def react_to_user_latest(
    ctx: ToolContext, channel_query: str, user_query: str, emoji: str
) -> dict:
    """React to the latest message from a given user in a channel."""
    from src.action_engine.tools.messages import _resolve_channel_obj, _serialise_message
    from src.action_engine.tools.members import resolve_member

    res = await resolve_member(ctx, user_query)
    if "error" in res:
        return res
    uid = res["id"]

    ch = await _resolve_channel_obj(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}

    target = None
    async for m in ch.history(limit=50):
        if m.author.id == uid:
            target = m
            break
    if target is None:
        return {"error": f"No recent message from {res['name']} in #{ch.name}."}

    payload = _resolve_emoji_payload(ctx, emoji)

    try:
        await target.add_reaction(payload)
    except discord.HTTPException as e:
        return {"error": f"Reaction failed: {e}"}
    return {"ok": True, "emoji": payload, "user": res["name"], "channel": ch.name}


async def react_to_recent(
    ctx: ToolContext, channel_query: str, emoji: str, count: int = 5
) -> dict:
    """React to the N most recent messages in a channel with an emoji.

    This is a batch operation — much more efficient than calling
    react_to_message N times.
    """
    from src.action_engine.tools.messages import _resolve_channel_obj

    ch = await _resolve_channel_obj(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}

    payload = _resolve_emoji_payload(ctx, emoji)

    reacted = []
    failed = []
    async for m in ch.history(limit=count):
        try:
            await m.add_reaction(payload)
            reacted.append(m.id)
        except discord.HTTPException as e:
            failed.append({"message_id": m.id, "error": str(e)})

    return {
        "ok": len(reacted) > 0,
        "channel": ch.name,
        "emoji": payload,
        "reacted_count": len(reacted),
        "reacted_message_ids": reacted,
        "failed": failed,
    }
