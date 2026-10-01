"""Message tools: read latest N messages, serialise embeds to JSON, describe
attachments/images/gifs via Groq vision so the agent can "see" them."""
from __future__ import annotations

from typing import Any

import discord

from src.action_engine.core.groq_pool import get_pool
from src.action_engine.tools.context import ToolContext
from src.action_engine.utils.embeds import embed_to_json
from src.action_engine.utils.fuzzy import fuzzy_search
from src.action_engine.config.settings import settings


def _parse_link_ids(link: str) -> tuple[int, int]:
    """Parse a Discord message link, return (channel_id, message_id)."""
    parts = [p for p in link.split("/") if p.isdigit()]
    if len(parts) < 2:
        raise ValueError("Invalid message link")
    return int(parts[-2]), int(parts[-1])


async def _fetch_message_from_link(ctx: ToolContext, link: str) -> discord.Message:
    """Fetch a message from a Discord jump link."""
    cid, mid = _parse_link_ids(link)
    ch = ctx.bot.get_channel(cid) or await ctx.bot.fetch_channel(cid)
    return await ch.fetch_message(mid)


def _attachment_kind(att: discord.Attachment) -> str:
    ct = (att.content_type or "").lower()
    if ct.startswith("image/gif") or (att.filename or "").lower().endswith(".gif"):
        return "gif"
    if ct.startswith("image"):
        return "image"
    if ct.startswith("video"):
        return "video"
    if ct.startswith("audio"):
        return "audio"
    return "file"


async def _describe_attachment(att: discord.Attachment) -> str:
    """Use Groq vision to describe an image/gif attachment."""
    pool = get_pool()
    kind = _attachment_kind(att)
    url = att.url
    if kind in ("image", "gif"):
        prompt = (
            "Describe this Discord image/gif attachment in one short sentence "
            "for an AI agent's context. Note if it is a meme, screenshot, photo, "
            "emoji sheet, etc."
        )
        return await pool.transcribe_image(url, prompt)
    if kind == "video":
        return f"[video attachment: {att.filename}]"
    if kind == "audio":
        return f"[audio attachment: {att.filename}]"
    return f"[file attachment: {att.filename} ({att.size} bytes)]"


def _sticker_info(sticker: discord.StickerItem | discord.Sticker) -> dict:
    return {
        "id": sticker.id,
        "name": sticker.name,
        "format": str(getattr(sticker, "format", "")),
        "url": str(sticker.url) if hasattr(sticker, "url") else None,
    }


async def _serialise_message(msg: discord.Message, with_vision: bool = True) -> dict[str, Any]:
    """Turn a Message into a JSON-able dict, enriching attachments via vision."""
    content = msg.content or ""
    attachments: list[dict] = []
    for att in msg.attachments:
        kind = _attachment_kind(att)
        entry = {
            "filename": att.filename,
            "kind": kind,
            "url": att.url,
            "size": att.size,
        }
        if with_vision and kind in ("image", "gif"):
            try:
                entry["description"] = await _describe_attachment(att)
            except Exception as e:  # noqa: BLE001
                entry["description"] = f"[vision failed: {e}]"
        attachments.append(entry)

    embeds = [embed_to_json(e) for e in msg.embeds]
    stickers = [_sticker_info(s) for s in (msg.stickers or [])]
    reactions = [
        {"emoji": str(r.emoji), "count": r.count}
        for r in msg.reactions
    ] if msg.reactions else []

    return {
        "id": msg.id,
        "author": {
            "id": msg.author.id,
            "name": msg.author.display_name,
            "username": str(msg.author),
            "mention": msg.author.mention,
        },
        "content": content,
        "attachments": attachments,
        "embeds": embeds,
        "stickers": stickers,
        "reactions": reactions,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
        "edited_at": msg.edited_at.isoformat() if msg.edited_at else None,
        "pinned": msg.pinned,
        "jump_url": msg.jump_url,
        "referenced": (
            {
                "author": msg.reference.resolved.author.display_name,
                "content": msg.reference.resolved.content,
            }
            if msg.reference and isinstance(msg.reference.resolved, discord.Message)
            else None
        ),
    }


async def _resolve_channel_obj(ctx: ToolContext, query: str) -> discord.TextChannel | discord.Thread | None:
    guild = ctx.require_guild()
    # restrict to text-like channels
    text_like = [
        c for c in guild.channels
        if isinstance(c, (discord.TextChannel, discord.Thread, discord.ForumChannel))
    ]
    results = fuzzy_search(query, text_like, key=lambda c: c.name, limit=1)
    if not results:
        return None
    ch = results[0].item
    if isinstance(ch, discord.ForumChannel):
        # pick the first thread if it's a forum
        threads = list(ch.threads)
        return threads[0] if threads else None
    return ch


async def get_recent_messages(
    ctx: ToolContext, channel_query: str, limit: int | None = None
) -> dict:
    """Read the latest N messages of a text channel (default from settings).

    Embeds are returned as JSON; image/gif attachments are described via Groq
    vision so the agent understands their nature.
    """
    limit = limit or settings.message_history_limit
    ch = await _resolve_channel_obj(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}

    try:
        history = [m async for m in ch.history(limit=limit)]
    except discord.Forbidden:
        return {"error": f"No permission to read history in #{ch.name}."}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to read #{ch.name}: {e}"}

    # history is newest-first; reverse so oldest is first (chronological)
    history.reverse()
    serialised = [await _serialise_message(m) for m in history]
    return {
        "channel": {"id": ch.id, "name": ch.name, "mention": ch.mention},
        "count": len(serialised),
        "messages": serialised,
    }


async def get_message_by_link(ctx: ToolContext, link: str) -> dict:
    """Fetch a single message by its Discord jump link.

    Link format: https://discord.com/channels/<guild>/<channel>/<message>
    """
    try:
        msg = await _fetch_message_from_link(ctx, link)
    except Exception as e:  # noqa: BLE001
        return {"error": f"Could not fetch message: {e}"}
    return await _serialise_message(msg)
