"""Sticker & GIF tools.

GIFs use the official Klipy GIF API (https://api.klipy.com). This requires
the KLIPY_API_KEY env var (loaded as settings.klipy_api_key). The returned
GIF URL auto-embeds when sent as message content in Discord.

Stickers are sent via the `stickers` parameter of `Message.send`.
"""
from __future__ import annotations

import discord
import httpx

from src.action_engine.tools.context import ToolContext
from src.action_engine.tools.messaging import _resolve_text_channel
from src.action_engine.utils.fuzzy import best_match
from src.action_engine.utils.logger import logger
from src.action_engine.config.settings import settings


# ---------------------------------------------------------------- GIF search
async def _search_klipy_gifs(ctx: ToolContext, query: str, limit: int = 5) -> list[dict]:
    """Search GIFs via the official Klipy GIF API.

    Uses settings.klipy_api_key. Klipy requires per_page >= 8, so we clamp
    the request to at least 8 and slice the results locally to the requested
    limit. GIF URL priority: file.md.gif.url, then file.hd.gif.url, then
    file.sm.gif.url.
    """
    key = settings.klipy_api_key
    if not key:
        logger.warning("KLIPY_API_KEY not set; gif search unavailable")
        return []
    # Klipy requires per_page >= 8 (max 50). Fetch at least 8, slice locally.
    per_page = max(8, min(50, limit))
    url = f"https://api.klipy.com/api/v1/{key}/gifs/search"
    params = {
        "q": query,
        "per_page": per_page,
        "rating": "pg",
        "locale": "us_US",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Klipy gif search failed ({type(e).__name__})")
        return []
    # Klipy nests the gif list under data["data"]["data"].
    items = (data.get("data") or {}).get("data") or []
    out = []
    for item in items[:limit]:
        file_obj = item.get("file") or {}
        # Priority: md (medium) -> hd (high) -> sm (small).
        gif_url = None
        for quality in ("md", "hd", "sm"):
            q = file_obj.get(quality) or {}
            g = q.get("gif") or {}
            if g.get("url"):
                gif_url = g["url"]
                break
        if gif_url:
            out.append({
                "url": gif_url,
                "title": item.get("title", query),
            })
    return out


async def search_gifs(ctx: ToolContext, query: str, limit: int = 5) -> dict:
    """Search for GIFs by keyword. Returns candidate gif URLs."""
    results = await _search_klipy_gifs(ctx, query, limit)
    if not results:
        return {"error": f"No gifs found for '{query}'."}
    return {"query": query, "gifs": results}


async def trending_gifs(ctx: ToolContext, limit: int = 5) -> dict:
    """Get trending GIFs from Klipy. Returns candidate gif URLs."""
    key = settings.klipy_api_key
    if not key:
        return {"error": "KLIPY_API_KEY not set"}
    # Klipy requires per_page >= 8 (max 50). Fetch at least 8, slice locally.
    per_page = max(8, min(50, limit))
    url = f"https://api.klipy.com/api/v1/{key}/gifs/trending"
    params = {
        "per_page": per_page,
        "rating": "pg",
        "locale": "us_US",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Klipy trending gifs failed ({type(e).__name__})")
        return {"error": f"Trending gifs request failed"}
    items = (data.get("data") or {}).get("data") or []
    out = []
    for item in items[:limit]:
        file_obj = item.get("file") or {}
        gif_url = None
        for quality in ("md", "hd", "sm"):
            q = file_obj.get(quality) or {}
            g = q.get("gif") or {}
            if g.get("url"):
                gif_url = g["url"]
                break
        if gif_url:
            out.append({
                "url": gif_url,
                "title": item.get("title", "trending"),
            })
    if not out:
        return {"error": "No trending gifs found."}
    return {"gifs": out}


async def send_gif(
    ctx: ToolContext,
    channel_query: str,
    query: str,
    caption: str | None = None,
    ping_users: list[str] | None = None,
    ping_roles: list[str] | None = None,
) -> dict:
    """Search a gif and send it to a channel. The URL auto-embeds in Discord.

    Supports optional ping_users and ping_roles (same as send_message).
    """
    gifs = await _search_klipy_gifs(ctx, query, 1)
    if not gifs:
        return {"error": f"No gifs found for '{query}'."}
    gif_url = gifs[0]["url"]

    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}
    try:
        # Build the text: optional pings + caption + gif URL.
        parts = []
        if ping_users:
            from src.action_engine.tools.members import resolve_member
            for uq in ping_users:
                res = await resolve_member(ctx, uq)
                if "id" in res:
                    parts.append(f"<@{res['id']}>")
        if ping_roles:
            from src.action_engine.tools.roles import resolve_role_sync
            for rq in ping_roles:
                res = resolve_role_sync(ctx, rq)
                if res and "id" in res:
                    parts.append(f"<@&{res['id']}>")
        if caption:
            parts.append(caption)
        parts.append(gif_url)
        text = " ".join(parts)
        sent = await ch.send(text)
    except discord.HTTPException as e:
        return {"error": f"Send failed: {e}"}
    return {"ok": True, "channel": ch.name, "gif_url": gif_url, "message_id": sent.id}


async def send_multiple_gifs(
    ctx: ToolContext, channel_query: str, query: str, count: int = 5
) -> dict:
    """Search and send multiple different gifs to a channel."""
    gifs = await _search_klipy_gifs(ctx, query, count)
    if not gifs:
        return {"error": f"No gifs found for '{query}'."}

    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}

    sent_ids = []
    for g in gifs[:count]:
        try:
            sent = await ch.send(g["url"])
            sent_ids.append(sent.id)
        except discord.HTTPException as e:
            logger.warning(f"Gif send failed: {e}")
    return {"ok": True, "channel": ch.name, "count": len(sent_ids), "message_ids": sent_ids}


# ---------------------------------------------------------------- Stickers
async def list_stickers(ctx: ToolContext, limit: int = 20) -> list[dict]:
    """List the guild's custom stickers."""
    guild = ctx.require_guild()
    stickers = list(guild.stickers)[:limit]
    return [
        {
            "id": s.id,
            "name": s.name,
            "format": str(s.format),
            "url": str(s.url) if hasattr(s, "url") else None,
        }
        for s in stickers
    ]


async def send_sticker(
    ctx: ToolContext, channel_query: str, sticker_query: str = "random"
) -> dict:
    """Send a guild sticker to a channel.

    If sticker_query is 'random', picks a random sticker from the guild.
    Otherwise fuzzy-matches the sticker name.
    """
    guild = ctx.require_guild()
    stickers = list(guild.stickers)
    if not stickers:
        return {"error": "This server has no custom stickers."}

    if sticker_query.lower().strip() in ("random", "any", "rand"):
        import random
        sticker = random.choice(stickers)
    else:
        r = best_match(sticker_query, stickers, key=lambda s: s.name, score_cutoff=40)
        if r is None:
            return {"error": f"No sticker matching '{sticker_query}'. Available: {', '.join(s.name for s in stickers)}"}
        sticker = r.item

    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}
    try:
        sent = await ch.send(stickers=[sticker])
    except discord.HTTPException as e:
        return {"error": f"Sticker send failed: {e}"}
    return {"ok": True, "sticker": sticker.name, "channel": ch.name, "message_id": sent.id}


async def send_multiple_stickers(
    ctx: ToolContext, channel_query: str, count: int = 5
) -> dict:
    """Send multiple different random stickers to a channel."""
    guild = ctx.require_guild()
    stickers = list(guild.stickers)
    if not stickers:
        return {"error": "This server has no custom stickers."}

    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}

    import random
    # Pick N different stickers (or all if fewer than N).
    to_send = random.sample(stickers, min(count, len(stickers)))
    sent_ids = []
    sent_names = []
    for sticker in to_send:
        try:
            sent = await ch.send(stickers=[sticker])
            sent_ids.append(sent.id)
            sent_names.append(sticker.name)
        except discord.HTTPException as e:
            logger.warning(f"Sticker send failed: {e}")
    return {"ok": True, "channel": ch.name, "count": len(sent_ids), "stickers": sent_names}
