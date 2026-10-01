"""Self-cleanup — don't leave walls of own messages in a channel.

Ported from the reference bot's actions/self_cleanup.py: a bot message is
"unanswered" if no human message came after it, it's not the newest in the
channel, and it's older than 10 min. Before each send we sweep those (keep
the most recent one). A periodic pass also deletes any of ours older than
6h or excess beyond 15 per channel.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

import discord

from src.action_engine.utils.logger import logger

UNANSWERED_MIN_AGE_S = 600          # only delete msgs older than 10 min
MAX_AGE_H = 6                       # periodic: delete ours older than 6h
MAX_PER_CHANNEL = 15                # periodic: cap our msgs per channel
_sweep_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def _is_ours(bot, msg: discord.Message) -> bool:
    return bot.user is not None and msg.author.id == bot.user.id


async def sweep_unanswered(bot, channel, *, keep_newest: bool = True) -> int:
    """Delete our own unanswered messages in `channel`. Called right
    before the bot sends — so a dead channel doesn't pile up bot walls.
    Returns count deleted."""
    if not hasattr(channel, "history"):
        return 0
    async with _sweep_locks[channel.id]:
        try:
            msgs = []
            async for m in channel.history(limit=20, oldest_first=False):
                msgs.append(m)
            msgs.reverse()                       # chronological
        except Exception:  # noqa: BLE001
            return 0

        # find the last HUMAN message index — everything of ours after it
        # is "unanswered"
        last_human = -1
        for i, m in enumerate(msgs):
            if not _is_ours(bot, m) and not m.author.bot:
                last_human = i
        now = discord.utils.utcnow().timestamp()
        mine_unanswered = [
            m for i, m in enumerate(msgs)
            if i > last_human and _is_ours(bot, m)
            and i < len(msgs) - 1                    # never the latest msg
            and now - m.created_at.timestamp() > UNANSWERED_MIN_AGE_S
        ]
        if keep_newest and mine_unanswered:
            mine_unanswered = mine_unanswered[:-1]
        deleted = 0
        for m in mine_unanswered:
            try:
                await m.delete()
                deleted += 1
                await asyncio.sleep(1.2)
            except discord.Forbidden:
                break
            except Exception:  # noqa: BLE001
                pass
        if deleted:
            logger.info(f"Self-cleanup: deleted {deleted} unanswered "
                        f"msg(s) in #{getattr(channel, 'name', channel.id)}")
        return deleted


async def sweep_channel(bot, channel) -> int:
    """Periodic hygiene: delete ours older than 6h + excess beyond 15."""
    if not hasattr(channel, "history"):
        return 0
    try:
        mine = []
        async for m in channel.history(limit=100):
            if _is_ours(bot, m):
                mine.append(m)
    except Exception:  # noqa: BLE001
        return 0
    now = time.time()
    stale = [m for m in mine
             if now - m.created_at.timestamp() > MAX_AGE_H * 3600]
    mine.sort(key=lambda m: m.created_at)
    excess = mine[:-MAX_PER_CHANNEL] if len(mine) > MAX_PER_CHANNEL else []
    to_del = {m.id for m in stale} | {m.id for m in excess}
    deleted = 0
    for m in mine:
        if m.id in to_del:
            try:
                await m.delete()
                deleted += 1
                await asyncio.sleep(1.2)
            except discord.Forbidden:
                break
            except Exception:  # noqa: BLE001
                pass
    return deleted
