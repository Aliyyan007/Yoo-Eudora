"""Scheduled-action tools — owner-only timed messaging.

"send X to #general every 5s until I say stop" → schedule_message with
interval_s=5. "stop it" → stop_scheduled matching by id/label/channel.
One-shots: interval_s=None + delay_s. Everything persists across restarts
(core/scheduler.py store) and deletes cleanly on cancel.
"""
from __future__ import annotations

import time

from src.actions.config.settings import settings
from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search


def _owner_only(ctx: ToolContext) -> dict | None:
    """Timed actions are owner-privileged — refuse anyone else."""
    if settings.owner_ids and not settings.is_owner(ctx.author_id):
        return {"error": "Only the owner can schedule actions."}
    return None


async def _resolve_text_channel(ctx: ToolContext, query: str):
    """Shared channel resolver — id / #mention / fuzzy name."""
    guild = ctx.require_guild()
    q = (query or "").strip().strip("<#>")
    if q.isdigit():
        ch = guild.get_channel(int(q))
        if ch:
            return ch
    if q.lower() in ("here", "this"):
        ch = ctx.get_current_channel()
        if ch:
            return ch
    chans = [c for c in guild.channels if hasattr(c, "send")]
    res = fuzzy_search(q, chans, key=lambda c: c.name, limit=1)
    return res[0].item if res else None


async def schedule_message(
    ctx: ToolContext,
    channel_query: str,
    content: str,
    delay_s: float = 0,
    interval_s: float = 0,
    label: str = "",
) -> dict:
    """Schedule a message: delay_s for the first send, interval_s to repeat
    forever until stopped (interval_s=0 = one-shot). Owner-only."""
    if e := _owner_only(ctx):
        return e
    from src.actions.core import scheduler as _sched_mod
    if _sched_mod.scheduler is None:
        return {"error": "Scheduler not running yet."}
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}
    delay_s = max(0.0, float(delay_s or 0))
    interval_s = max(0.0, float(interval_s or 0))
    tid = _sched_mod.scheduler.add({
        "kind": "send_message",
        "guild_id": ch.guild.id,
        "channel_id": ch.id,
        "channel_name": ch.name,
        "content": content,
        "interval_s": interval_s or None,
        "run_at": time.time() + delay_s,
        "created_by": ctx.author_id,
        "label": label or f"{content[:40]} -> #{ch.name}",
    })
    when = f"in {delay_s:.0f}s" if delay_s else "now"
    rep = f", repeating every {interval_s:.0f}s" if interval_s else ", one-shot"
    return {
        "ok": True, "task_id": tid, "channel": ch.name,
        "note": f"Scheduled {when}{rep}. Stop with stop_scheduled '{tid}'.",
    }


async def list_scheduled(ctx: ToolContext) -> dict:
    """List all active scheduled tasks."""
    if e := _owner_only(ctx):
        return e
    from src.actions.core import scheduler as _sched_mod
    if _sched_mod.scheduler is None:
        return {"tasks": []}
    now = time.time()
    return {"tasks": [
        {
            "id": t["id"], "channel": t.get("channel_name"),
            "label": t.get("label"), "content": t.get("content", "")[:80],
            "every_s": t.get("interval_s"),
            "next_in_s": round(max(0.0, t.get("run_at", 0) - now), 1),
        }
        for t in _sched_mod.scheduler.list()
    ]}


async def stop_scheduled(ctx: ToolContext, query: str = "") -> dict:
    """Cancel scheduled task(s) by id, label, or channel name. Empty query
    cancels ALL scheduled tasks. Owner-only."""
    if e := _owner_only(ctx):
        return e
    from src.actions.core import scheduler as _sched_mod
    if _sched_mod.scheduler is None:
        return {"error": "Scheduler not running."}
    q = (query or "").strip().lower()

    def _match(t: dict) -> bool:
        if not q:
            return True
        if t["id"] == q:
            return True
        blob = f"{t.get('label','')} {t.get('channel_name','')} {t.get('content','')}".lower()
        return q in blob

    killed = _sched_mod.scheduler.cancel(_match)
    if not killed:
        return {"ok": False, "error": f"No scheduled task matching '{query}'.",
                "tasks": (await list_scheduled(ctx)).get("tasks", [])}
    return {
        "ok": True, "cancelled": len(killed),
        "removed": [{"id": t["id"], "label": t.get("label")} for t in killed],
    }
