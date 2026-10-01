"""Persistent scheduled actions — timed tasks that survive restarts.

Owner asks things like "send X to #general every 5 seconds until I say
stop" or "at 9pm ping the mods". Tasks live in ``data/scheduled_tasks.json``;
the scheduler wakes every second, executes whatever's due, reschedules
interval tasks and deletes one-shots after they fire. On boot, overdue
interval tasks shift forward (we don't spam missed ticks); overdue
one-shots fire once immediately.

Task shape::

    {
      "id": "a1b2c3d4",
      "kind": "send_message",        # extensible — agent_prompt etc later
      "guild_id": 123, "channel_id": 456,
      "channel_name": "general",     # human label for lists/cancels
      "content": "Hey! sending after 5 sec",
      "interval_s": 5.0 | null,      # repeat every N seconds
      "run_at": 1730... (epoch float),
      "created_by": 789, "created_at": ...,
      "label": "spam general until stopped"
    }
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from src.action_engine.utils.logger import logger

STORE = Path("data/scheduled_tasks.json")


class Scheduler:
    """One instance per bot — owns the task store + the tick loop."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.tasks: dict[str, dict] = {}
        self._job: asyncio.Task | None = None
        self._stopping = False

    # ---------------- persistence ---------------- #
    def _load(self) -> None:
        try:
            self.tasks = {t["id"]: t for t in json.loads(STORE.read_text())}
        except Exception:  # noqa: BLE001
            self.tasks = {}

    def _save(self) -> None:
        try:
            STORE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STORE.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self.tasks.values()), indent=1))
            tmp.replace(STORE)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"scheduler save failed: {e}")

    # ---------------- lifecycle ---------------- #
    async def start(self) -> None:
        self._load()
        now = time.time()
        for t in self.tasks.values():
            if t.get("interval_s") and t.get("run_at", 0) < now:
                # overdue interval task → next tick from now (don't catch up)
                t["run_at"] = now + t["interval_s"]
        self._save()
        self._stopping = False
        self._job = asyncio.create_task(self._loop())
        if self.tasks:
            logger.info(f"Scheduler: resumed {len(self.tasks)} task(s).")

    async def stop(self) -> None:
        self._stopping = True
        if self._job:
            self._job.cancel()
            try:
                await self._job
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ---------------- api ---------------- #
    def add(self, task: dict) -> str:
        tid = uuid.uuid4().hex[:8]
        task.setdefault("id", tid)
        task.setdefault("created_at", time.time())
        task["run_at"] = task.get("run_at", 0) or time.time()
        self.tasks[tid] = task
        self._save()
        return tid

    def cancel(self, pred) -> list[dict]:
        """Cancel tasks matching pred(task) -> bool. Returns removed tasks."""
        killed = [t for t in self.tasks.values() if pred(t)]
        for t in killed:
            del self.tasks[t["id"]]
        if killed:
            self._save()
        return killed

    def list(self) -> list[dict]:
        return list(self.tasks.values())

    # ---------------- execution ---------------- #
    async def _loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(1.0)
            now = time.time()
            due = [
                t for t in self.tasks.values()
                if t.get("run_at", 9e18) <= now
            ]
            if not due:
                continue
            for t in due:
                asyncio.create_task(self._run(t))
                if t.get("interval_s"):
                    t["run_at"] = now + t["interval_s"]
                else:
                    del self.tasks[t["id"]]
            self._save()

    async def _run(self, t: dict) -> None:
        try:
            ch = self.bot.get_channel(t["channel_id"])
            if ch is None:
                ch = await self.bot.fetch_channel(t["channel_id"])
            if t["kind"] == "send_message":
                await ch.send(t["content"])
            logger.info(
                f"Scheduler fired {t['kind']} #{t['id']} -> #{t.get('channel_name')}")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"scheduled task {t.get('id')} failed: {e}")


scheduler: Scheduler | None = None


def init(bot) -> Scheduler:
    global scheduler
    scheduler = Scheduler(bot)
    return scheduler
