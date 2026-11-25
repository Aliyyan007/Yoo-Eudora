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
from src.ai import metrics

# Anchor persistence to the repository, not the process cwd — Render/Docker
# and local invocations do not always start from the same directory.
STORE = Path(__file__).resolve().parents[3] / "data" / "scheduled_tasks.json"
MAX_ONESHOT_FAILURES = 5


class Scheduler:
    """One instance per bot — owns the task store + the tick loop."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.tasks: dict[str, dict] = {}
        self._job: asyncio.Task | None = None
        self._stopping = False
        # In-flight _run() tasks — tracked so stop()/rotation can cancel
        # them instead of letting sends fire on a dead client.
        self._run_tasks: set[asyncio.Task] = set()

    # ---------------- persistence ---------------- #
    @staticmethod
    def _valid_task(t) -> bool:
        """Minimal shape validation for persisted records."""
        return (
            isinstance(t, dict)
            and isinstance(t.get("id"), str) and t["id"]
            and isinstance(t.get("kind"), str) and t["kind"]
            and isinstance(t.get("channel_id"), int)
            and isinstance(t.get("run_at"), (int, float))
        )

    def _load(self) -> None:
        try:
            raw = json.loads(STORE.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("scheduler store root must be a list")
            valid = [t for t in raw if self._valid_task(t)]
            bad = len(raw) - len(valid)
            self.tasks = {t["id"]: t for t in valid}
            if bad:
                quarantine = STORE.with_suffix(".corrupt.json")
                quarantine.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                logger.warning(f"Scheduler: quarantined {bad} malformed task record(s)")
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
        # READY refires / rebinding must not create duplicate tick loops.
        if self._job is not None and not self._job.done():
            return
        self._load()
        now = time.time()
        for t in self.tasks.values():
            t.pop("_running", None)  # stale marker from a crash/rotation
            if t.get("interval_s") and t.get("run_at", 0) < now:
                # overdue interval task → next tick from now (don't catch up)
                t["run_at"] = now + t["interval_s"]
        self._save()
        self._stopping = False
        self._job = asyncio.create_task(self._loop())
        metrics.incr("scheduler.started")
        metrics.gauge("scheduler.tasks", len(self.tasks))
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
            self._job = None
        # Kill in-flight task runs — they hold a reference to the bot
        # client and would keep sending on a dead connection otherwise.
        pending = [t for t in self._run_tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._run_tasks.clear()

    # ---------------- api ---------------- #
    def add(self, task: dict) -> str:
        tid = uuid.uuid4().hex[:8]
        task.setdefault("id", tid)
        task.setdefault("created_at", time.time())
        task["run_at"] = task.get("run_at", 0) or time.time()
        self.tasks[tid] = task
        self._save()
        metrics.incr("scheduler.task_added", kind=task.get("kind", "unknown"))
        metrics.gauge("scheduler.tasks", len(self.tasks))
        return tid

    def cancel(self, pred) -> list[dict]:
        """Cancel tasks matching pred(task) -> bool. Returns removed tasks."""
        killed = [t for t in self.tasks.values() if pred(t)]
        for t in killed:
            del self.tasks[t["id"]]
        if killed:
            self._save()
            metrics.incr("scheduler.task_cancelled", amount=len(killed))
            metrics.gauge("scheduler.tasks", len(self.tasks))
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
                if t.get("_running") or t.get("dead"):
                    continue
                t["_running"] = True
                if t.get("interval_s"):
                    # Interval work is rescheduled immediately even if this
                    # tick fails; the interval is a cadence, not a retry queue.
                    t["run_at"] = now + t["interval_s"]
                rt = asyncio.create_task(self._run_and_finalize(t))
                self._run_tasks.add(rt)
                rt.add_done_callback(self._run_tasks.discard)
            self._save()

    async def _run_and_finalize(self, t: dict) -> None:
        """Execute one due task, then durably record its outcome.

        One-shots used to be deleted before the send actually succeeded, so a
        rotation/429/client failure permanently lost the reminder. Now they
        remain pending with bounded exponential retry; only a confirmed send
        removes them.
        """
        success = False
        cancelled = False
        try:
            success = await self._run(t)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if cancelled:
                # Rotation/cancellation must not consume or retry the task —
                # the next activation may rebind and execute it safely. Do not
                # return here: cancellation must still propagate.
                t.pop("_running", None)
                self._save()
            else:
                t.pop("_running", None)
                metrics.incr("scheduler.run_ok" if success else "scheduler.run_failed",
                             kind=t.get("kind", "unknown"))
                if t.get("interval_s"):
                    if not success:
                        t["failure_count"] = int(t.get("failure_count") or 0) + 1
                    else:
                        t.pop("failure_count", None)
                elif success:
                    self.tasks.pop(t["id"], None)
                else:
                    failures = int(t.get("failure_count") or 0) + 1
                    t["failure_count"] = failures
                    if failures >= MAX_ONESHOT_FAILURES:
                        # Keep the record for operator inspection but exclude it
                        # from future due scans instead of silently deleting it.
                        t["dead"] = True
                        t["dead_reason"] = "max retries exceeded"
                        metrics.incr("scheduler.task_dead", kind=t.get("kind", "unknown"))
                    else:
                        t["run_at"] = time.time() + min(300.0, 30.0 * (2 ** (failures - 1)))
                self._save()
                metrics.gauge("scheduler.tasks", len(self.tasks))

    async def _run(self, t: dict) -> bool:
        try:
            ch = self.bot.get_channel(t["channel_id"])
            if ch is None:
                ch = await self.bot.fetch_channel(t["channel_id"])
            if ch is None:
                raise RuntimeError("channel not found")
            if t["kind"] == "send_message":
                await ch.send(t["content"])
            else:
                raise RuntimeError(f"unknown scheduled task kind {t['kind']!r}")
            logger.info(
                f"Scheduler fired {t['kind']} #{t['id']} -> #{t.get('channel_name')}")
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"scheduled task {t.get('id')} failed: {e}")
            return False


scheduler: Scheduler | None = None


def init(bot) -> Scheduler:
    global scheduler
    scheduler = Scheduler(bot)
    return scheduler
