"""Virtual clock for the simulation.

Patches ``time.time``/``time.sleep``/``asyncio.sleep`` so the bot's
long sleeps (typing delays, loop intervals, cooldowns) are controlled
by the test instead of real wall-clock time.

Semantics
---------
* ``clock.sleep(d)`` parks the task until the clock reaches now+d —
  advanced by ``await clock.advance(seconds)`` or ``set_time``.
* ``await clock.advance(s)`` moves virtual time forward in order,
  waking sleepers at their deadlines and yielding between each so
  woken tasks run before time continues.
* ``time.sleep(d)`` (sync, e.g. llm key-cooldown waits) advances the
  clock instantly without yielding — a sync wait blocks the thread
  anyway, so fast-forwarding is the honest equivalent.
* Untracked yields (``asyncio.sleep(0)``) still pass through to the
  real asyncio.sleep(0) so the event loop schedules normally.

Patch ``time.monotonic`` is NOT done — asyncio's loop timing relies on
it and discord internals use it for heartbeats.
"""
from __future__ import annotations

import asyncio
import heapq
import time as _real_time


class VirtualClock:
    def __init__(self, start: float = None):
        self._now = start if start is not None else _real_time.time()
        # heap of (deadline, seq, future, task)
        self._waiters = []
        self._seq = 0
        # Tasks woken by advance() but not yet re-parked — virtual time
        # must not outrun them or their next sleep() lands past the window.
        self._inflight = set()

    # ── time sources ────────────────────────────────────────────────
    def time(self) -> float:
        return self._now

    def sleep_blocking(self, delay: float) -> None:
        """Replacement for time.sleep — jump the clock forward."""
        if delay and delay > 0:
            self._now += delay

    # ── asyncio.sleep replacement ───────────────────────────────────
    async def sleep(self, delay: float):
        if delay is None or delay <= 0:
            await _real_sleep(0)
            return
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._seq += 1
        self._inflight.discard(task)
        heapq.heappush(self._waiters, (self._now + delay, self._seq, fut, task))
        try:
            await fut
        except asyncio.CancelledError:
            # Drop the waiter so advance() doesn't wake a dead task
            try:
                self._waiters.remove((self._now + delay, self._seq, fut, task))
                heapq.heapify(self._waiters)
            except ValueError:
                pass
            raise

    # ── test-side controls ──────────────────────────────────────────
    async def advance(self, seconds: float):
        """Move time forward through the next `seconds` virtual seconds,
        waking every sleeper whose deadline lands inside the window —
        including sleepers that park *during* the advance (chained
        typing/read delays). Terminates when no waiter <= end has
        appeared for a short settle period.

        The settle uses *real* short sleeps, not bare yields: production
        code hops through ``run_in_executor`` (real threads), so a task
        can be runnable-but-not-parked for real milliseconds. Bare
        ``sleep(0)`` loops would finish the advance before the thread
        hands control back, stranding the task's next virtual sleep.
        """
        end = self._now + seconds
        last_fire = _real_time_time()
        while True:
            # Don't outrun a woken-but-unre-parked task: after firing its
            # waiter, give it real time to reach its next clock.sleep()
            # (executor hops are real-ms). Otherwise perpetual 1s sleepers
            # (schedulers) burn the whole window while the task is mid-hop
            # and its next sleep parks past `end` — stranded forever.
            if self._inflight:
                self._inflight = {t for t in self._inflight if not t.done()}
                if self._inflight:
                    if _real_time_time() - last_fire > 0.5:
                        self._inflight.clear()
                    else:
                        await _real_sleep(0.002)
                        continue
            if self._waiters and self._waiters[0][0] <= end:
                deadline, _, fut, task = heapq.heappop(self._waiters)
                self._now = max(self._now, deadline)
                if not fut.done():
                    fut.set_result(None)
                    if task is not None:
                        self._inflight.add(task)
                last_fire = _real_time_time()
                await _real_sleep(0.001)
                continue
            # No due waiter — production code hops through run_in_executor
            # (real threads) between virtual sleeps, so a task can be
            # runnable-but-not-parked for real tens of ms. Give stragglers
            # ~0.5s real quiescence before declaring the window done; a
            # waiter parked past `end` correctly lets the grace expire.
            if _real_time_time() - last_fire > 0.5:
                break
            await _real_sleep(0.002)
        self._now = end
        self._inflight.clear()
        for _ in range(5):
            await _real_sleep(0.002)

    async def flush(self, rounds: int = 10):
        """Yield several times so pending callbacks/tasks settle."""
        for _ in range(rounds):
            await _real_sleep(0)

    def pending(self) -> int:
        return len(self._waiters)


# Bound at install() time so replacements can delegate to the real thing.
_real_sleep = asyncio.sleep
_real_time_time = _real_time.time
_real_time_sleep = _real_time.sleep


class ClockPatch:
    """Context manager / pytest helper: patch time+sleep onto a clock."""

    def __init__(self, clock: VirtualClock):
        self.clock = clock
        self._orig = {}

    def install(self):
        import time as time_mod
        self._orig["time.time"] = time_mod.time
        self._orig["time.sleep"] = time_mod.sleep
        self._orig["asyncio.sleep"] = asyncio.sleep
        time_mod.time = self.clock.time
        time_mod.sleep = self.clock.sleep_blocking
        asyncio.sleep = self.clock.sleep
        return self

    def uninstall(self):
        import time as time_mod
        time_mod.time = self._orig["time.time"]
        time_mod.sleep = self._orig["time.sleep"]
        asyncio.sleep = self._orig["asyncio.sleep"]
