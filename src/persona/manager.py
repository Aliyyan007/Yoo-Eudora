"""Persona rotation supervisor — one account online at a time, rotating
through the configured personas every ~2-3 hours.

Design
------
* Only one discord.Client is ever connected: the supervisor builds a fresh
  AIPersonaClient for the incoming persona, runs it, then closes it cleanly
  before the next one starts. Fresh client = private in-memory state
  (conversation tracker, sticky windows, reply history, mood) for free;
  module-level singletons (ping cooldowns, engagement walls, bump state)
  are process-wide so server limits survive rotation.
* Rotation window: PERSONA_ROTATE_MIN (default 150) ± PERSONA_ROTATE_JITTER_MIN
  (default 30) → 2-3h human-feeling cadence.
* Restart recovery: data/persona_state.json records who's active + when they
  started. On boot the supervisor resumes the same persona's remaining window
  instead of restarting the clock.
* Failure tolerance: a persona that fails to connect retries with backoff
  (3 tries) before the supervisor moves to the next account. If every account
  fails the loop sleeps and retries — the process never dies on auth/network
  errors.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Awaitable, Callable, Dict, List, Optional

from loguru import logger

from . import runtime
from .profiles import PROFILES, PersonaProfile, get_profile

ROTATE_MIN = int(os.getenv("PERSONA_ROTATE_MIN", "150"))           # 2.5h base
ROTATE_JITTER_MIN = int(os.getenv("PERSONA_ROTATE_JITTER_MIN", "30"))
_RESUME_MIN_REMAINING_S = 300   # <5min left in the window → just rotate
_CONNECT_RETRY_LIMIT = 3
_ALL_FAIL_SLEEP_S = 120         # every account failed — breathe, retry


def _window_seconds() -> float:
    return (ROTATE_MIN + random.uniform(0, ROTATE_JITTER_MIN)) * 60.0


class PersonaSupervisor:
    """Owns the account lifecycle. `client_factory` builds+starts a wired
    client for a persona and returns (client, run_coro)."""

    def __init__(self, client_factory: Callable[[PersonaProfile], Awaitable]):
        """client_factory(profile) -> (client, run_coroutine) where run_coroutine
        is awaitable client.start(token)."""
        self._factory = client_factory
        self._accounts: List[PersonaProfile] = self._load_accounts()
        self._idx = 0
        self._client = None
        self._run_task: Optional[asyncio.Task] = None

    # ── Accounts ────────────────────────────────────────────────────────
    def _load_accounts(self) -> List[PersonaProfile]:
        """Personas with a token configured, in registry order."""
        out = []
        for pid, p in PROFILES.items():
            if (os.getenv(p.token_env) or "").strip():
                out.append(p)
        if not out:
            # Back-compat: single-token installs still run (eudora only)
            logger.warning("[persona] no persona tokens resolved — check env")
        else:
            logger.info(
                "[persona] accounts ready: "
                + ", ".join(f"{p.full_name}({p.id})" for p in out))
        return out

    def _next(self, seq: int) -> PersonaProfile:
        return self._accounts[seq % len(self._accounts)]

    # ── Lifecycle ───────────────────────────────────────────────────────
    async def _start_account(self, profile: PersonaProfile) -> asyncio.Task:
        """Activate the persona, build its client, and start connecting."""
        runtime.activate(profile)
        client, run_coro = await self._factory(profile)
        self._client = client
        self._run_task = asyncio.create_task(run_coro, name=f"persona:{profile.id}")
        return self._run_task

    async def _stop_account(self) -> None:
        client, task = self._client, self._run_task
        self._client = self._run_task = None
        if client is not None:
            try:
                await client.teardown()     # cancel loops, leave voice cleanly
            except Exception as e:
                logger.debug(f"[persona] teardown error (continuing): {e}")
            try:
                await asyncio.wait_for(client.close(), timeout=15)
            except Exception as e:
                logger.debug(f"[persona] close error (continuing): {e}")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # The client task's own cancellation is expected here — but
                # if the SUPERVISOR task itself was cancelled, swallowing it
                # would loop forever. Re-raise when we were the target.
                cur = asyncio.current_task()
                if cur is not None and cur.cancelling() > 0:
                    raise
            except Exception:
                pass

    async def run(self) -> None:
        """Supervisor loop — never returns unless cancelled."""
        if not self._accounts:
            raise RuntimeError("No persona accounts configured")

        # ── Restart recovery — resume the in-progress window ────────────
        state = runtime.load_rotation_state()
        seq = int(state.get("seq", 0))
        remaining = 0.0
        saved_id = state.get("active")
        if saved_id and saved_id in {p.id for p in self._accounts}:
            elapsed = time.time() - float(state.get("activated_at", 0))
            remaining = _window_seconds() - elapsed
            if remaining < _RESUME_MIN_REMAINING_S:
                remaining = 0.0     # window basically over — advance
                seq += 1
        if remaining <= 0:
            runtime.clear_rotation_state()

        consecutive_fails = 0
        while True:
            # Every account failing in a row → total outage (network/auth) —
            # breathe instead of hot-looping through connect attempts.
            if consecutive_fails >= len(self._accounts):
                logger.warning(
                    f"[persona] all {len(self._accounts)} accounts failing — "
                    f"retrying in {_ALL_FAIL_SLEEP_S}s")
                await asyncio.sleep(_ALL_FAIL_SLEEP_S)
                consecutive_fails = 0

            profile = self._next(seq)
            window = remaining if remaining > 0 else _window_seconds()
            remaining = 0.0
            runtime.save_rotation_state(profile.id, time.time(), seq)
            logger.info(
                f"[persona] rotating → {profile.full_name} "
                f"(window ~{window/60:.0f}min, seq {seq})")

            # ── Connect with bounded retries ────────────────────────────
            task = None
            for attempt in range(1, _CONNECT_RETRY_LIMIT + 1):
                try:
                    task = await self._start_account(profile)
                    break
                except Exception as e:
                    logger.warning(
                        f"[persona] {profile.id} connect attempt {attempt} "
                        f"failed: {e}")
                    await self._stop_account()
                    if attempt < _CONNECT_RETRY_LIMIT:
                        await asyncio.sleep(min(30 * attempt, 120))
            if task is None:
                logger.warning(f"[persona] {profile.id} unreachable — skipping")
                seq += 1
                consecutive_fails += 1
                continue
            consecutive_fails = 0

            # ── Hold the window; bail early if the connection dies ──────
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=window)
                # Task finished on its own (client closed itself) → rotate
                logger.info(f"[persona] {profile.id} connection ended — rotating")
            except asyncio.TimeoutError:
                pass                        # normal rotation point
            except asyncio.CancelledError:
                # Inner client task cancelled vs. supervisor itself being
                # cancelled — only swallow the former.
                if not task.cancelled():
                    raise
            except Exception as e:
                logger.warning(f"[persona] {profile.id} crashed: {e!r} — rotating")

            await self._stop_account()
            seq += 1
