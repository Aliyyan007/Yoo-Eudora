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
# A connect task that dies this fast was an auth/transport failure, not a
# normal session end — counts toward the all-accounts-failing backoff.
_FAST_CRASH_S = 120
# Bounded READY wait: client.start() can sit inside a dropped/stalled
# gateway handshake for the whole rotation window with ZERO log output —
# the "service running, bots never online" failure. Fail it fast instead.
_CONNECT_WATCHDOG_S = int(os.getenv("CONNECT_WATCHDOG_S", "120"))


def _fail_hint(exc: BaseException) -> str:
    """Attach an actionable hint to a connect/crash error so a Discord-side
    outage is diagnosable from the log alone."""
    kind = type(exc).__name__
    text = str(exc)
    if "LoginFailure" in kind or "Improper token" in text or "401" in text:
        return " — token rejected (invalid/revoked); re-check the persona token env var"
    if "captcha" in text.lower() or "cloudflare" in text.lower() or "403" in text:
        return (" — Discord is challenging this host's IP (datacenter IPs get"
                " flagged for user tokens); set DISCORD_PROXY to a clean proxy")
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timed out" in text.lower():
        return (" — connect timed out; if this repeats on every account the egress"
                " IP is likely being dropped — try DISCORD_PROXY")
    return ""


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

    async def _wait_ready(self, timeout: float) -> None:
        """Watch the connect: return once the client's READY event fires.
        Raise the run-task's real error if it dies first, or TimeoutError
        if neither happens — a hung gateway handshake is a failure, not a
        reason to hold the window."""
        client, task = self._client, self._run_task
        if client is None or task is None:
            raise RuntimeError("connect produced no client")
        # Poll is_ready() rather than await client.wait_until_ready():
        # the _ready event only exists after login() has begun, so the
        # awaited version races the connect task and can raise
        # RuntimeError("not properly initialised") as a false failure.
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if client.is_ready():
                return
            if task.done():
                exc = task.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError("client closed before READY")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"no READY after {timeout:.0f}s")
            await asyncio.sleep(0.5)

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

            # ── Connect with bounded retries + READY watchdog ──────────
            task = None
            for attempt in range(1, _CONNECT_RETRY_LIMIT + 1):
                try:
                    await self._start_account(profile)
                    # client.start() can hang silently inside the gateway
                    # handshake on a flagged/dropped IP — wait for READY
                    # or the task's death, whichever comes first.
                    await self._wait_ready(_CONNECT_WATCHDOG_S)
                    task = self._run_task
                    break
                except Exception as e:
                    logger.warning(
                        f"[persona] {profile.id} connect attempt {attempt} "
                        f"failed: {type(e).__name__}: {e}{_fail_hint(e)}")
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
            started_at = time.time()
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
                fast = (time.time() - started_at) < _FAST_CRASH_S
                logger.warning(
                    f"[persona] {profile.id} crashed after "
                    f"{time.time() - started_at:.0f}s: {e!r} — rotating"
                    f"{_fail_hint(e)}")
                # A fast crash = connect/auth failure (login rejections land
                # here, not in the retry loop above). Count it so every
                # account dying → all-fail backoff + loud hint, instead of
                # hot-rotating forever with the process looking alive.
                if fast:
                    consecutive_fails += 1
                    if consecutive_fails >= len(self._accounts):
                        logger.error(
                            "[persona] EVERY account failed to connect. If the"
                            " tokens are valid, Discord is likely blocking this"
                            " host's IP — set DISCORD_PROXY (and"
                            " DISCORD_PROXY_AUTH=user:pass if needed) to route"
                            " traffic through a clean IP.")
                else:
                    consecutive_fails = 0

            await self._stop_account()
            seq += 1
