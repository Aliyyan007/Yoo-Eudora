"""
Bump scheduler — API-based slash command bumping via discord.py-self.
NO Playwright, NO browser. Triggers /bump directly through the Discord API.
Supports multiple bump bots with INDEPENDENT cooldown tracking per bot.

Each bump bot has its own cooldown:
  - Bumper:    1 hour
  - OneBump:   2 hours
  - Disboard:  2 hours
  - Bump4You:  2 hours
  - Carl Bot:  6 hours

The scheduler runs a single loop that checks every 60 seconds which bots
are ready to bump, and bumps only those whose cooldown has expired.
This maximizes bump frequency without wasting commands on cooling bots.
"""
import asyncio
import json
import os
import random
import time
from typing import List, Dict, Optional, Callable, Tuple
from loguru import logger
import discord

from .utils.delays import jittered_delay


# Default bump bots: (name, application_id, cooldown_hours)
DEFAULT_BUMP_BOTS = [
    ("Disboard",   302050872383242240,   2.0),   # 2h cooldown
    ("Bumper",     1153715777594200074,  1.0),   # 1h cooldown
    ("Carl Bot",   235148962103951360,   6.0),   # 6h cooldown
    ("OneBump",    1028956609382199346,  2.0),   # 2h cooldown
    ("Bump4You",   1089935069927456849,  2.0),   # 2h cooldown
]

# Safety margin: wait a bit extra after cooldown to avoid "still on cooldown" errors
_COOLDOWN_SAFETY_MARGIN_S = 60  # 1 minute extra

# How often to check if any bot is ready to bump
_CHECK_INTERVAL_S = 60  # check every 60 seconds


class BumpScheduler:
    """
    Schedules and executes /bump slash commands via the Discord API.
    Each bot is tracked independently with its own cooldown timer.
    """

    def __init__(
        self,
        client: discord.Client,
        channel_id: int,
        bump_bots: List = None,
        base_interval_hours: float = 3.0,  # kept for compat, not used for per-bot scheduling
        jitter_hours: float = 2.0,         # kept for compat
        post_bump_callback: Optional[Callable] = None,
    ):
        self.client = client
        self.channel_id = channel_id
        # Store bots with cooldowns: list of (name, app_id, cooldown_hours)
        self.bump_bots = bump_bots or DEFAULT_BUMP_BOTS
        self.post_bump_callback = post_bump_callback
        self._running = False
        self._cached_commands = None
        self._last_cache_refresh = 0
        self._cache_refresh_interval = 300  # Refresh command cache every 5 min

        # Per-bot last bump time tracking (bot_name -> timestamp)
        # PERSISTED to data/bump_state.json — persona rotation rebuilds this
        # scheduler under a new account, but bump cooldowns are server-wide
        # and must NOT reset when the active persona switches.
        self._last_bump_time: Dict[str, float] = self._load_state()
        # Per-bot next eligible bump time (bot_name -> timestamp)
        self._next_bump_time: Dict[str, float] = {}

        # Initialize: all bots eligible immediately (first run)
        now = time.time()
        for name, _app_id, _cooldown in self.bump_bots:
            self._last_bump_time.setdefault(name, 0)
            self._next_bump_time[name] = now  # ready immediately on first run

        # Track recent bumps for post-bump callback (only fire once per cycle)
        self._bumps_this_cycle = 0

    # ── persisted cooldown state (survives restarts + persona rotation) ──
    _STATE_FILE = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "bump_state.json")

    def _load_state(self) -> Dict[str, float]:
        try:
            with open(self._STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: float(v) for k, v in data.get("last_bump", {}).items()}
        except Exception:
            return {}

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._STATE_FILE), exist_ok=True)
            tmp = self._STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"last_bump": self._last_bump_time}, f)
            os.replace(tmp, self._STATE_FILE)
        except Exception as e:
            logger.debug(f"Bump state save failed: {e}")

    async def start(self):
        """Start the per-bot bump loop."""
        self._running = True
        bot_names = [b[0] for b in self.bump_bots]
        cooldowns = {b[0]: b[2] for b in self.bump_bots}
        logger.info(f"Starting bump scheduler with {len(self.bump_bots)} bots: {bot_names}")
        logger.info(f"Per-bot cooldowns: {cooldowns}")

        # Wait for the bot to be ready
        await self.client.wait_until_ready()

        # Initial delay before first bump — adaptive: if bots are already
        # due (restart, persona rotation, downtime) a full 2-min settle-in
        # per activation can outlive the whole rotation window and nothing
        # ever bumps. Short jitter when work is pending, full delay when not.
        due = len(self._get_ready_bots())
        initial_delay = random.uniform(20, 60) if due else 120
        logger.info(f"First bump check in {initial_delay:.0f}s ({due} bot(s) due)")
        await asyncio.sleep(initial_delay)

        while self._running and not self.client.is_closed():
            try:
                await self._check_and_bump()
            except Exception as e:
                logger.error(f"Bump check error: {e}")

            # Wait before next check
            await asyncio.sleep(_CHECK_INTERVAL_S)

    async def stop(self):
        """Stop the bump loop."""
        self._running = False
        logger.info("Bump scheduler stopped")

    def _bot_ready_time(self, bot_name: str, cooldown_hours: float) -> float:
        """Calculate when a bot is ready to bump again."""
        last = self._last_bump_time.get(bot_name, 0)
        cooldown_s = cooldown_hours * 3600 + _COOLDOWN_SAFETY_MARGIN_S
        return last + cooldown_s

    def _get_ready_bots(self) -> List[Tuple[str, int, float]]:
        """Return list of bots that are ready to bump (cooldown expired)."""
        now = time.time()
        ready = []
        for name, app_id, cooldown in self.bump_bots:
            ready_at = self._bot_ready_time(name, cooldown)
            if now >= ready_at:
                ready.append((name, app_id, cooldown))
        return ready

    def _get_next_bump_info(self) -> Optional[Tuple[str, float]]:
        """Return (bot_name, seconds_until_ready) for the soonest-ready bot."""
        now = time.time()
        soonest_name = None
        soonest_delta = float('inf')
        for name, _app_id, cooldown in self.bump_bots:
            ready_at = self._bot_ready_time(name, cooldown)
            delta = ready_at - now
            if delta < soonest_delta:
                soonest_delta = delta
                soonest_name = name
        if soonest_name:
            return (soonest_name, max(soonest_delta, 0))
        return None

    async def _check_and_bump(self):
        """Check which bots are ready and bump them. Runs every 60 seconds."""
        ready_bots = self._get_ready_bots()
        if not ready_bots:
            # Log next ready bot occasionally (every 10 min = every 10 checks)
            return

        channel = self.client.get_channel(self.channel_id)
        if not channel:
            logger.warning(f"Could not find bump channel {self.channel_id}")
            return

        logger.info(f"=== Bump check: {len(ready_bots)} bot(s) ready: {[b[0] for b in ready_bots]} ===")

        for i, (bot_name, bot_id, cooldown) in enumerate(ready_bots, 1):
            logger.info(f"Bumping with {bot_name} [{i}/{len(ready_bots)}] (cooldown: {cooldown}h)")
            success = await self._trigger_bump(channel, bot_name, bot_id)

            if success:
                self._last_bump_time[bot_name] = time.time()
                self._save_state()
                next_ready = self._bot_ready_time(bot_name, cooldown)
                next_in_h = (next_ready - time.time()) / 3600
                logger.info(f"{bot_name} bumped. Next eligible in {next_in_h:.1f}h")
            else:
                # On failure, set a short retry (5 min)
                self._last_bump_time[bot_name] = time.time() - (cooldown * 3600) + 300
                logger.warning(f"{bot_name} bump failed. Will retry in 5 min")

            # Human-like delay between bumps (5-15 seconds)
            if i < len(ready_bots):
                delay = random.randint(5, 15)
                logger.debug(f"Waiting {delay}s before next bump...")
                await asyncio.sleep(delay)

        # Trigger post-bump messaging callback
        if self.post_bump_callback and len(ready_bots) > 0:
            try:
                await self.post_bump_callback()
            except Exception as e:
                logger.warning(f"Post-bump callback error: {e}")

        # Log next bump info
        next_info = self._get_next_bump_info()
        if next_info:
            name, delta = next_info
            if delta > 0:
                logger.info(f"Next bump: {name} in {delta / 60:.1f} min")

    async def _get_bump_commands(self, channel) -> List:
        """Fetch and cache application commands from the channel."""
        now = time.time()
        if self._cached_commands and (now - self._last_cache_refresh) < self._cache_refresh_interval:
            return self._cached_commands

        try:
            self._cached_commands = await channel.application_commands()
            self._last_cache_refresh = now
            logger.debug(f"Fetched {len(self._cached_commands)} application commands from channel")
            return self._cached_commands
        except Exception as e:
            logger.warning(f"Failed to fetch application commands: {e}")
            return self._cached_commands or []

    async def _trigger_bump(self, channel, bot_name: str, bot_id: int) -> bool:
        """
        Find and trigger /bump for a specific bot via the API.
        Returns True if successful.
        """
        commands = await self._get_bump_commands(channel)
        if not commands:
            logger.warning(f"No application commands found in channel — cannot bump {bot_name}")
            return False

        # Find the bump command for this specific bot
        bump_cmd = None
        for cmd in commands:
            if cmd.name == "bump" and str(getattr(cmd, "application_id", "")) == str(bot_id):
                bump_cmd = cmd
                break

        if not bump_cmd:
            # Try by name only (some bots might have different command structures)
            for cmd in commands:
                if cmd.name == "bump":
                    app_id = getattr(cmd, "application_id", None)
                    if app_id and str(app_id) == str(bot_id):
                        bump_cmd = cmd
                        break
            if not bump_cmd:
                logger.warning(f"Could not find /bump command for {bot_name} (ID: {bot_id})")
                return False

        try:
            await bump_cmd(channel)
            logger.info(f"Bumped with {bot_name} via API")
            return True
        except discord.HTTPException as e:
            if e.status == 429:
                retry_after = getattr(e, "retry_after", 60)
                logger.warning(f"Rate limited while bumping {bot_name}. Waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                # Retry once
                try:
                    await bump_cmd(channel)
                    logger.info(f"Bumped with {bot_name} on retry")
                    return True
                except Exception as e2:
                    logger.error(f"Retry failed for {bot_name}: {e2}")
                    return False
            else:
                logger.error(f"HTTP error bumping {bot_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to bump with {bot_name}: {e}")
            return False

    async def _perform_bump_batch(self):
        """Force-bump all bots immediately (used for manual/command triggers).

        This bumps ALL bots regardless of cooldown status. Used when a user
        explicitly asks to bump the server.
        """
        channel = self.client.get_channel(self.channel_id)
        if not channel:
            logger.error(f"Could not find bump channel {self.channel_id}")
            return

        logger.info(f"=== Manual bump batch in #{channel.name} ===")
        results: Dict[str, bool] = {}

        for i, (bot_name, bot_id, _cooldown) in enumerate(self.bump_bots, 1):
            logger.info(f"Bumping with {bot_name} [{i}/{len(self.bump_bots)}]")
            success = await self._trigger_bump(channel, bot_name, bot_id)
            results[bot_name] = success

            if success:
                self._last_bump_time[bot_name] = time.time()
                self._save_state()

            # Human-like delay between bumps (5-15 seconds)
            if i < len(self.bump_bots):
                delay = random.randint(5, 15)
                logger.debug(f"Waiting {delay}s before next bump...")
                await asyncio.sleep(delay)

        # Summary
        successful = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        logger.info(f"Manual bump batch complete: {successful}/{len(self.bump_bots)} successful, {failed} failed")

        # Trigger post-bump messaging callback
        if self.post_bump_callback and successful > 0:
            try:
                await self.post_bump_callback()
            except Exception as e:
                logger.warning(f"Post-bump callback error: {e}")

    def get_status(self) -> Dict[str, dict]:
        """Return status of all bump bots (for debugging/monitoring)."""
        now = time.time()
        status = {}
        for name, _app_id, cooldown in self.bump_bots:
            last = self._last_bump_time.get(name, 0)
            ready_at = self._bot_ready_time(name, cooldown)
            status[name] = {
                "cooldown_hours": cooldown,
                "last_bumped": last,
                "ready_at": ready_at,
                "ready_now": now >= ready_at,
                "seconds_until_ready": max(0, ready_at - now),
            }
        return status
