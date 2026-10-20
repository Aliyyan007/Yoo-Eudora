"""
Auto Bump System — Enhanced AI Persona Edition
Entry point: loads config, initializes all modules, and starts the bot.

Features:
- API-based bumping (no browser needed)
- AI-powered human-like chat with mood engine
- User memory and channel style learning
- Smart channel discovery and selection
- Proactive messaging and dead chat revival
- Multi-key Groq rotation for 99.9% uptime
- Web search for factual questions
- Self-reflection for continuous improvement
"""
import os
import sys
import asyncio
from typing import Set, Optional
from loguru import logger
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env"))

# Load YAML config
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    YAML_CONFIG = yaml.safe_load(_f) or {}

from .utils.logger import setup_logger
from .keep_alive import start_keep_alive_server
from .discord_client import AIPersonaClient
from .bump_scheduler import BumpScheduler, DEFAULT_BUMP_BOTS
from .proactive_messaging import ProactiveMessenger
from .persona import load_persona
from .persona.manager import PersonaSupervisor
from .persona.profiles import PROFILES, get_profile
from .ai import llm


def load_env_and_channels():
    """Load all configuration from environment variables."""
    # Discord token — either the legacy single token or ANY persona token
    # (isla/rowan/eudora). A persona-only install must not hard-fail on
    # DISCORD_TOKEN when its accounts are configured.
    token = os.getenv("DISCORD_TOKEN")
    persona_tokens = [
        os.getenv(p.token_env) for p in PROFILES.values()
        if (os.getenv(p.token_env) or "").strip()
    ]
    if not (token or persona_tokens):
        logger.error("No account token set — set DISCORD_TOKEN or a persona token (DISCORD_TOKEN_EUDORA/ISLA/ROWAN) in .env")
        sys.exit(1)

    # Bump channel — only required when bumping is wanted; without it the
    # persona client still runs (scheduler just stays disabled).
    bump_channel_id = os.getenv("BUMP_CHANNEL_ID")
    if bump_channel_id:
        bump_channel_id = int(bump_channel_id)
    else:
        bump_channel_id = 0
        logger.warning("BUMP_CHANNEL_ID not set — bump scheduler disabled")

    # Text channels (comma-separated)
    text_channel_str = os.getenv("TEXT_CHANNEL_IDS", "")
    text_channel_ids: Set[int] = set()
    if text_channel_str.strip():
        for ch_id in text_channel_str.split(","):
            ch_id = ch_id.strip()
            if ch_id:
                try:
                    text_channel_ids.add(int(ch_id))
                except ValueError:
                    logger.warning(f"Invalid channel ID: {ch_id}")

    logger.info(f"Text channels configured ({len(text_channel_ids)}): {text_channel_ids}")
    logger.info(f"Bump channel configured: {bump_channel_id}")

    # Chat revive ping role
    ping_role_id = os.getenv("CHAT_REVIVE_PING_ROLE")
    if ping_role_id:
        ping_role_id = int(ping_role_id)
        logger.info(f"Chat revive ping role configured: {ping_role_id}")

    # Bump settings
    base_interval = float(os.getenv("BUMP_BASE_INTERVAL_HOURS", "3"))
    jitter_hours = float(os.getenv("BUMP_JITTER_HOURS", "2"))

    return {
        "token": token,
        "bump_channel_id": bump_channel_id,
        "text_channel_ids": text_channel_ids,
        "ping_role_id": ping_role_id,
        "base_interval": base_interval,
        "jitter_hours": jitter_hours,
    }


async def main():
    """Main entry point — initialize and run the bot."""
    # Setup logging
    setup_logger(level="INFO")

    logger.info("=" * 60)
    logger.info("  Auto Bump System — Enhanced AI Persona Edition")
    logger.info("=" * 60)

    # Load config
    config = load_env_and_channels()

    # Check Groq keys
    key_count = llm.get_key_count()
    if key_count == 0:
        logger.error("No Groq API keys configured! Set GROQ_API_KEY in .env")
        sys.exit(1)
    logger.info(f"Groq client initialized with {key_count} key(s)")

    # Load persona from YAML config
    persona = load_persona(YAML_CONFIG)

    # ── Client factory: builds + fully wires one persona's client ──────────
    async def build_client(profile):
        """Create a wired AIPersonaClient for `profile` and return
        (client, client.start(token) coroutine). Every subsystem that holds
        a client reference is rebuilt per activation so it can't straddle
        two accounts; server-wide state (ping cooldowns, bump timers,
        engagement walls) lives in process-level stores and persists."""
        token = (os.getenv(profile.token_env) or "").strip()
        if not token:
            raise RuntimeError(f"{profile.token_env} not set for {profile.id}")

        # Optional egress proxy — Discord flags datacenter IPs (Render/VPS)
        # for user-account logins, so DISCORD_PROXY routes all Discord
        # traffic (REST + gateway) through a clean IP when set.
        proxy = (os.getenv("DISCORD_PROXY") or "").strip() or None
        proxy_auth = None
        if proxy:
            raw_auth = (os.getenv("DISCORD_PROXY_AUTH") or "").strip()
            if ":" in raw_auth:
                import aiohttp
                _pu, _pp = raw_auth.split(":", 1)
                proxy_auth = aiohttp.BasicAuth(_pu, _pp)
            logger.info(f"[persona] routing {profile.id} via proxy {proxy.split('@')[-1]}")

        client = AIPersonaClient(persona=profile, proxy=proxy, proxy_auth=proxy_auth)

        proactive = ProactiveMessenger(
            client=client,
            channel_ids=config["text_channel_ids"],
            ping_role_id=config["ping_role_id"],
            dead_chat_threshold_min=int(os.getenv("DEAD_CHAT_THRESHOLD_MIN", "20")),
            check_interval_min=int(os.getenv("PROACTIVE_CHECK_MIN", "5")),
            proactive_interval_min=(
                int(os.getenv("PROACTIVE_MIN_MIN", "40")),
                int(os.getenv("PROACTIVE_MAX_MIN", "80")),
            ),
            auto_chat_chance=float(os.getenv("AUTO_CHAT_CHANCE", "0.10")),
        )
        client.proactive_messenger = proactive

        async def post_bump_callback():
            await proactive.send_post_bump_messages()

        if config["bump_channel_id"]:
            bump_scheduler = BumpScheduler(
                client=client,
                channel_id=config["bump_channel_id"],
                base_interval_hours=config["base_interval"],
                jitter_hours=config["jitter_hours"],
                post_bump_callback=post_bump_callback,
            )
            client.bump_scheduler = bump_scheduler
        # NOTE: do NOT spawn scheduler.start() here — it awaits
        # client.wait_until_ready() which raises RuntimeError on a client
        # that hasn't logged in yet. The task died ~0.2s after every
        # activation (this was the "no bumps on Render" bug). on_ready
        # spawns it once the client is initialized.

        return client, client.start(token)

    # Boot-time reachability probe: answers "is this host's IP being
    # blocked?" instead of a silent login hang. TCP-443 to discord.com /
    # gateway.discord.gg + GETs of /api/v9/gateway and /login — failures
    # here mean Discord (or Cloudflare in front of it) is dropping this
    # egress IP and DISCORD_PROXY is the fix. Non-fatal, ~30s worst case.
    async def _preflight_probe():
        import socket
        import aiohttp as _aio
        loop = asyncio.get_running_loop()

        def _tcp(host):
            try:
                s = socket.create_connection((host, 443), timeout=8)
                s.close()
                return "OK"
            except Exception as e:
                return f"FAIL ({type(e).__name__}: {e})"

        for host in ("discord.com", "gateway.discord.gg"):
            r = await loop.run_in_executor(None, _tcp, host)
            logger.info(f"[preflight] tcp443 {host}: {r}")
        try:
            to = _aio.ClientTimeout(total=12)
            async with _aio.ClientSession(timeout=to) as sess:
                for url in ("https://discord.com/api/v9/gateway",
                            "https://discord.com/login"):
                    try:
                        async with sess.get(url) as resp:
                            logger.info(f"[preflight] GET {url}: HTTP {resp.status}")
                    except Exception as e:
                        logger.info(f"[preflight] GET {url}: FAIL ({type(e).__name__}: {e})")
        except Exception as e:
            logger.debug(f"[preflight] http probe skipped: {e}")

    try:
        await _preflight_probe()
    except Exception as e:
        logger.debug(f"[preflight] probe error (non-fatal): {e}")

    supervisor = PersonaSupervisor(build_client)
    logger.info(
        f"Connecting to Discord — {len(supervisor._accounts)} persona "
        "account(s) configured, rotation active"
    )
    await supervisor.run()


def run():
    """Run the bot."""
    start_keep_alive_server()
    if sys.platform != "win32":
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass  # uvloop is Linux-only (Render); absent locally/on Windows
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
