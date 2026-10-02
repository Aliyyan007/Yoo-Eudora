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
    # Discord token
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("DISCORD_TOKEN not set in .env")
        sys.exit(1)

    # Bump channel
    bump_channel_id = os.getenv("BUMP_CHANNEL_ID")
    if not bump_channel_id:
        logger.error("BUMP_CHANNEL_ID not set in .env")
        sys.exit(1)
    bump_channel_id = int(bump_channel_id)

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

    # Server invite
    server_invite = os.getenv("SERVER_INVITE", "")

    # Bump settings
    base_interval = float(os.getenv("BUMP_BASE_INTERVAL_HOURS", "3"))
    jitter_hours = float(os.getenv("BUMP_JITTER_HOURS", "2"))

    return {
        "token": token,
        "bump_channel_id": bump_channel_id,
        "text_channel_ids": text_channel_ids,
        "ping_role_id": ping_role_id,
        "server_invite": server_invite,
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

        client = AIPersonaClient(persona=profile)

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

        bump_scheduler = BumpScheduler(
            client=client,
            channel_id=config["bump_channel_id"],
            base_interval_hours=config["base_interval"],
            jitter_hours=config["jitter_hours"],
            post_bump_callback=post_bump_callback,
        )
        client.bump_scheduler = bump_scheduler
        client._spawn(bump_scheduler.start())

        # Auto-join server if invite is configured (each account joins once)
        if config["server_invite"]:
            async def join_server():
                await client.wait_until_ready()
                invite_code = config["server_invite"].split("/")[-1]
                try:
                    try:
                        invite = await client.fetch_invite(invite_code)
                        gid = getattr(getattr(invite, "guild", None), "id", None)
                        if gid and any(g.id == gid for g in client.guilds):
                            return
                    except Exception:
                        pass
                    logger.info(f"[{profile.id}] Attempting to join server: {invite_code}")
                    await client.accept_invite(invite_code)
                    logger.info(f"[{profile.id}] Joined server: {invite_code}")
                except Exception as e:
                    logger.warning(f"[{profile.id}] Could not join server (may already be a member): {e}")
            client._spawn(join_server())

        return client, client.start(token)

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
