"""
AI-driven new member greeting system for the Eudora persona.

This module is intentionally NOT standard template-based welcome code.
The greeting message, the channel choice, the tone, and even the
*decision to greet at all* are all delegated to the LLM. The code
around it only:

- gathers server / channel context algorithmically,
- enforces cooldowns so we don't spam,
- wraps the blocking LLM call in ``asyncio.to_thread``,
- sends the greeting with human-like delays so it doesn't feel bot-like.

Flow (``handle_new_member``):
1. Check per-user and per-guild cooldowns.
2. Gather server context (active channels, member count, server name).
3. Call the LLM to decide: should we greet? which channel? what message?
4. If the LLM says yes, send the greeting with a human-like delay.
5. Record the greeting so cooldowns apply next time.
"""
import asyncio
import json
import os
import random
import re
import time
from typing import Dict, List, Optional

import discord
from loguru import logger

from .llm import call_fast
from . import prompts
from .channel_nature import get_cached_nature


# ── Channel name heuristics ───────────────────────────────────────────────────

# Names that strongly suggest a welcoming / main chat channel.
_WELCOME_NAMES = ("general", "chat", "main", "lobby", "welcome")
# Names that suggest a secondary chat channel.
_SECONDARY_NAMES = ("talk", "off-topic", "lounge", "offtopic", "chatter")
# Names that suggest we should NOT greet here.
_AVOID_NAMES = ("rules", "announcements", "bot", "commands", "logs",
                "mod", "audit", "staff", "admin", "welcome-rules")


class WelcomeSystem:
    """
    AI-driven welcome system.

    The greeting itself, the channel choice, the tone, and the decision
    of whether to greet at all are all produced by the LLM. This class
    handles the surrounding logistics: cooldown enforcement, context
    gathering, human-like delays, and graceful error handling.
    """

    def __init__(self):
        # user_id -> timestamp of last greeting (any guild)
        self._greeted_users: Dict[int, float] = {}
        # Don't greet the same user again within 1 hour
        self._cooldown_s = 3600
        # 2 min minimum between greetings in the same server
        self._greet_cooldown_s = 120
        # guild_id -> last greet timestamp
        self._last_greet_time: Dict[int, float] = {}

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle_new_member(self, member: discord.Member,
                                client: discord.Client) -> None:
        """
        Called when a new member joins a server.

        Flow:
        1. Check cooldown (don't greet same user twice within 1 hour)
        2. Check server greet cooldown (don't greet too frequently)
        3. Gather server context (active channels, member count, server name)
        4. Call LLM to decide: should we greet? which channel? what message?
        5. If LLM says yes, send the greeting with human-like delay
        6. Record the greeting

        Never raises — all exceptions are caught so the bot never crashes.
        """
        try:
            guild = member.guild
            if guild is None:
                return

            user_id = member.id
            guild_id = guild.id

            # 1 & 2 — cooldown checks
            if not self._should_greet(user_id, guild_id):
                logger.debug(
                    f"[welcome] skipping {member} in '{guild.name}' "
                    f"(cooldown)"
                )
                return

            # 2.5 — probabilistic skip: a human doesn't greet EVERY join.
            # Sometimes she just lets them settle in. (env WELCOME_SKIP_CHANCE)
            try:
                skip_chance = float(os.getenv("WELCOME_SKIP_CHANCE", "0.15"))
            except ValueError:
                skip_chance = 0.15
            if random.random() < skip_chance:
                self._mark_greeted(user_id, guild_id)  # still burns cooldown
                logger.info(f"[welcome] randomly skipping greeting for {member} in '{guild.name}'")
                return

            # 3 — gather server context
            channel = self._find_most_active_channel(guild)
            if channel is None:
                logger.debug(
                    f"[welcome] no suitable channel in '{guild.name}', skipping"
                )
                return

            channels_info = self._build_channels_info(guild)

            # 4 — ask the LLM what to do
            decision = await self._generate_greeting(
                member, guild, channel, channels_info
            )

            if decision is None:
                # LLM failed — use a simple fallback greeting
                logger.warning(
                    f"[welcome] LLM failed for {member} in '{guild.name}', "
                    f"using fallback"
                )
                fallback_msg = (
                    f"yo {member.mention}, welcome to {guild.name}"
                )
                await self._send_greeting(channel, fallback_msg)
                self._mark_greeted(user_id, guild_id)
                return

            if not decision.get("should_greet", False):
                logger.info(
                    f"[welcome] LLM decided NOT to greet {member} "
                    f"in '{guild.name}' (tone={decision.get('tone')})"
                )
                # Still mark a light cooldown so we don't re-ask instantly
                self._mark_greeted(user_id, guild_id)
                return

            # The LLM may pick a different channel than our heuristic.
            target_channel = self._resolve_channel_from_decision(
                guild, decision, channel
            )

            message = decision.get("message") or ""
            message = self._inject_mention(message, member)

            if not message.strip():
                message = f"yo {member.mention}, welcome to {guild.name}"

            # 5 — send with human-like delay
            await self._send_greeting(target_channel, message)

            # 6 — record
            self._mark_greeted(user_id, guild_id)
            logger.info(
                f"[welcome] greeted {member} in "
                f"#{target_channel.name} ('{guild.name}') "
                f"tone={decision.get('tone')}"
            )

        except Exception as exc:  # never crash the bot
            logger.exception(f"[welcome] handle_new_member failed: {exc}")

    # ── Channel selection ─────────────────────────────────────────────────────

    def _find_most_active_channel(
        self, guild: discord.Guild
    ) -> Optional[discord.TextChannel]:
        """
        Algorithmically find the most active text channel in a guild.

        Heuristic:
        1. Consider only text channels we can send messages in.
        2. Score each channel (see ``_get_channel_activity_score``).
        3. Return the highest-scoring channel, or None if none qualify.
        """
        best: Optional[discord.TextChannel] = None
        best_score = -10**9

        try:
            for channel in guild.text_channels:
                # Must be able to send messages
                try:
                    if not channel.permissions_for(guild.me).send_messages:
                        continue
                except Exception:
                    continue

                score = self._get_channel_activity_score(channel)
                if score > best_score:
                    best_score = score
                    best = channel
        except Exception as exc:
            logger.exception(f"[welcome] _find_most_active_channel failed: {exc}")
            return None

        return best

    def _get_channel_activity_score(self,
                                    channel: discord.TextChannel) -> int:
        """
        Score a channel by how active / welcoming it is.

        Scoring:
        - +10 if name contains general/chat/main/lobby/welcome
        - +5  if name contains talk/off-topic/lounge
        - -10 if name contains rules/announcements/bot/commands/logs/...
        - +3  for each member in the channel (if visible)
        - +5  if channel nature cache says "general"
        """
        score = 0
        try:
            name = (channel.name or "").lower()

            for kw in _WELCOME_NAMES:
                if kw in name:
                    score += 10
                    break
            for kw in _SECONDARY_NAMES:
                if kw in name:
                    score += 5
                    break
            for kw in _AVOID_NAMES:
                if kw in name:
                    score -= 10
                    break

            # Member count (may be 0 if not visible / large guild)
            try:
                member_count = len(channel.members)
            except Exception:
                member_count = 0
            score += min(member_count * 3, 60)  # cap to avoid huge guild skew

            # Channel nature cache
            try:
                nature = get_cached_nature(str(channel.id))
                if nature == "general":
                    score += 5
            except Exception:
                pass
        except Exception:
            pass

        return score

    def _build_channels_info(self, guild: discord.Guild) -> str:
        """
        Build a short text summary of the top channels for the LLM prompt.
        """
        lines: List[str] = []
        try:
            scored = []
            for channel in guild.text_channels:
                try:
                    if not channel.permissions_for(guild.me).send_messages:
                        continue
                except Exception:
                    continue
                scored.append((self._get_channel_activity_score(channel),
                               channel))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, channel in scored[:8]:
                nature = None
                try:
                    nature = get_cached_nature(str(channel.id))
                except Exception:
                    pass
                nature_str = f" (nature: {nature})" if nature else ""
                lines.append(
                    f"- #{channel.name} — score {score}{nature_str}"
                )
        except Exception:
            pass
        return "\n".join(lines) if lines else "(no channels available)"

    def _resolve_channel_from_decision(
        self,
        guild: discord.Guild,
        decision: dict,
        fallback: discord.TextChannel,
    ) -> discord.TextChannel:
        """
        Resolve the channel the LLM chose. Falls back to the heuristic
        channel (or the first available text channel) if the LLM's choice
        can't be found or isn't writable.
        """
        chosen_name = (decision.get("channel_name") or "").lower().strip()
        if chosen_name:
            try:
                for channel in guild.text_channels:
                    try:
                        if channel.name.lower() == chosen_name and \
                                channel.permissions_for(guild.me).send_messages:
                            return channel
                    except Exception:
                        continue
                # fuzzy match
                for channel in guild.text_channels:
                    try:
                        if chosen_name in channel.name.lower() and \
                                channel.permissions_for(guild.me).send_messages:
                            return channel
                    except Exception:
                        continue
            except Exception:
                pass

        # fallback: heuristic channel, else first writable text channel
        if fallback is not None:
            return fallback
        try:
            for channel in guild.text_channels:
                try:
                    if channel.permissions_for(guild.me).send_messages:
                        return channel
                except Exception:
                    continue
        except Exception:
            pass
        # last resort — return the fallback even if non-ideal
        return fallback

    # ── LLM interaction ───────────────────────────────────────────────────────

    async def _generate_greeting(
        self,
        member: discord.Member,
        guild: discord.Guild,
        channel: discord.TextChannel,
        channels_info: str,
    ) -> Optional[dict]:
        """
        Call the LLM to generate a greeting decision.

        Returns a dict like::

            {
                "should_greet": true,
                "channel_name": "general",
                "message": "yo welcome to global friends, what brings u here?",
                "tone": "casual"
            }

        or None if the LLM call or JSON parse fails.
        """
        system = self._build_system_prompt()
        user = self._build_user_prompt(member, guild, channels_info)

        try:
            raw = await asyncio.to_thread(
                call_fast,
                "welcome_decision",
                system,
                user,
            )
        except Exception as exc:
            logger.exception(f"[welcome] LLM call failed: {exc}")
            return None

        return self._parse_decision(raw)

    def _parse_decision(self, raw: str) -> Optional[dict]:
        """Parse the LLM's JSON response, tolerating markdown fences."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Try to extract a JSON object from the text
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        logger.warning(f"[welcome] could not parse LLM decision: {raw!r}")
        return None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return f"""{prompts.PERSONA}

You are deciding how to welcome a new member to a Discord server.

DECISION FRAMEWORK:
1. SHOULD WE GREET? Consider:
   - Is the server active enough that a greeting will be seen?
   - Is the new member likely a real person (not a bot)?
   - Greet everyone unless there's a clear reason not to
2. WHICH CHANNEL? Choose the most active, welcoming channel:
   - Prefer general/chat/main channels
   - Avoid rules/announcements/bot-command channels
   - Choose where the greeting will be seen and responded to
3. WHAT MESSAGE? Generate a greeting in {prompts._ACTIVE.name.capitalize()}'s voice:
   - Casual, warm, not overly enthusiastic
   - Reference the server name or vibe if known
   - Maybe ask a question to start conversation
   - Keep it SHORT (1-2 sentences max)
   - VARIETY IS KEY — never the same shape twice. Most should be very short
     and effortless: "yo welcome", "ayyy welcome", "welcome bro", "welcome in".
     Only occasionally a longer playful greeting or a question. Real humans
     low-effort welcome people — sound like that.
   - Use British slang naturally
   - Don't be generic ("welcome to the server") — be specific and human
   - Do NOT include @mentions or markdown in the message; the system adds the mention itself

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:
{{
  "should_greet": true,
  "channel_name": "general",
  "message": "yo welcome to global friends, what brings u here?",
  "tone": "casual"
}}"""

    def _build_user_prompt(self, member: discord.Member,
                           guild: discord.Guild,
                           channels_info: str) -> str:
        """Build the user prompt with member and server context."""
        try:
            created_at = member.created_at.isoformat() \
                if member.created_at else "unknown"
        except Exception:
            created_at = "unknown"
        has_avatar = bool(getattr(member, "avatar", None) or
                          getattr(member, "display_avatar", None))

        return f"""New member joined:
- Username: {member.name}
- Display name: {member.display_name}
- Account created: {created_at}
- Avatar: {has_avatar}

Server: {guild.name} ({guild.member_count} members)

Available channels (by activity):
{channels_info}

Decide how to welcome them and respond with the JSON plan."""

    # ── Cooldown management ───────────────────────────────────────────────────

    def _should_greet(self, user_id: int, guild_id: int) -> bool:
        """
        Check cooldowns — should we greet this user in this server?

        Returns False if:
        - the user was greeted anywhere within ``_cooldown_s`` (1h), or
        - this guild had any greeting within ``_greet_cooldown_s`` (2m).
        """
        now = time.time()

        last_user = self._greeted_users.get(user_id)
        if last_user is not None and now - last_user < self._cooldown_s:
            return False

        last_guild = self._last_greet_time.get(guild_id)
        if last_guild is not None and now - last_guild < self._greet_cooldown_s:
            return False

        return True

    def _mark_greeted(self, user_id: int, guild_id: int) -> None:
        """Record that we greeted a user (updates both cooldown trackers)."""
        now = time.time()
        self._greeted_users[user_id] = now
        self._last_greet_time[guild_id] = now

    # ── Sending ───────────────────────────────────────────────────────────────

    async def _send_greeting(self, channel: discord.TextChannel,
                             message: str) -> None:
        """
        Send a greeting with human-like behaviour:
        - wait 10-30 seconds before sending (don't greet instantly)
        - trigger the typing indicator first
        - never raise on failure
        """
        try:
            # Human-like delay — she notices someone joined, wanders over,
            # THEN greets. Not instant.
            delay = random.uniform(15, 45)
            await asyncio.sleep(delay)

            try:
                async with channel.typing():
                    # short typing pause proportional to message length
                    await asyncio.sleep(
                        min(len(message) * 0.04, 4.0)
                    )
            except Exception:
                pass

            await channel.send(message)
        except Exception as exc:
            logger.exception(f"[welcome] failed to send greeting: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_mention(message: str, member: discord.Member) -> str:
        """
        Ensure the new member is mentioned in the greeting.

        If the LLM already included a mention, leave it. Otherwise prepend
        a casual mention so the new member gets pinged.
        """
        if not message:
            return f"yo {member.mention}"
        # crude check for an existing mention
        if "<@" in message or member.mention in message:
            return message
        # Prepend the mention naturally
        return f"{message} {member.mention}".strip()


# ── Singleton ─────────────────────────────────────────────────────────────────

_welcome_system: Optional[WelcomeSystem] = None


def get_welcome_system() -> WelcomeSystem:
    """Get the global :class:`WelcomeSystem` singleton."""
    global _welcome_system
    if _welcome_system is None:
        _welcome_system = WelcomeSystem()
    return _welcome_system
