"""Proactive voice engagement — bot independently asks questions to keep VC active.

When the conversation dies (no one has spoken for a while), the bot:
1. Picks a user to address (the quietest one, or a random one)
2. Generates a contextual question using the LLM
3. Speaks it via TTS, addressing the user by name

This makes the bot feel like a real person who's actively engaged in the
conversation, not just a passive listener that only responds when spoken to.

Multi-user handling:
- When multiple users are in the VC, the bot says the user's display name
  first to address them: "hey Sarah, what do you think about..."
- The bot picks the user who's been quiet the longest, to give everyone
  a chance to participate
- The bot avoids asking the same user too many times in a row
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .pipeline import VoicePipeline
    from .manager import VoiceManager


# ── Proactive engagement settings ───────────────────────────────────────────
_SILENCE_THRESHOLD_S = 40      # Bot asks a question after 40s of silence
_CHECK_INTERVAL_S = 10         # Check every 10s
_MIN_USERS_TO_ENGAGE = 1       # Need at least 1 user in VC
_COOLDOWN_S = 90               # Min 90s between proactive questions (was 45s — too frequent)
_MAX_RECENT_USERS = 3          # Don't ask the same user again if they were
                               # asked in the last 3 proactive turns
_BOT_SPEECH_COOLDOWN_S = 15    # Don't proactively engage if bot spoke in last 15s

# Pre-written fallback questions (used if LLM fails or for speed)
_FALLBACK_QUESTIONS = [
    "so what have you been up to?",
    "you been doing anything interesting lately?",
    "what music are you into at the moment?",
    "so what are you working on these days?",
    "you been watching anything good recently?",
    "what's your take on the whole AI thing?",
    "so, random question — what's your favourite food?",
    "you got any plans for the weekend?",
    "what's been the highlight of your day so far?",
    "so how's your week been?",
    "you been anywhere nice lately?",
    "what's a hobby you've always wanted to try?",
    "so, controversial take — pineapple on pizza, yes or no?",
    "what's the last thing that made you laugh?",
    "so what are you studying, or are you working?",
    "you more of a morning person or a night owl?",
    "what's your favourite way to waste time?",
    "so, if you could go anywhere right now, where would you go?",
    "what's the best thing you've eaten this week?",
    "you been listening to any good podcasts lately?",
]

# Questions that work well when addressing a specific user by name
_ADDRESSED_QUESTIONS = [
    "hey {name}, what have you been up to?",
    "so {name}, how's your day going?",
    "{name}, you been doing anything interesting lately?",
    "hey {name}, what are you working on these days?",
    "so {name}, what music are you into at the moment?",
    "{name}, you been watching anything good recently?",
    "hey {name}, how's your week been?",
    "so {name}, what's the highlight of your day so far?",
    "{name}, you got any plans for the weekend?",
    "hey {name}, what's the last thing that made you laugh?",
]


class ProactiveEngager:
    """Background task that proactively asks questions when VC goes quiet."""

    def __init__(self, voice_manager: "VoiceManager"):
        self._manager = voice_manager
        self._task: Optional[asyncio.Task] = None
        self._recent_asked: list[int] = []  # user_ids asked recently
        self._last_ask_time: float = 0.0
        # Presence-check escalation: consecutive proactive engages that got
        # no response → the questions shift from topics to "you still there?"
        self._unanswered: dict[int, int] = {}
        self._engage_mark: dict[int, float] = {}

    def start(self, client):
        """Start the proactive engagement background loop."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(client))
        logger.info("[proactive] Proactive engagement started")

    def stop(self):
        """Stop the proactive engagement loop."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        logger.info("[proactive] Proactive engagement stopped")

    async def _loop(self, client):
        """Main loop — check for silence and ask questions."""
        while True:
            try:
                await asyncio.sleep(_CHECK_INTERVAL_S)

                for guild_id, pipeline in list(self._manager._pipelines.items()):
                    try:
                        await self._check_and_engage(client, guild_id, pipeline)
                    except Exception as e:
                        logger.debug(f"[proactive] Error for guild {guild_id}: {e}")

            except asyncio.CancelledError:
                logger.info("[proactive] Loop cancelled")
                break
            except Exception as e:
                logger.debug(f"[proactive] Loop error: {e}")

    async def _check_and_engage(self, client, guild_id: int, pipeline: "VoicePipeline"):
        """Check if we should proactively engage in a specific guild's VC."""
        # Don't engage if the bot is currently speaking
        if pipeline.is_speaking:
            return

        # Don't engage if the bot spoke recently (avoid stacking)
        now = time.time()
        if now - pipeline._last_bot_speech_time < _BOT_SPEECH_COOLDOWN_S:
            return

        # Check cooldown
        if now - self._last_ask_time < _COOLDOWN_S:
            return

        # Get the VC channel to check who's in it
        vc_id = self._manager._current_vcs.get(guild_id)
        if vc_id is None:
            return

        guild = client.get_guild(guild_id)
        if guild is None:
            return

        vc_channel = guild.get_channel(vc_id)
        if vc_channel is None:
            return

        # Get non-bot members in the VC
        non_bot_members = [m for m in vc_channel.members if not m.bot and m.id != self._manager._bot_id]
        if len(non_bot_members) < _MIN_USERS_TO_ENGAGE:
            return

        # Check if there's been enough silence
        last_speech = pipeline.get_last_speech_time()
        if last_speech == 0:
            # No one has spoken yet — wait a bit more
            return

        silence_duration = now - last_speech
        if silence_duration < _SILENCE_THRESHOLD_S:
            return

        # Track unanswered rounds — if the silence timestamp hasn't moved
        # since the last engage, nobody responded to the last question
        if last_speech < self._engage_mark.get(guild_id, 0):
            self._unanswered[guild_id] = self._unanswered.get(guild_id, 0) + 1
        else:
            self._unanswered[guild_id] = 0

        # Don't engage if someone's mic is currently active
        for uid, active in pipeline._user_mic_active.items():
            if active:
                logger.debug(f"[proactive] User {uid} mic is active, not engaging")
                return

        # Pick a user to address
        target_user = self._pick_target_user(pipeline, non_bot_members)
        if target_user is None:
            return

        target_id, target_name = target_user

        # If she's asked twice already with zero response, she switches to
        # presence checks ("you still there?") — algorithmic escalation, not
        # a fixed pattern
        presence_check = self._unanswered.get(guild_id, 0) >= 2 and random.random() < 0.6

        logger.info(f"[proactive] {silence_duration:.0f}s silence — engaging {target_name} (id={target_id}, presence={presence_check})")

        # Generate and speak the question
        await self._ask_question(pipeline, target_name, non_bot_members, guild_id, client, presence_check=presence_check)

        self._engage_mark[guild_id] = now

        # Track this engagement
        self._last_ask_time = time.time()
        self._recent_asked.append(target_id)
        if len(self._recent_asked) > _MAX_RECENT_USERS:
            self._recent_asked.pop(0)

    def _pick_target_user(self, pipeline: "VoicePipeline", members: list) -> Optional[tuple]:
        """Pick the best user to address. Prefers the quietest user who
        hasn't been asked recently."""
        now = time.time()
        candidates = []

        for member in members:
            # Skip users who were asked recently
            if member.id in self._recent_asked:
                continue
            # Skip users who just said goodbye — they're leaving, let them go
            if now - getattr(pipeline, "_user_farewell_at", {}).get(member.id, 0) < 300:
                continue

            name = member.display_name or member.name
            # Update the pipeline's display name tracking
            pipeline.set_user_display_name(member.id, name)

            # Get how long they've been quiet
            last_speech = pipeline._user_last_speech_time.get(member.id, 0)
            if last_speech > 0:
                silence = now - last_speech
            else:
                # Never spoken — high priority to engage them
                silence = float('inf')

            candidates.append((silence, member.id, name))

        if not candidates:
            # All users were asked recently — reset and try again
            if self._recent_asked:
                self._recent_asked.clear()
                return self._pick_target_user(pipeline, members)
            return None

        # Sort by silence descending — quietest user first
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 70% chance: pick the quietest user
        # 30% chance: pick a random user (for variety)
        if random.random() < 0.7 or len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = random.choice(candidates[:3])

        return (chosen[1], chosen[2])

    async def _ask_question(self, pipeline: "VoicePipeline", target_name: str,
                             all_members: list, guild_id: int, client,
                             presence_check: bool = False):
        """Generate a question and speak it. Uses LLM for contextual questions,
        falls back to pre-written ones."""
        # Get conversation history for context
        history_text = ""
        try:
            # Access the discord client's voice history
            discord_client = client
            if hasattr(discord_client, '_get_voice_history'):
                history = discord_client._get_voice_history(guild_id)
                history_text = discord_client._format_voice_history(history)
        except Exception:
            pass

        # Get other user names for context
        other_names = [m.display_name or m.name for m in all_members if m.display_name != target_name]

        # Try LLM-generated question for more natural, contextual engagement
        try:
            from ..ai import llm
            from ..ai import prompts as ai_prompts

            system = ai_prompts.VOICE_REPLY_SYSTEM
            user_prompt = self._build_proactive_prompt(
                target_name, other_names, history_text, presence_check
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.call_voice(
                    "proactive_question", system, user_prompt,
                    want_json=False,
                )
            )

            if response and response.strip():
                # Clean up the response
                import re
                response = re.sub(r'\*+([^*]+)\*+', r'\1', response)
                response = response.replace('\n', ' ').strip()
                from ..ai.reply import clean_for_speech
                response = clean_for_speech(response)

                if response and len(response) > 5:
                    logger.info(f"[proactive] LLM question for {target_name}: {response[:80]}")
                    await pipeline._speak(response)
                    return
        except Exception as e:
            logger.debug(f"[proactive] LLM question failed: {e}")

        # Fallback to pre-written question
        if presence_check:
            presence_lines = [
                f"hello? {target_name}, you still there?",
                f"{target_name} — everyone go afk or what?",
                f"why's it gone dead silent, {target_name} you alive?",
                f"oi {target_name}, still with us?",
                f"{target_name}, it's gone proper quiet — you lot still there?",
            ]
            question = random.choice(presence_lines)
            logger.info(f"[proactive] Presence check for {target_name}: {question}")
            await pipeline._speak(question)
            return

        if len(all_members) > 1:
            # Multi-user: address by name
            question = random.choice(_ADDRESSED_QUESTIONS).format(name=target_name)
        else:
            # Single user: don't need to address by name every time
            if random.random() < 0.5:
                question = random.choice(_ADDRESSED_QUESTIONS).format(name=target_name)
            else:
                question = random.choice(_FALLBACK_QUESTIONS)

        logger.info(f"[proactive] Fallback question for {target_name}: {question}")
        await pipeline._speak(question)

    def _build_proactive_prompt(self, target_name: str, other_names: list,
                                 history_text: str, presence_check: bool = False) -> str:
        """Build the prompt for generating a proactive question."""
        parts = []

        if presence_check:
            parts.append(
                f"You've asked questions twice and got nothing — the call feels "
                f"empty now. Ask {target_name} (or the group) if they're still "
                f"there, why it's gone quiet — like you're checking if anyone's alive."
            )
        else:
            parts.append(
                f"The voice call has gone quiet. You want to get the conversation going again. "
                f"Ask {target_name} a question to engage them."
            )

        if other_names:
            parts.append(f"[others in the call: {', '.join(other_names)}]")

        if history_text and history_text.strip():
            parts.append(f"[recent conversation context]\n{history_text}")

        parts.append(
            f"\nGenerate a natural, casual question to ask {target_name}. "
            f"Rules:\n"
            f"- Address them by name at the start: 'hey {target_name}, ...' or 'so {target_name}, ...'\n"
            f"- Ask something genuine and interesting — not generic small talk\n"
            f"- If there's conversation context, reference it (e.g. 'so you mentioned you like pizza...')\n"
            f"- 1-2 sentences max. End with a question.\n"
            f"- Use full words (no text abbreviations). Sound natural when spoken.\n"
            f"- Don't ask the same thing twice. Be creative.\n"
            f"- Say ONLY what you'd speak out loud. No formatting."
        )

        return "\n".join(parts)
