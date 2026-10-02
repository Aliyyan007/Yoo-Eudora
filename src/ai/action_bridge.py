"""Outsider action-worker bridge.

Connects the host bot's reply path to the vendored action engine
(src/action_engine/ â€” the action-performing machinery transplanted from
the "Robust Action Performer System" codebase). The host keeps its own
chat, persona, context, and voice engines completely untouched â€” this
module only intercepts *action-performing* requests that the native
command regexes don't already handle, and executes them through the
vendored tool-calling agent in near-isolation.

Flow (text):
    message â†’ _handle_reply â†’ detect_command (native, handles known
    commands â€” untouched) â†’ if no native match â†’ ActionWorker
    â†’ classify() decides ACTION vs CHAT â†’ Agent.run(categories=...)
    â†’ tools perform the action â†’ silent return (never sends a reply).

Flow (voice):
    voice transcript â†’ pipeline._finalize_inner â†’ native leave/sing
    paths run first â†’ _action_cb â†’ ActionWorker â†’ same engine.

Design guarantees:
- No "done"/confirmation messages â€” the agent's return text is dropped
  (the donor's bot.py is what sent it; we never forward it).
- A human-like delay before acting, plus a typing indicator on the text
  path while the worker runs.
- leave_vc can never reach the worker: the native path handles it first,
  and the vendored leave_voice tool is removed from the schema set and
  stubbed out anyway.
- Permission gate wraps dispatch: state-changing/third-party tools are
  owner-only; non-owners get the benign read/send/react set, with
  ping_roles stripped from their sends (blocks @everyone mass-pings).
- Hard failure â†’ returns False â†’ caller falls through to the normal
  reply path, so a broken worker degrades to plain chatting.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Optional

from loguru import logger


# â”€â”€ Tunables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_ENABLED = os.getenv("ACTION_WORKER_ENABLED", "1") != "0"
_DELAY_MIN = float(os.getenv("ACTION_DELAY_MIN", "2.0"))
_DELAY_MAX = float(os.getenv("ACTION_DELAY_MAX", "6.0"))
_TIMEOUT = float(os.getenv("ACTION_TIMEOUT_S", "45"))

# Tools a non-owner requester may never trigger, regardless of phrasing.
# Everything not listed here OR in _ALLOWED_FOR_ALL is denied by default.
_OWNER_ONLY = {
    # third-party side effects / arbitrary command invocation
    "use_slash_command", "bump_with_bot", "bump_all", "find_bump_commands",
    # persistent scheduled spam that survives restarts
    "schedule_message", "list_scheduled", "stop_scheduled",
    # persistent owner preferences
    "set_preference", "get_preferences", "delete_preference",
    # account/guild state mutation
    "change_nickname", "change_status", "change_custom_status", "change_bio",
    # own-message management (delete/edit can rewrite bot history)
    "delete_message", "delete_last_message", "edit_message",
    "cleanup_my_messages",
    # DM-ing arbitrary users is a spam/abuse vector
    "send_dm",
    # profile reads reveal account details
    "get_my_profile",
    # moving the bot's own voice connection around on anyone's say-so
    "move_voice",
}

# Non-owner tools where role pings are silently stripped â€” they can resolve
# @everyone/@here and mass-ping the whole server.
_STRIP_PING_ROLES = {
    "send_message", "send_dm", "send_gif", "send_vc_text",
    "send_multiple_gifs",
}

# Human-readable labels for performed tools â€” fed back into the main reply
# context so the bot can acknowledge the action naturally.
_ACTION_LABELS = {
    "send_message": "sent the message", "send_dm": "sent the DM",
    "send_vc_text": "sent the VC text", "send_gif": "sent the GIF",
    "send_sticker": "sent the sticker", "send_multiple_gifs": "sent the GIFs",
    "send_multiple_stickers": "sent the stickers",
    "react_to_message": "reacted", "react_to_user_latest": "reacted",
    "react_to_recent": "reacted to the messages",
    "join_voice": "joined the voice channel", "move_voice": "moved voice channels",
    "mute_self": "muted", "deafen_self": "deafened",
    "delete_message": "deleted the message", "delete_last_message": "deleted the messages",
    "edit_message": "edited the message", "cleanup_my_messages": "cleaned up the messages",
    "use_slash_command": "ran the slash command", "bump_with_bot": "bumped",
    "bump_all": "bumped with all bump bots", "schedule_message": "scheduled it",
    "stop_scheduled": "stopped the scheduled task",
    "change_nickname": "changed the nickname", "change_status": "changed the status",
    "change_custom_status": "changed the status", "change_bio": "changed the bio",
    "say_in_vc": "spoke in the VC", "speak_in_stage": "spoke on stage",
}


def _note_for(performed: list) -> Optional[str]:
    """('send_message', True), ... -> 'sent the message; reacted' summary.
    Returns an honest 'tried but failed' note when tools ran but all failed."""
    done = [_ACTION_LABELS.get(n, n.replace("_", " ")) for n, ok in performed if ok]
    if done:
        return "; ".join(dict.fromkeys(done))
    if performed:
        tried = [_ACTION_LABELS.get(n, n.replace("_", " ")) for n, _ in performed]
        return f"tried to {'; '.join(dict.fromkeys(tried))} but it didn't work"
    return None


# Stricter confirmation prompt â€” the heuristic router flags anything with an
# action word, including declarative statements ("users with X can use @everyone
# ping"). This second check kills those: only an imperative instruction
# directed AT the bot may reach the worker.
_CONFIRM_PROMPT = (
    "You decide if a Discord message is a direct instruction TO the bot to DO "
    "something right now. Answer only YES or NO.\n"
    "YES examples: 'send a msg to #general', 'ping sarah', 'react with fire to "
    "that', 'join my vc', 'use /bump in here', 'delete your last message'.\n"
    "NO examples: statements, questions or chatter that merely MENTION actions "
    "â€” 'users with the member role can use @everyone ping once a week', 'he "
    "pinged me yesterday', 'how do I send a gif', 'can people ping roles here', "
    "'im gonna go', gossip, jokes, or talk aimed at someone else.\n"
    "Answer with only YES or NO."
)

# Info-question arbiter — catches lookups phrased as questions ("who's our
# newest member?", "gimme info on sarah") that never trip action verbs.
_INFO_CONFIRM_PROMPT = (
    "You decide if a Discord message asks for factual information about the "
    "server that a helper could LOOK UP — members (who joined/left, member "
    "info, avatars, bios, ids), channels, roles, or recent messages. "
    "Answer only YES or NO.\n"
    "YES examples: 'who's the most recent joiner', 'gimme info on sarah', "
    "'how many members are online', 'when was #general made', 'what's his id'.\n"
    "NO examples: personal questions ('wyd', 'how are you'), opinions, plans, "
    "or instructions to DO something ('send a message', 'ping him').\n"
    "Answer with only YES or NO."
)

_CONFIRM_ENABLED = os.getenv("ACTION_CONFIRM_LLM", "1").strip() != "0"

# Silently-delegated to the host's voice system â€” these SHOULD still work for
# anyone (joining/muting are ordinary VC requests the host already honors).
# leave_voice is unreachable: schema-removed in vendored registry.


def _is_owner_id(user_id) -> bool:
    try:
        from .owner_system import is_owner
        return is_owner(str(user_id))
    except Exception:
        return False


class ActionWorker:
    """The 'outsider' engine: classify â†’ guard â†’ run â†’ silent."""

    def __init__(self, client):
        self.client = client
        self._agent = None
        self._ready = False
        self._inflight: set = set()          # (scope_id, user_id) dedup
        self._tasks: set = set()             # background exec tasks (awaitable)

    # ------------------------------------------------------------------ #
    def _ensure_ready(self) -> bool:
        """Lazily wire the vendored engine. Never raises â€” a misconfigured
        engine must not take the reply path down with it."""
        if self._ready:
            return True
        try:
            from src.action_engine.config.settings import settings as _ae_settings
            # Sync the vendored owner list with the host's OWNER_IDS env â€”
            # vendored owner-gated tools read settings.owner_ids.
            owner_env = os.getenv("OWNER_IDS") or os.getenv("OWNER_ID") or ""
            if owner_env.strip() and _ae_settings.owner_user_id in ("", "0"):
                _ae_settings.owner_user_id = owner_env

            from src.action_engine.tools.context import ToolContext  # noqa: F401
            from src.action_engine.ai import agent as _agent_mod
            from src.action_engine.tools.registry import dispatch as _real_dispatch

            worker = self

            async def _guarded_dispatch(ctx, name, args):
                """Wrapper around the vendored dispatch â€” enforces the
                permission gate and records what actually ran."""
                allowed, args = worker._check_allowed(ctx, name, args)
                if not allowed:
                    logger.info(f"[actions] denied '{name}' for user {ctx.author_id}")
                    return json.dumps({"error": "not allowed for this user"})
                result = await _real_dispatch(ctx, name, args)
                ok = '"error"' not in (result or "")[:120]
                try:
                    ctx.performed.append((name, ok))
                except AttributeError:
                    pass
                return result

            # Patch the dispatch binding inside the vendored agent module
            # (it did `from tools.registry import dispatch` at import).
            _agent_mod.dispatch = _guarded_dispatch
            self._ctx_cls = ToolContext
            self._agent_mod = _agent_mod

            # Scheduler singleton â€” vendored schedule tools look it up.
            try:
                from src.action_engine.core import scheduler as _sched
                if _sched.scheduler is None:
                    _sched.init(self.client)
                    asyncio.create_task(_sched.scheduler.start())
            except Exception as e:
                logger.warning(f"[actions] scheduler init failed (non-fatal): {e}")

            self._ready = True
            logger.info("[actions] action worker initialised")
        except Exception as e:
            logger.warning(f"[actions] engine init failed â€” disabled this session: {e}")
        return self._ready

    def _agent_for(self, ctx):
        # A fresh Agent per run — background execs can overlap and a shared
        # agent would swap ctx mid-loop (cross-talk between requests).
        agent = self._agent_mod.Agent(ctx)
        self._agent = agent  # tests poke at this handle to reset the pool
        return agent

    # ------------------------------------------------------------------ #
    def _check_allowed(self, ctx, name, args):
        """(allowed, possibly-sanitized-args)"""
        if _is_owner_id(ctx.author_id):
            return True, args
        if name in _OWNER_ONLY:
            return False, args
        args = dict(args or {})
        if name in _STRIP_PING_ROLES and args.get("ping_roles"):
            args["ping_roles"] = []          # non-owner: no role pings
        if name in _STRIP_PING_ROLES and args.get("ping_users"):
            # sanity: cap at 5 pings so "ping everyone one-by-one" loops
            args["ping_users"] = list(args["ping_users"])[:5]
        return True, args

    # ------------------------------------------------------------------ #
    async def _confirm_action(self, text: str) -> bool:
        """Second-opinion check: is this a direct imperative TO the bot?
        The heuristic flags anything containing action words â€” this cheap
        YES/NO arbiter rejects declarative statements that merely mention
        actions ("users with X role can use @everyone ping once a week").
        Fails OPEN â†’ a Groq outage never breaks actions."""
        if not _CONFIRM_ENABLED:
            return True
        try:
            from src.action_engine.core.groq_pool import get_pool
            from src.action_engine.config.settings import settings as _ae_settings
            resp = await get_pool().chat(
                model=_ae_settings.router_model,
                messages=[
                    {"role": "system", "content": _CONFIRM_PROMPT},
                    {"role": "user", "content": text[:400]},
                ],
                tools=None, tool_choice="none",
                temperature=0.0, max_tokens=8,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            if "NO" in verdict:
                logger.info(f"[actions] heuristic said ACTION but arbiter said NO: '{text[:60]}'")
                return False
            return True
        except Exception as e:
            logger.debug(f"[actions] confirm arbiter failed, allowing: {e}")
            return True

    async def _confirm_info(self, text: str) -> bool:
        """Same second-opinion shape as _confirm_action, but for questions
        asking for server facts the worker can look up (who joined, member
        info, channel details). Fails CLOSED — a miss just means the normal
        reply engine answers instead."""
        if not _CONFIRM_ENABLED:
            return True
        try:
            from src.action_engine.core.groq_pool import get_pool
            from src.action_engine.config.settings import settings as _ae_settings
            resp = await get_pool().chat(
                model=_ae_settings.router_model,
                messages=[
                    {"role": "system", "content": _INFO_CONFIRM_PROMPT},
                    {"role": "user", "content": text[:400]},
                ],
                tools=None, tool_choice="none",
                temperature=0.0, max_tokens=8,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            return "YES" in verdict
        except Exception as e:
            logger.debug(f"[actions] info arbiter failed, skipping: {e}")
            return False

    async def _classify(self, text: str):
        """ACTION route or None. Heuristic first (free), LLM arbiter only
        for genuinely ambiguous utterances."""
        from src.action_engine.ai import router
        route = router.classify(text)
        if route is None:
            route = await router.classify_llm(text)
        return route

    # Read-only categories â€” a request needing ONLY these is an info
    # lookup ("who joined recently?", "what's sarah's id?"). Info runs
    # BEFORE the reply so the answer can be relayed; exec replies first
    # and runs in the background (human-like ordering).
    _INFO_CATS = frozenset({"channels", "members", "roles", "messages"})

    # Question forms that ask for server facts the worker can look up —
    # the heuristic CHATs these because they're questions, not commands.
    _INFO_HINTS = re.compile(
        r"\b(who('s| is)?|whose|what('s| is)|which|how many|count|list|info|"
        r"details|recent|newest|latest|avatar|bio|profile|joined|"
        r"members?|online|channels?|roles?)\b",
        re.I)

    async def classify_request(self, text: str) -> Optional[tuple]:
        """Cheap gate — returns (kind, categories) with kind 'info' or
        'exec', or None for chat. Callers pass categories through to the
        queue/run call so the agent gets the router's filtered tool set."""
        if not _ENABLED or not self._ensure_ready():
            return None
        try:
            route = await self._classify(text)
        except Exception as e:
            logger.debug(f"[actions] classify failed: {e}")
            return None
        if route is None or route.kind != "action":
            # A question-shaped server-facts request that never reached the
            # action route — "gimme info on sarah", "who joined most
            # recently". Info lookups are read-only: a false positive is
            # harmless (worst case a wasted lookup).
            if self._INFO_HINTS.search(text) and await self._confirm_info(text):
                return ("info", set(self._INFO_CATS))
            return None
        if route.categories and route.categories <= self._INFO_CATS:
            return ("info", route.categories)  # read-only — no imperative gate
        if not await self._confirm_action(text):
            return None
        return ("exec", route.categories or set())

    def _ctx_for(self, channel, author_id: int):
        guild = getattr(channel, "guild", None)
        if guild is None:
            guilds = getattr(self.client, "guilds", [])
            guild = guilds[0] if len(guilds) == 1 else None
        ctx = self._ctx_cls(
            bot=self.client,
            guild=guild,
            current_channel_id=channel.id,
            author_id=author_id,
        )
        ctx.performed = []
        return ctx

    async def _run_agent(self, ctx, text: str, categories, mode: str,
                         label: str, speaker_name: Optional[str] = None) -> Optional[str]:
        """Human-like delay â†’ agent.run â†’ done. The agent's reply text is
        NEVER sent to the channel (silent worker); performed tools are
        tracked on ctx.performed. Returns the agent's reply text."""
        await asyncio.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))
        agent = self._agent_for(ctx)
        try:
            reply = await asyncio.wait_for(
                agent.run(text, mode=mode, categories=categories or None,
                          speaker_name=speaker_name),
                timeout=_TIMEOUT,
            )
            logger.info(
                f"[actions] {label} '{text[:60]}' â†’ performed="
                f"{getattr(ctx, 'performed', [])} (agent said: {str(reply)[:60]!r} â€” suppressed)"
            )
            return str(reply or "")
        except asyncio.TimeoutError:
            logger.warning(f"[actions] {label} worker timed out: '{text[:60]}'")
            return _note_for(getattr(ctx, "performed", []))
        except Exception as e:
            logger.error(f"[actions] {label} worker error: {e!r}")
            return None

    # ------------------------------------------------------------------ #
    #  TEXT entry points
    # ------------------------------------------------------------------ #
    def queue_text_action(self, message, trigger_text: str,
                          categories=None) -> bool:
        """Queue an EXEC action in the background â€” the normal reply goes
        out first, then the action fires after a human-like delay (the
        requester sees 'on it' then the thing happens, like a person).
        False when a request for this user+channel is already inflight."""
        key = (str(message.channel.id), str(message.author.id))
        if key in self._inflight:
            return False
        self._inflight.add(key)

        from src.action_engine.config.prompts import CHAT_MODE

        async def _job():
            try:
                ctx = self._ctx_for(message.channel, message.author.id)
                try:
                    async with message.channel.typing():
                        await self._run_agent(
                            ctx, trigger_text, categories, CHAT_MODE, "text",
                            speaker_name=message.author.display_name)
                except Exception:
                    await self._run_agent(
                        ctx, trigger_text, categories, CHAT_MODE, "text",
                        speaker_name=message.author.display_name)
            finally:
                self._inflight.discard(key)

        t = asyncio.create_task(_job())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(
            lambda f: None if f.cancelled() or not f.exception()
            else logger.debug(f"[actions] bg task error: {f.exception()}"))
        return True

    async def run_text_info(self, message, trigger_text: str) -> Optional[str]:
        """Run an INFO lookup now (read-only tools) and return the answer
        text â€” the caller relays it through the normal reply engine."""
        key = (str(message.channel.id), str(message.author.id))
        if key in self._inflight:
            return None
        self._inflight.add(key)
        try:
            from src.action_engine.config.prompts import CHAT_MODE
            ctx = self._ctx_for(message.channel, message.author.id)
            try:
                async with message.channel.typing():
                    answer = await self._run_agent(
                        ctx, trigger_text, self._INFO_CATS, CHAT_MODE, "info",
                        speaker_name=message.author.display_name)
            except Exception:
                answer = await self._run_agent(
                    ctx, trigger_text, self._INFO_CATS, CHAT_MODE, "info",
                    speaker_name=message.author.display_name)
            if answer and not answer.startswith("(sorry"):
                return answer
            return None
        finally:
            self._inflight.discard(key)

    # ------------------------------------------------------------------ #
    #  VOICE entry points
    # ------------------------------------------------------------------ #
    def queue_voice_action(self, user_id: int, transcript: str, voice_channel,
                           categories=None) -> bool:
        key = (str(getattr(voice_channel, "id", 0)), str(user_id))
        if key in self._inflight:
            return False
        self._inflight.add(key)

        guild = getattr(voice_channel, "guild", None)
        member = guild.get_member(user_id) if guild else None
        speaker = getattr(member, "display_name", None) or str(user_id)

        from src.action_engine.config.prompts import VOICE_MODE

        async def _job():
            try:
                ctx = self._ctx_for(voice_channel, user_id)
                await self._run_agent(
                    ctx, f"[{speaker}]: {transcript}", categories, VOICE_MODE, "voice")
            finally:
                self._inflight.discard(key)

        t = asyncio.create_task(_job())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(
            lambda f: None if f.cancelled() or not f.exception()
            else logger.debug(f"[actions] bg voice task error: {f.exception()}"))
        return True

    async def run_voice_info(self, user_id: int, transcript: str, voice_channel) -> Optional[str]:
        key = (str(getattr(voice_channel, "id", 0)), str(user_id))
        if key in self._inflight:
            return None
        self._inflight.add(key)
        try:
            guild = getattr(voice_channel, "guild", None)
            member = guild.get_member(user_id) if guild else None
            speaker = getattr(member, "display_name", None) or str(user_id)
            from src.action_engine.config.prompts import VOICE_MODE
            ctx = self._ctx_for(voice_channel, user_id)
            answer = await self._run_agent(
                ctx, f"[{speaker}]: {transcript}", self._INFO_CATS, VOICE_MODE, "voice-info")
            if answer and not answer.startswith("(sorry"):
                return answer
            return None
        finally:
            self._inflight.discard(key)


# ---------------------------------------------------------------------- #
_worker: Optional[ActionWorker] = None


def get_action_worker(client) -> ActionWorker:
    global _worker
    if _worker is None:
        _worker = ActionWorker(client)
    return _worker
