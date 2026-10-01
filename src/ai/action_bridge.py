"""Outsider action-worker bridge.

Connects the host bot's reply path to the vendored action engine
(src/action_engine/ — the action-performing machinery transplanted from
the "Robust Action Performer System" codebase). The host keeps its own
chat, persona, context, and voice engines completely untouched — this
module only intercepts *action-performing* requests that the native
command regexes don't already handle, and executes them through the
vendored tool-calling agent in near-isolation.

Flow (text):
    message → _handle_reply → detect_command (native, handles known
    commands — untouched) → if no native match → ActionWorker
    → classify() decides ACTION vs CHAT → Agent.run(categories=...)
    → tools perform the action → silent return (never sends a reply).

Flow (voice):
    voice transcript → pipeline._finalize_inner → native leave/sing
    paths run first → _action_cb → ActionWorker → same engine.

Design guarantees:
- No "done"/confirmation messages — the agent's return text is dropped
  (the donor's bot.py is what sent it; we never forward it).
- A human-like delay before acting, plus a typing indicator on the text
  path while the worker runs.
- leave_vc can never reach the worker: the native path handles it first,
  and the vendored leave_voice tool is removed from the schema set and
  stubbed out anyway.
- Permission gate wraps dispatch: state-changing/third-party tools are
  owner-only; non-owners get the benign read/send/react set, with
  ping_roles stripped from their sends (blocks @everyone mass-pings).
- Hard failure → returns False → caller falls through to the normal
  reply path, so a broken worker degrades to plain chatting.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Optional

from loguru import logger


# ── Tunables ──────────────────────────────────────────────────────────────
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

# Non-owner tools where role pings are silently stripped — they can resolve
# @everyone/@here and mass-ping the whole server.
_STRIP_PING_ROLES = {
    "send_message", "send_dm", "send_gif", "send_vc_text",
    "send_multiple_gifs",
}

# Silently-delegated to the host's voice system — these SHOULD still work for
# anyone (joining/muting are ordinary VC requests the host already honors).
# leave_voice is unreachable: schema-removed in vendored registry.


def _is_owner_id(user_id) -> bool:
    try:
        from .owner_system import is_owner
        return is_owner(str(user_id))
    except Exception:
        return False


class ActionWorker:
    """The 'outsider' engine: classify → guard → run → silent."""

    def __init__(self, client):
        self.client = client
        self._agent = None
        self._ready = False
        self._inflight: set = set()          # (scope_id, user_id) dedup

    # ------------------------------------------------------------------ #
    def _ensure_ready(self) -> bool:
        """Lazily wire the vendored engine. Never raises — a misconfigured
        engine must not take the reply path down with it."""
        if self._ready:
            return True
        try:
            from src.action_engine.config.settings import settings as _ae_settings
            # Sync the vendored owner list with the host's OWNER_IDS env —
            # vendored owner-gated tools read settings.owner_ids.
            owner_env = os.getenv("OWNER_IDS") or os.getenv("OWNER_ID") or ""
            if owner_env.strip() and _ae_settings.owner_user_id in ("", "0"):
                _ae_settings.owner_user_id = owner_env

            from src.action_engine.tools.context import ToolContext  # noqa: F401
            from src.action_engine.ai import agent as _agent_mod
            from src.action_engine.tools.registry import dispatch as _real_dispatch

            worker = self

            async def _guarded_dispatch(ctx, name, args):
                """Wrapper around the vendored dispatch — enforces the
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

            # Scheduler singleton — vendored schedule tools look it up.
            try:
                from src.action_engine.core import scheduler as _sched
                if _sched.scheduler is None:
                    _sched.init(self.client)
                    asyncio.get_event_loop().create_task(_sched.scheduler.start())
            except Exception as e:
                logger.warning(f"[actions] scheduler init failed (non-fatal): {e}")

            self._ready = True
            logger.info("[actions] action worker initialised")
        except Exception as e:
            logger.warning(f"[actions] engine init failed — disabled this session: {e}")
        return self._ready

    def _agent_for(self, ctx):
        if self._agent is None:
            self._agent = self._agent_mod.Agent(ctx)
        else:
            self._agent.ctx = ctx
        return self._agent

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
    async def _classify(self, text: str):
        """ACTION route or None. Heuristic first (free), LLM arbiter only
        for genuinely ambiguous utterances."""
        from src.action_engine.ai import router
        route = router.classify(text)
        if route is None:
            route = await router.classify_llm(text)
        return route

    # ------------------------------------------------------------------ #
    async def try_handle_text(self, message, trigger_text: str) -> bool:
        """True → an action ran (caller must NOT also reply). False →
        normal reply path continues."""
        if not _ENABLED or not self._ensure_ready():
            return False
        try:
            route = await self._classify(trigger_text)
        except Exception as e:
            logger.debug(f"[actions] classify failed: {e}")
            return False
        if route.kind != "action":
            return False

        key = (str(message.channel.id), str(message.author.id))
        if key in self._inflight:
            return False
        self._inflight.add(key)
        try:
            guild = message.guild
            if guild is None:
                guilds = getattr(self.client, "guilds", [])
                guild = guilds[0] if len(guilds) == 1 else None
            ctx = self._ctx_cls(
                bot=self.client,
                guild=guild,
                current_channel_id=message.channel.id,
                author_id=message.author.id,
            )
            ctx.performed = []

            # Human-like delay — nobody executes a request in 0.0s.
            await asyncio.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))

            from src.action_engine.config.prompts import CHAT_MODE
            agent = self._agent_for(ctx)
            try:
                async with message.channel.typing():
                    reply = await asyncio.wait_for(
                        agent.run(
                            trigger_text,
                            mode=CHAT_MODE,
                            categories=route.categories or None,
                            speaker_name=message.author.display_name,
                        ),
                        timeout=_TIMEOUT,
                    )
            except asyncio.TimeoutError:
                logger.warning(f"[actions] worker timed out for: '{trigger_text[:60]}'")
                return bool(getattr(ctx, "performed", [])) or ctx.did_send
            logger.info(
                f"[actions] text request '{trigger_text[:60]}' → performed="
                f"{getattr(ctx, 'performed', [])} (agent said: {str(reply)[:60]!r} — suppressed)"
            )
            performed = getattr(ctx, "performed", [])
            return ctx.did_send or any(ok for _, ok in performed)
        except Exception as e:
            logger.error(f"[actions] worker error: {e!r}")
            return False
        finally:
            self._inflight.discard(key)

    # ------------------------------------------------------------------ #
    async def try_handle_voice(self, user_id: int, transcript: str,
                               voice_channel) -> bool:
        """Same worker, fed by a VC transcript. Never touches leave-vc —
        the pipeline's leave_cmd path already returned before this runs,
        and leave_vc_score is re-checked here as a second line of defence."""
        if not _ENABLED or not self._ensure_ready():
            return False
        try:
            from src.voice.vc_intent import leave_vc_score
            if leave_vc_score(transcript) >= 0.5:
                return False
        except Exception:
            pass
        try:
            route = await self._classify(transcript)
        except Exception as e:
            logger.debug(f"[actions] voice classify failed: {e}")
            return False
        if route.kind != "action":
            return False

        guild = getattr(voice_channel, "guild", None)
        key = (str(getattr(voice_channel, "id", 0)), str(user_id))
        if key in self._inflight:
            return False
        self._inflight.add(key)
        try:
            member = guild.get_member(user_id) if guild else None
            speaker = getattr(member, "display_name", None) or str(user_id)
            ctx = self._ctx_cls(
                bot=self.client,
                guild=guild,
                # VoiceChannel IS the messageable for VC tools (self-bot).
                current_channel_id=getattr(voice_channel, "id", None),
                author_id=user_id,
            )
            ctx.performed = []

            await asyncio.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))

            from src.action_engine.config.prompts import VOICE_MODE
            agent = self._agent_for(ctx)
            try:
                reply = await asyncio.wait_for(
                    agent.run(
                        f"[{speaker}]: {transcript}",
                        mode=VOICE_MODE,
                        categories=route.categories or None,
                    ),
                    timeout=_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[actions] voice worker timed out: '{transcript[:60]}'")
                return bool(getattr(ctx, "performed", []))
            logger.info(
                f"[actions] voice request '{transcript[:60]}' → performed="
                f"{getattr(ctx, 'performed', [])} (agent said: {str(reply)[:60]!r} — suppressed)"
            )
            performed = getattr(ctx, "performed", [])
            return ctx.did_send or any(ok for _, ok in performed)
        except Exception as e:
            logger.error(f"[actions] voice worker error: {e!r}")
            return False
        finally:
            self._inflight.discard(key)


# ---------------------------------------------------------------------- #
_worker: Optional[ActionWorker] = None


def get_action_worker(client) -> ActionWorker:
    global _worker
    if _worker is None:
        _worker = ActionWorker(client)
    return _worker
