"""Side-channel action worker — the vendored Engager tool loop.

The main bot keeps its own personality/reply path. When a directed message
looks like a request to DO something on Discord (send a message, ping someone,
react, change status, bump, schedule...) the classifier hands it here and the
vendored agent loop executes tools against the same discord client.

Design notes:
- Execution is serialized (one request at a time) — the vendored tools mutate
  ToolContext fields per turn and the scheduler/bump state is module-global.
- The worker's own text output is NEVER sent — the caller injects the returned
  note into the main bot's context, so the bot acknowledges in-character.
- A human-ish delay runs before the first tool call (user asked for it).
- Router runs BEFORE the agent: cheap heuristic first, LLM arbiter (~8 output
  tokens) only when ambiguous.
"""
import asyncio
import random
from typing import Optional

from loguru import logger

from .intent import looks_like_action
from .tools.context import ToolContext
from .tools.registry import dispatch as _registry_dispatch

# Tool name -> short verb for the context note (kept terse, it's for the
# main bot's prompt not for display)
_TOOL_VERBS = {
    "send_message": "sent a message",
    "send_dm": "sent them a DM",
    "send_gif": "sent a GIF",
    "send_multiple_gifs": "sent GIFs",
    "send_sticker": "sent a sticker",
    "send_multiple_stickers": "sent stickers",
    "send_vc_text": "sent a message to the vc chat",
    "delete_message": "deleted the message",
    "delete_last_message": "deleted your last message",
    "cleanup_my_messages": "cleaned up your old messages",
    "edit_message": "edited the message",
    "react_to_message": "reacted",
    "react_to_user_latest": "reacted to their message",
    "react_to_recent": "reacted to recent messages",
    "change_nickname": "changed your nickname",
    "change_status": "changed your status",
    "change_custom_status": "changed your custom status",
    "change_bio": "changed your bio",
    "use_slash_command": "ran the slash command",
    "list_slash_commands": "checked slash commands",
    "bump_with_bot": "bumped",
    "bump_all": "bumped the server",
    "find_bump_commands": "checked bump commands",
    "get_bump_status": "checked bump status",
    "schedule_message": "scheduled the message",
    "list_scheduled": "listed scheduled tasks",
    "stop_scheduled": "cancelled the scheduled task",
    "set_preference": "saved the preference",
    "get_preferences": "checked preferences",
    "delete_preference": "removed the preference",
    "list_channels": "listed channels",
    "search_channels": "searched channels",
    "get_channel_details": "got channel details",
    "get_channel_mention": "got the channel mention",
    "get_channel_link": "got the channel link",
    "search_members": "searched members",
    "get_member_details": "got member details",
    "get_member_mention": "got the member mention",
    "get_member_count": "counted members",
    "get_online_members": "checked who's online",
    "get_recent_joins": "checked recent joins",
    "get_recent_leaves": "checked recent leaves",
    "identify_bots": "identified bots",
    "search_roles": "searched roles",
    "get_role_mention": "got the role mention",
    "get_recent_messages": "read recent messages",
    "get_message_by_link": "read the linked message",
    "list_voice_channels": "listed voice channels",
    "get_voice_state": "checked voice state",
    "get_current_vc": "checked your vc",
    "get_user_voice_state": "checked their voice state",
    "get_vc_text_chat": "read the vc chat",
    "get_my_profile": "checked your profile",
    "search_gifs": "searched GIFs",
    "trending_gifs": "checked trending GIFs",
    "list_stickers": "listed stickers",
}


class ActionWorker:
    """Serial executor for natural-language Discord action requests."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def run_request(self, client, *, guild, channel, author_id,
                          author_name, text) -> Optional[str]:
        """Classify → maybe execute. Returns a context note string for the
        main bot's prompt ("[ACTION PERFORMED: ...]"), or None when this
        isn't an action request / nothing ran."""
        if not looks_like_action(text):
            return None

        # Lazy import — the vendored agent spins up the Groq pool on first use
        from .router import classify_llm, ROUTE_ACTION
        from . import agent as _agent_mod
        from .config.prompts import COMMAND_MODE

        try:
            route = await classify_llm(text)
        except Exception as e:
            logger.debug(f"[action] router failed: {e}")
            return None
        if route is None or route.kind != ROUTE_ACTION:
            return None

        ctx = ToolContext(
            bot=client,
            guild=guild,
            current_channel_id=getattr(channel, "id", None),
            author_id=author_id,
        )

        called: list[str] = []

        # Record which tools actually ran — the agent's text is suppressed
        # (the main bot acknowledges in its own voice), tool names are the
        # ground truth for the context note.
        orig_dispatch = _agent_mod.dispatch

        async def _spy_dispatch(c, name, args):
            called.append(name)
            return await orig_dispatch(c, name, args)

        async with self._lock:
            # Human-ish delay before acting — user asked for a real pause,
            # not an instant bot-speed action
            await asyncio.sleep(random.uniform(1.5, 4.0))
            _agent_mod.dispatch = _spy_dispatch
            try:
                agent = _agent_mod.Agent(ctx)
                reply_text = await agent.run(
                    text,
                    mode=COMMAND_MODE,
                    categories=route.categories or None,
                )
            except Exception as e:
                logger.warning(f"[action] worker crashed: {e}")
                return f"[ACTION FAILED — tried '{text[:60]}' but it errored; apologize briefly]"
            finally:
                _agent_mod.dispatch = orig_dispatch

        if not called:
            # Router said ACTION but no tool ran — don't claim anything
            logger.info(f"[action] router=ACTION but no tool ran for: '{text[:60]}'")
            return None

        verbs = ", ".join(dict.fromkeys(
            _TOOL_VERBS.get(n, n.replace("_", " ")) for n in called
        ))
        note = f"[ACTION PERFORMED for {author_name}: {verbs}]"
        rt = (reply_text or "").strip()
        if rt and rt not in ("(no reply)", "(reached max tool rounds without a final answer)"):
            note = note[:-1] + f" — worker detail: {rt[:140]}]"
        logger.info(f"[action] {author_name}: '{text[:60]}' -> {called}")
        return note


_worker: Optional[ActionWorker] = None


def get_action_worker() -> ActionWorker:
    global _worker
    if _worker is None:
        _worker = ActionWorker()
    return _worker
