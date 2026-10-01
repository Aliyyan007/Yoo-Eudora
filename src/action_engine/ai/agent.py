"""Central dynamic AI agent.

This is the brain of Engager Bot. It uses Groq's native tool/function-calling
to turn a natural-language message into a sequence of tool calls (channel
search, member lookup, send message, react, join voice, ...) until the
request is satisfied, then returns a final natural-language reply.

Three modes:
  - CHAT_MODE:    someone mentioned/replied to/named the bot in a channel.
                  Uses the human-like CHAT_SYSTEM_PROMPT. Prioritises natural
                  conversation. Has access to tools but doesn't need them for
                  casual chat.
  - COMMAND_MODE: the owner issued an explicit instruction (command channel
                  or DM). Uses COMMAND_SYSTEM_PROMPT. Prioritises task
                  execution.
  - EVENT_MODE:   a Discord event happened. The agent decides autonomously
                  whether to act or reply NOACTION.

The loop:
  1. Compose messages = [system, ...history, ...context, user_request].
  2. Call Groq with the tool schemas.
  3. If the model returns tool_calls, dispatch each one, append the tool
     results as "tool" role messages, and loop again.
  4. If the model returns a plain content reply, that's the final answer.
  5. Stop after MAX_TOOL_ROUNDS to avoid infinite loops.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.action_engine.config.prompts import (
    CHAT_MODE,
    COMMAND_MODE,
    EVENT_MODE,
    VOICE_MODE,
    EVENT_PREAMBLE,
    EVENT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT_PLAIN,
    system_prompt_for,
)
from src.action_engine.config.settings import settings
from src.action_engine.core.groq_pool import get_pool
from src.action_engine.tools.context import ToolContext
from src.action_engine.tools.registry import build_tools, dispatch
from src.action_engine.utils.logger import logger


class Agent:
    """Stateful wrapper around the Groq tool-calling loop."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.pool = get_pool()
        self.max_rounds = settings.max_tool_rounds
        # Tools are built per-mode (chat mode uses a reduced set to stay
        # under free-tier TPM limits). We cache the full set here and
        # build the chat subset on demand.
        self._tools_full = build_tools("command")
        self._tools_chat = build_tools("chat")
        # Chat mode: max 5 tool rounds (vs 8 for command mode) to save tokens
        # but allow multi-action requests like "send 5 gifs and 5 stickers".
        self._chat_max_rounds = 5

    # ------------------------------------------------------------------ #
    async def run(
        self,
        user_request: str,
        *,
        mode: str = COMMAND_MODE,
        history: list[dict] | None = None,
        context: str | None = None,
        speaker_name: str | None = None,
        categories: set | None = None,
        no_tools: bool = False,
    ) -> str:
        """Process a request and return the agent's final natural-language reply.

        Args:
            user_request: The natural-language message from the user, OR
                (in event_mode) a description of the Discord event.
            mode: One of CHAT_MODE, COMMAND_MODE, EVENT_MODE.
            history: Optional prior conversation turns (list of role/content
                dicts) to give the agent continuity.
            context: Optional extra context string (e.g. recent channel
                messages) injected before the user request.
            speaker_name: The display name of whoever is talking to the bot
                (used in chat mode so the agent knows who it's talking to).
        """
        # Tool-free runs get a tool-free prompt — mentioning tools makes the
        # model hallucinate calls and burn retries on tool_use_failed errors.
        sys_prompt = (
            CHAT_SYSTEM_PROMPT_PLAIN
            if (no_tools and mode == CHAT_MODE)
            else system_prompt_for(mode)
        )
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]

        # Inject conversation history (prior turns in this channel).
        if history:
            messages.extend(history)

        # Inject context (recent messages, etc.) as a system note.
        if context:
            messages.append({
                "role": "system",
                "content": f"Here are the recent messages in this channel for your context:\n{context}",
            })

        # Build the user message.
        if mode == EVENT_MODE:
            user_content = f"{EVENT_PREAMBLE}\n\nEVENT: {user_request}"
        elif mode == CHAT_MODE and speaker_name:
            user_content = f"[{speaker_name}]: {user_request}"
        else:
            user_content = user_request

        messages.append({"role": "user", "content": user_content})

        # Chat mode: higher temperature for more natural/varied responses.
        # Command/event mode: lower temperature for reliability.
        temperature = 0.7 if mode in (CHAT_MODE, VOICE_MODE) else 0.4
        # gpt-oss-* are reasoning models — token budget includes hidden
        # "thinking" tokens. Voice was truncated at 150 (empty content ->
        # "(no reply)" literal). 400 + low reasoning effort keeps it snappy.
        max_tokens = 400 if mode == VOICE_MODE else (600 if mode == CHAT_MODE else 800)

        # Select tools + model per mode.
        # - Voice CONVERSATION (no_tools): pure text reply, no schemas ->
        #   ~4x fewer prompt tokens and zero hallucinated tool calls.
        # - Voice/CHAT ACTION (categories): filtered schema subset, only the
        #   tools the task needs + resolvers.
        # - Command/event mode: full set.
        extra: dict = {}
        if mode == VOICE_MODE and no_tools:
            tools = None
            model = settings.groq_model_chat
            max_rounds = 1
            extra = {"reasoning_effort": "low"}
        elif mode == VOICE_MODE:
            tools = build_tools("command", categories=categories) if categories \
                else self._tools_chat
            model = settings.groq_model_chat
            max_rounds = 3
            extra = {"reasoning_effort": "low"}
        elif mode == CHAT_MODE and no_tools:
            tools = None
            model = settings.groq_model_chat
            max_rounds = 1
            extra = {"reasoning_effort": "low"}
        elif categories:
            tools = build_tools("command", categories=categories)
            model = settings.groq_model_chat
            max_rounds = 6
            extra = {"reasoning_effort": "low"}
        elif mode == CHAT_MODE:
            tools = self._tools_chat
            model = settings.groq_model_chat
            max_rounds = self._chat_max_rounds
            extra = {"reasoning_effort": "low"}
        else:
            tools = self._tools_full
            model = settings.groq_model_text
            max_rounds = self.max_rounds

        for round_idx in range(1, max_rounds + 1):
            logger.debug(
                f"Agent round {round_idx}/{max_rounds} ({mode}): {len(messages)} msgs, "
                f"{len(tools) if tools else 0} tools."
            )
            try:
                resp = await self.pool.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto" if tools else "none",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                )
            except Exception as e:  # noqa: BLE001
                # Rescue: Groq 400 'tool_use_failed' means the model emitted a
                # malformed tool call (hallucinated name like
                # 'search_channels<|channel|>commentary' or null for a string
                # param). Parse the failed generation, sanitize, run the tool
                # ourselves and keep the loop alive instead of dying.
                if await self._rescue_failed_tool_call(e, messages, round_idx):
                    continue
                # If Groq fails (e.g. tool call validation error), try one
                # more round without tools to get a text reply.
                logger.warning(f"Groq call failed in round {round_idx}: {e}")
                if round_idx >= max_rounds:
                    return "" if mode == VOICE_MODE else "(sorry, i had trouble processing that)"
                # Retry without tools to force a text response.
                try:
                    resp = await self.pool.chat(
                        model=model,
                        messages=messages,
                        tools=None,
                        tool_choice="none",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    final = (resp.choices[0].message.content or "").strip()
                    if mode == VOICE_MODE:
                        final = re.sub(r"^\[[^\]]{1,40}\]:\s*", "", final).strip()
                    if final:
                        logger.info(f"Agent fallback reply ({mode}): {final[:200]}")
                        return final
                except Exception:
                    pass
                return "" if mode == VOICE_MODE else "(sorry, i had trouble processing that)"
            # Log token usage for monitoring.
            if hasattr(resp, "usage") and resp.usage:
                u = resp.usage
                logger.info(
                    f"Tokens ({mode} r{round_idx}): "
                    f"prompt={u.prompt_tokens} completion={u.completion_tokens} "
                    f"total={u.total_tokens}"
                )
            if not getattr(resp, "choices", None):
                logger.warning(f"Groq returned no choices in round {round_idx}")
                return "" if mode == VOICE_MODE else "(sorry, i had trouble processing that)"
            choice = resp.choices[0]
            msg = choice.message

            # Append the assistant message (may contain content + tool_calls).
            assistant_entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_entry["content"] = msg.content
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            # No tool calls -> final answer.
            if not msg.tool_calls:
                final = (msg.content or "").strip()
                # gpt-oss sometimes leaks <|endoftext|> + hallucinated junk
                # after the real answer — truncate at the first special token.
                final = re.split(r"<\|[^|]*\|>", final, maxsplit=1)[0].strip()
                # The model sometimes parrots the "[Name]:" transcript format
                # back as a literal prefix (or even impersonates the speaker).
                if mode == VOICE_MODE:
                    final = re.sub(r"^\[[^\]]{1,40}\]:\s*", "", final).strip()
                logger.info(f"Agent final reply ({mode}): {final[:200]}")
                if mode == VOICE_MODE and not final:
                    return ""  # silence — session speaks nothing
                return final if final else "(no reply)"

            # Execute each tool call and feed results back.
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    args = {}
                    logger.warning(f"Bad tool args for {name}: {e}")
                logger.info(f"Tool call ({mode}): {name}({args})")
                result = await dispatch(self.ctx, name, args)
                logger.debug(f"Tool {name} result: {result[:300]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    }
                )

        return "" if mode == VOICE_MODE else "(reached max tool rounds without a final answer)"

    # ------------------------------------------------------------------ #
    async def _rescue_failed_tool_call(self, e: Exception, messages: list, round_idx: int) -> bool:
        """Recover from Groq's 'tool_use_failed' 400s.

        The model emitted a tool call Groq rejected (name not in tools —
        e.g. 'search_channels<|channel|>commentary', or a null arg where the
        schema wants a string). The error body's failed_generation still
        contains what the model MEANT — parse it, sanitize, dispatch it
        ourselves, append the result, and let the loop continue.
        Returns True if a tool call was recovered and executed.
        """
        try:
            body = getattr(e, "body", None)
            failed = None
            if isinstance(body, dict):
                failed = (body.get("error") or {}).get("failed_generation")
            if not failed:
                import re as _re
                m = _re.search(r'"failed_generation"\s*:\s*"((?:[^"\\]|\\.)*)"', str(e))
                if m:
                    failed = m.group(1).encode().decode("unicode_escape")
            if not failed:
                return False
            gen = json.loads(failed) if isinstance(failed, str) else failed
            name = str(gen.get("name") or "")
            args = gen.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            # Sanitize: strip hallucinated suffixes + null/empty args.
            name = name.split("<")[0].split("|")[0].strip()
            args = {k: v for k, v in args.items() if k and v not in ("", None)}
            if not name:
                return False
            tc_id = f"rescued_{round_idx}_{int(time.time())}"
            logger.info(f"Rescued malformed tool call -> {name}({args})")
            messages.append({"role": "assistant", "tool_calls": [{
                "id": tc_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }]})
            result = await dispatch(self.ctx, name, args)
            messages.append({
                "role": "tool", "tool_call_id": tc_id,
                "name": name, "content": result,
            })
            return True
        except Exception as e2:  # noqa: BLE001
            logger.debug(f"tool_use_failed rescue failed: {e2}")
            return False

    # ------------------------------------------------------------------ #
    async def run_stream(
        self,
        user_request: str,
        *,
        context: str | None = None,
    ):
        """Streaming no-tools voice reply — yields text deltas.

        Overlapping LLM generation with TTS synthesis and playback is how
        production voice stacks (Pipecat/LiveKit) cut first-audio latency.
        Only valid for the single-round no-tools path (tool calls need the
        full response before dispatching).
        """
        messages: list[dict] = [
            {"role": "system", "content": system_prompt_for(VOICE_MODE)}
        ]
        if context:
            messages.append({
                "role": "system",
                "content": f"Here are the recent messages in this channel for your context:\n{context}",
            })
        messages.append({"role": "user", "content": user_request})
        t0 = time.monotonic()
        n = 0
        async for delta in self.pool.chat_stream(
            model=settings.groq_model_chat,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
            extra={"reasoning_effort": "low"},
        ):
            n += len(delta)
            yield delta
        logger.info(
            f"Voice stream done ({time.monotonic() - t0:.2f}s, {n} chars)"
        )
