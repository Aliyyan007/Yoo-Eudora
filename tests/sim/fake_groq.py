"""Scriptable fake for the Groq API — covers BOTH client shapes:

* ``llm.py`` uses the sync ``Groq`` client:
    ``client.chat.completions.create(messages=..., ...)`` -> resp.choices[0].message.content
    with ``stream=True`` -> iterator of chunks (``chunk.choices[0].delta.content``).

* ``action_engine.core.groq_pool`` uses ``AsyncGroq`` via ``pool.chat(...)``:
    awaited create() -> resp.choices[0].message (.content, .tool_calls) + resp.usage.

The responder is programmable per test: ``llm.on_request`` is a
``fn(system, user, kwargs) -> str | None`` — return None to fall through to
the built-in default responder, which discriminates on prompt content and
produces persona-shaped replies so the whole real pipeline (prompt build,
JSON parse, sanitization, humanize) is exercised.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Response shapes (duck-typed Groq objects)
# ─────────────────────────────────────────────────────────────────────────────

class _ToolCall:
    def __init__(self, name: str, arguments: dict | str, call_id: str = None):
        self.id = call_id or f"call_{id(self) & 0xffff:x}"
        self.type = "function"
        self.function = SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        )


class _Resp:
    def __init__(self, content: str = None, tool_calls: list = None):
        msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        self.choices = [SimpleNamespace(message=msg, delta=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _StreamChunk:
    def __init__(self, text: str):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=text, reasoning=None))]


# ─────────────────────────────────────────────────────────────────────────────
# The scripted brain
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedLLM:
    """One fake "model" behind every client — records all calls, answers
    prompts using rules the test can override.

    Attributes:
        calls: list of every invocation {system, user, model, kwargs}
        on_request: optional fn(system, user, kwargs) -> str|None — first
            non-None result wins; use it to script specific replies.
        reply_text / reply_fn: what the reply generator should say
            (reply_fn gets the extracted trigger message text).
        verdict: YES/NO for confirm-arbiter calls (action/info confirms,
            intent LLM checks).
        agent_planner: fn(request_text, tools) -> (content, tool_calls)
            used by the action-engine pool — decide tools to run.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.on_request: Optional[Callable] = None
        self.reply_text: str = "lol fr"
        self.reply_fn: Optional[Callable[[str, str], Any]] = None
        self.verdict: str = "YES"
        self.verdict_fn: Optional[Callable[[str, str], str]] = None
        self.agent_planner: Optional[Callable[[str, list], tuple]] = None
        self.agent_text: str = "on it"
        self.proactive_text: str = "yo anyone up"
        self.voice_text: str = "yeah honestly that's fair"
        self.memory_facts: list = []
        self.topic_text: str = "general chat and banter"
        self.style_text: str = "- lowercase, casual\n- short messages"
        self.fail_next: int = 0          # N calls to fail (empty response)
        self.rate_limit: bool = False    # raise 429s forever
        self._reply_n: int = 0           # rotates the default-reply pool so
        # two replies in a conversation never echo word-for-word (the real
        # client's duplicate-suppression drops verbatim repeats)

    # ── call recording ────────────────────────────────────────────────
    def record(self, model, system, user, kwargs):
        self.calls.append({"model": model, "system": system, "user": user,
                           "kwargs": kwargs})

    # ── the default responder ─────────────────────────────────────────
    def respond(self, system: str, user: str, kwargs: dict) -> str:
        """Heuristic default — good enough for the majority of prompts.
        Tests override with on_request for specific behaviors."""
        if self.on_request is not None:
            out = self.on_request(system, user, kwargs)
            if out is not None:
                return out

        sysl = (system or "").lower()
        user_l = (user or "").lower()

        # YES/NO arbiters (action confirm, info confirm, vc-intent, router)
        if "only yes or no" in sysl or "answer only yes or no" in sysl:
            if self.verdict_fn:
                return self.verdict_fn(system, user)
            return self.verdict

        # Reply JSON — system contains the OUTPUT JSON contract
        if '"reply"' in sysl and "output json" in sysl:
            trigger = self._extract_trigger(user)
            reply = self._make_reply(trigger, user_l)
            return json.dumps({
                "reply": reply,
                "burst_reply": None,
                "reaction": None,
                "new_status": None,
                "new_mood": None,
                "emotion_intensity": 2,
                "search_query": None,
            })

        # Intent detection system
        if "directed_at_bot" in sysl:
            return json.dumps({
                "directed_at_bot": True, "confidence": 0.8,
                "reason": "scripted", "is_question": "?" in (user or ""),
                "is_greeting": False, "is_relatable": False,
                "should_reply": True,
            })

        # JSON array prompts (memory extraction, self-reflection)
        if "json array" in sysl:
            return json.dumps(self.memory_facts)

        # Vision-style reply (image context) — plain short reaction
        if "shared image" in sysl or "meme" in sysl:
            return json.dumps({"reply": "💀", "burst_reply": None,
                               "reaction": None, "new_status": None,
                               "new_mood": None, "search_query": None})

        # Voice call system — full words, no abbreviations
        if "voice call" in sysl or "spoken aloud" in sysl:
            return self.voice_text

        # Proactive system — plain casual line
        if "break the silence" in sysl or "real person would say" in sysl:
            return self.proactive_text

        # Style analyst
        if "linguistic analyst" in sysl or "style guide" in sysl:
            return self.style_text

        # Topic summariser
        if "summarising" in sysl or "channel topic" in sysl:
            return self.topic_text

        # Second-pass reply-with-search
        if "web search result" in user_l:
            return json.dumps({"reply": "oh apparently it's a thing", "burst_reply": None,
                               "reaction": None, "new_status": None,
                               "new_mood": None, "search_query": None})

        # Fallback — short ack
        return "ok"

    @staticmethod
    def _extract_trigger(user: str) -> str:
        """Pull the '[message you must respond to] name: text' line."""
        m = re.search(r"\[message you must respond to\]\s*[^:]+:\s*(.+)", user or "")
        return m.group(1).strip() if m else ""

    def _make_reply(self, trigger: str, user_l: str) -> str:
        if self.reply_fn:
            out = self.reply_fn(trigger, user_l)
            if out is not None:
                return out
        t = (trigger or "").lower()
        # A small behavioral model — human-ish canned logic so tests see
        # plausible context-sensitive output rather than one canned line.
        if t in ("hi", "hey", "hello", "yo", "sup", "heya", "hiya"):
            return "yo"
        if "how are you" in t or t in ("hru", "hru?"):
            return "i'm good hbu"
        if "your name" in t or "who are you" in t:
            return "eudora lol"
        if "what's my name" in t or "whats my name" in t:
            return "hmm idk actually"
        if "wyd" in t or "what are you doing" in t or "what are u doing" in t:
            return "just sketching rn wbu"
        if t.endswith("?"):
            pool = ["honestly no clue lol", "good question tbh",
                    "hmm probably", "not sure rn", "idk maybe"]
            self._reply_n += 1
            return pool[self._reply_n % len(pool)]
        pool = ["lol fr", "nah real", "honestly same", "true tho",
                "yeah fair enough", "lmao facts", "fr", "no way"]
        self._reply_n += 1
        return pool[self._reply_n % len(pool)] if self.reply_text == "lol fr" \
            else self.reply_text


# ─────────────────────────────────────────────────────────────────────────────
# Sync client (llm.py — Groq)
# ─────────────────────────────────────────────────────────────────────────────

class _SyncCompletions:
    def __init__(self, brain: ScriptedLLM, api_key: str):
        self._brain = brain
        self._key = api_key

    def create(self, messages=None, model=None, stream=False, **kwargs):
        system = next((m["content"] for m in (messages or []) if m.get("role") == "system"), "")
        user = " ".join(
            (m["content"] if isinstance(m["content"], str)
             else json.dumps(m["content"]))
            for m in (messages or []) if m.get("role") == "user")
        self._brain.record(model, system, user, kwargs)

        if self._brain.rate_limit:
            raise Exception("429 rate limit exceeded — Please try again in 60s")
        if self._brain.fail_next > 0:
            self._brain.fail_next -= 1
            return _Resp(content="")

        text = self._brain.respond(system, user, kwargs)
        if stream:
            pieces = re.findall(r"\S+\s*|\s+", text) or [text]

            def _gen() -> Iterator[_StreamChunk]:
                for p in pieces:
                    yield _StreamChunk(p)
            return _gen()
        return _Resp(content=text)


class FakeGroqClient:
    """Duck-typed ``groq.Groq`` — ``client.api_key`` + ``client.chat.completions.create``."""

    def __init__(self, api_key: str, brain: ScriptedLLM):
        self.api_key = api_key
        self._brain = brain
        self.chat = SimpleNamespace(completions=_SyncCompletions(brain, api_key))


# ─────────────────────────────────────────────────────────────────────────────
# Async pool (action_engine.groq_pool)
# ─────────────────────────────────────────────────────────────────────────────

class FakeGroqPool:
    """Duck-typed ``GroqPool``: async ``chat``/``chat_stream``/``call``/
    ``transcribe_image`` driven by the same ScriptedLLM brain.

    ``agent_planner(system_user_text, tools) -> (content, [(name, args)])``
    decides whether the agent calls tools; default: no tools, plain text.
    """

    def __init__(self, brain: ScriptedLLM):
        self._brain = brain
        self.calls: list[dict] = []

    async def chat(self, *, model=None, messages=None, tools=None,
                   tool_choice="auto", temperature=0.4, max_tokens=2048,
                   max_retries=6, extra=None):
        messages = messages or []
        system = " ".join(m["content"] for m in messages if m.get("role") == "system")
        user = " ".join(
            (m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]))
            for m in messages if m.get("role") in ("user", "tool"))
        self.calls.append({"model": model, "system": system, "user": user,
                           "tools": [t["function"]["name"] for t in (tools or [])
                                     if isinstance(t, dict) and "function" in t]})
        self._brain.record(model, system, user, {})

        if self._brain.rate_limit:
            raise Exception("429 rate limit")
        if self._brain.fail_next > 0:
            self._brain.fail_next -= 1
            return _Resp(content="")

        # YES/NO arbiters first (they're system prompts w/o tools)
        sysl = system.lower()
        if tools is None and ("only yes or no" in sysl or "answer only yes or no" in sysl):
            if self._brain.verdict_fn:
                return _Resp(content=self._brain.verdict_fn(system, user))
            return _Resp(content=self._brain.verdict)

        if self._brain.agent_planner:
            content, planned = self._brain.agent_planner(user, tools or [])
            tcs = [_ToolCall(n, a) for n, a in (planned or [])]
            return _Resp(content=content, tool_calls=tcs)
        return _Resp(content=self._brain.agent_text)

    async def chat_stream(self, *, model=None, messages=None,
                          temperature=0.4, max_tokens=400,
                          max_retries=4, extra=None):
        messages = messages or []
        system = " ".join(m["content"] for m in messages if m.get("role") == "system")
        user = " ".join(
            (m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]))
            for m in messages if m.get("role") == "user")
        self._brain.record(model, system, user, {})
        text = self._brain.respond(system, user, {})
        for piece in re.findall(r"\S+\s*|\s+", text) or [text]:
            yield piece

    async def call(self, fn, *, max_retries: int = 4):
        client = FakeGroqClient("fake-async-key", self._brain)
        return await fn(client)

    async def transcribe_image(self, image_url: str, prompt: str) -> str:
        return f"[image: {image_url}]"


def install_llm_fakes(brain: ScriptedLLM, n_keys: int = 3):
    """Point src.ai.llm at fakes. Returns the fake client list."""
    from src.ai import llm
    clients = [FakeGroqClient(f"fake-key-{i}", brain) for i in range(n_keys)]
    llm._groq_clients = clients
    llm._groq_cooldowns = {}
    llm._groq_current_key_idx = 0
    return clients


def install_pool_fake(brain: ScriptedLLM):
    """Patch every get_pool() binding so the action engine uses FakeGroqPool."""
    pool = FakeGroqPool(brain)
    import src.action_engine.core.groq_pool as gp
    gp._pool = pool

    original_get_pool = gp.get_pool

    def _get_pool():
        return pool

    gp.get_pool = _get_pool
    # Rebind names imported elsewhere at module load
    for mod_name in (
        "src.action_engine.ai.agent",
        "src.action_engine.ai.router",
        "src.ai.action_bridge",
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            if hasattr(mod, "get_pool"):
                setattr(mod, "get_pool", _get_pool)
        except Exception:
            pass
    return pool, original_get_pool
