"""Multi-key Groq client pool with automatic rotation.

Groq rate-limits are per-ACCOUNT (not per-key), so the bot is configured with
several keys from separate accounts. This pool:

- round-robins across keys for each request,
- on a 429 / rate-limit error, marks the key "cooling" and retries the next,
- on a 5xx, retries with backoff,
- exposes a single `chat()` entrypoint used everywhere.

The pool is async-safe (an asyncio.Lock guards cooldown mutations).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import AsyncGroq
from groq import BadRequestError as GroqBadRequestError
from groq import NotFoundError as GroqNotFoundError
from groq import RateLimitError as GroqRateLimitError
from loguru import logger

from src.actions.config.settings import settings


@dataclass
class _KeyState:
    key: str
    client: AsyncGroq
    cooling_until: float = 0.0
    errors: int = 0


@dataclass
class GroqPool:
    keys: list[str] = field(default_factory=list)
    _states: list[_KeyState] = field(default_factory=list)
    _idx: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _vision_dead: bool = False   # vision model 404'd once — skip future calls

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = list(settings.groq_keys)
        if not self.keys:
            raise RuntimeError(
                "No GROQ_API_KEY* found in .env — add at least one Groq key."
            )
        self._states = [_KeyState(k, AsyncGroq(api_key=k)) for k in self.keys]
        logger.info(f"GroqPool initialised with {len(self._states)} key(s).")

    # ------------------------------------------------------------------ #
    def _next_ready(self) -> Optional[_KeyState]:
        """Return the next non-cooling key (round-robin), or None if all cool."""
        now = time.monotonic()
        n = len(self._states)
        for _ in range(n):
            st = self._states[self._idx % n]
            self._idx = (self._idx + 1) % n
            if st.cooling_until <= now:
                return st
        return None

    async def _mark_cooling(self, st: _KeyState, seconds: float) -> None:
        async with self._lock:
            st.cooling_until = time.monotonic() + seconds
            st.errors += 1
            logger.warning(
                f"Groq key #{self._states.index(st)+1} cooling {seconds:.0f}s "
                f"(total errors: {st.errors})."
            )

    # ------------------------------------------------------------------ #
    async def chat(
        self,
        *,
        model: Optional[str] = None,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: Any = "auto",
        temperature: float = 0.4,
        max_tokens: int = 2048,
        max_retries: int = 6,
        extra: dict | None = None,
    ) -> Any:
        """Make a chat completion call, rotating keys on rate-limit/errors."""
        model = model or settings.groq_model_text
        messages = messages or []
        extra = extra or {}
        last_err: Exception | None = None

        for attempt in range(max_retries):
            st = self._next_ready()
            if st is None:
                # All keys cooling — wait for the soonest one.
                wait = min(s.cooling_until for s in self._states) - time.monotonic()
                wait = max(wait, 1.0)
                logger.warning(f"All Groq keys cooling — waiting {wait:.0f}s.")
                await asyncio.sleep(wait)
                continue

            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                if tools:
                    kwargs["tool_choice"] = tool_choice
                resp = await st.client.chat.completions.create(**kwargs)
                # reset error counter on success
                st.errors = 0
                return resp
            except GroqRateLimitError as e:
                # Try to read retry-after from the error body/headers.
                retry_after = getattr(e, "retry_after", None)
                try:
                    retry_after = float(retry_after)
                except (TypeError, ValueError):
                    retry_after = 30.0
                await self._mark_cooling(st, retry_after)
                last_err = e
                continue
            except GroqBadRequestError:
                # 400s (bad tool args, malformed payload) — retrying the same
                # request on another key will never succeed. Fail fast.
                raise
            except GroqNotFoundError:
                # 404 model_not_found — rotating keys won't resurrect a
                # model that doesn't exist on this tier. Fail fast.
                raise
            except Exception as e:  # noqa: BLE001
                # 5xx / transient — short backoff, stay on rotation.
                logger.error(f"Groq call failed ({type(e).__name__}): {e}")
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 20))
                continue

        raise RuntimeError(f"Groq chat failed after {max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    async def chat_stream(
        self,
        *,
        model: Optional[str] = None,
        messages: list[dict] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 400,
        max_retries: int = 4,
        extra: dict | None = None,
    ):
        """Streaming variant of chat() for the voice path.

        Async-generator yielding content-delta strings. Key rotation happens
        only until the first token arrives — mid-stream failures abort (the
        caller re-invokes if it wants a retry).
        """
        model = model or settings.groq_model_text
        messages = messages or []
        extra = extra or {}
        last_err: Exception | None = None

        for attempt in range(max_retries):
            st = self._next_ready()
            if st is None:
                wait = min(s.cooling_until for s in self._states) - time.monotonic()
                wait = max(wait, 1.0)
                logger.warning(f"All Groq keys cooling — waiting {wait:.0f}s.")
                await asyncio.sleep(wait)
                continue
            try:
                stream = await st.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **extra,
                )
                st.errors = 0
                first = True
                # inactivity guard — a stalled stream (seen: 26s silence
                # on a live VC turn) must not hold the turn forever.
                # Pre-first-token stalls retry on the next key; mid-stream
                # stalls just end the turn gracefully.
                it = stream.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            it.__anext__(), timeout=8.0)
                    except asyncio.TimeoutError:
                        if first:
                            raise RuntimeError("stream stalled before first token")
                        break  # partial content is better than none
                    except StopAsyncIteration:
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        first = False
                        yield delta.content
                if first:
                    last_err = RuntimeError("stream produced no content")
                    continue
                break
            except GroqRateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                try:
                    retry_after = float(retry_after)
                except (TypeError, ValueError):
                    retry_after = 30.0
                await self._mark_cooling(st, retry_after)
                last_err = e
                continue
            except GroqBadRequestError:
                raise
            except GroqNotFoundError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"Groq stream failed ({type(e).__name__}): {e}")
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 10))
                continue
        else:
            raise RuntimeError(f"Groq stream failed: {last_err}")

    # ------------------------------------------------------------------ #
    async def call(
        self,
        fn,
        *,
        max_retries: int = 4,
    ) -> Any:
        """Run ``fn(client)`` (an arbitrary AsyncGroq API call) with the same
        key-rotation / cooldown behaviour as :meth:`chat`. Used by voice STT/TTS.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries):
            st = self._next_ready()
            if st is None:
                wait = min(s.cooling_until for s in self._states) - time.monotonic()
                wait = max(wait, 1.0)
                logger.warning(f"All Groq keys cooling — waiting {wait:.0f}s.")
                await asyncio.sleep(wait)
                continue
            try:
                resp = await fn(st.client)
                st.errors = 0
                return resp
            except GroqRateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                try:
                    retry_after = float(retry_after)
                except (TypeError, ValueError):
                    retry_after = 30.0
                await self._mark_cooling(st, retry_after)
                last_err = e
                continue
            except GroqBadRequestError:
                raise  # non-retryable — another key produces the same 400
            except GroqNotFoundError:
                raise  # 404 model_not_found — same on every key
            except Exception as e:  # noqa: BLE001
                logger.error(f"Groq call failed ({type(e).__name__}): {e}")
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 15))
                continue
        raise RuntimeError(f"Groq call failed after {max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    async def transcribe_image(self, image_url: str, prompt: str) -> str:
        """Use the vision model to describe an image (attachment/gif).

        Returns a short textual description suitable for the agent's context.
        Falls back to a URL-based note if the vision model is unavailable.
        """
        if self._vision_dead:
            return f"[image sent: {image_url}]"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        try:
            resp = await self.chat(
                model=settings.groq_model_vision,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            if not resp.choices or not resp.choices[0].message:
                return f"[image sent: {image_url} — vision returned empty]"
            return (resp.choices[0].message.content or "").strip() or f"[image sent: {image_url}]"
        except GroqNotFoundError:
            # the vision model isn't on this tier — remember it so we stop
            # burning a call per image forever
            self._vision_dead = True
            logger.warning(
                "Vision model unavailable on this Groq tier — disabling "
                "image descriptions for this process.")
            return f"[image sent: {image_url} — vision unavailable]"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Vision describe failed for {image_url}: {e}")
            # Fallback: return a note with the URL so the agent at least knows
            # an image was sent and can reference it.
            return f"[image sent: {image_url} — vision unavailable]"


# Module-level singleton, lazily created.
_pool: GroqPool | None = None


def get_pool() -> GroqPool:
    global _pool
    if _pool is None:
        _pool = GroqPool()
    return _pool
