"""
Unified LLM caller with multi-key Groq rotation.
Cycles through multiple API keys, rotates on 429s, persists cooldowns.
Ensures 99.9% uptime even under heavy usage.
"""
import os
import re
import json
import time
import threading
from typing import Optional, List
from loguru import logger
from groq import Groq
from dotenv import load_dotenv

# Load .env from config directory
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    ".env",
)
load_dotenv(_ENV_PATH)

# ── Key management ────────────────────────────────────────────────────────────

def _load_keys() -> List[str]:
    """Load all Groq API keys from environment (GROQ_API_KEY, GROQ_API_KEY_2, ...)."""
    keys = []
    # Primary key
    primary = os.getenv("GROQ_API_KEY")
    if primary and primary.strip():
        keys.append(primary.strip())
    # Additional keys
    for i in range(2, 20):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k and k.strip() and k.strip() not in keys:
            keys.append(k.strip())
    return keys


_groq_keys = _load_keys()
_groq_clients = [Groq(api_key=k) for k in _groq_keys] if _groq_keys else []
_groq_cooldowns: dict = {}  # api_key -> cooldown_until timestamp
_groq_current_key_idx = 0
_llm_lock = threading.Lock()
_MAX_COOLDOWN_WAIT_S = 90

# Cooldown persistence
_COOLDOWN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "groq_cooldowns.json",
)


def _load_cooldowns() -> dict:
    """Load persisted cooldowns from disk."""
    if os.path.exists(_COOLDOWN_FILE):
        try:
            with open(_COOLDOWN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                now = time.time()
                return {k: v for k, v in data.items() if v > now}
        except Exception:
            pass
    return {}


def _save_cooldowns(cooldowns: dict):
    """Persist cooldowns to disk so they survive restarts."""
    try:
        os.makedirs(os.path.dirname(_COOLDOWN_FILE), exist_ok=True)
        with open(_COOLDOWN_FILE, "w", encoding="utf-8") as f:
            json.dump(cooldowns, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to save cooldowns: {e}")


_groq_cooldowns = _load_cooldowns()


def _parse_groq_wait_time(err_msg: str) -> int:
    """Extract wait seconds from Groq error like 'Please try again in 1h23m4s'."""
    match = re.search(r"try again in ([\dhms.]+)", err_msg)
    if not match:
        return 0
    time_str = match.group(1)
    total = 0
    if time_str.endswith('s') and time_str[:-1].replace('.', '', 1).isdigit():
        if 'm' not in time_str and 'h' not in time_str:
            return int(float(time_str[:-1]))
    parts = re.findall(r"(\d+)([hms])", time_str)
    for val, unit in parts:
        val = int(val)
        if unit == 'h':
            total += val * 3600
        elif unit == 'm':
            total += val * 60
        elif unit == 's':
            total += val
    return total


def get_key_count() -> int:
    """Return number of loaded Groq keys."""
    return len(_groq_clients)


def rotate_key():
    """Force rotation to the next Groq key (bypass current one)."""
    global _groq_current_key_idx
    with _llm_lock:
        n = len(_groq_clients)
        if n > 0:
            _groq_current_key_idx = (_groq_current_key_idx + 1) % n
            logger.debug(f"Force-rotated to key {_groq_current_key_idx + 1}/{n}")


def _call_llm(
    task: str,
    system: str,
    user: str,
    temperature: float = 0.9,
    max_tokens: int = 800,
    image_urls: Optional[List[str]] = None,
    want_json: bool = True,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_wait_s: Optional[float] = None,
) -> str:
    """
    Try Groq keys one-at-a-time, rotating on 429s.
    If ALL keys are cooling, waits for the soonest one.
    Returns the response text, or empty string on failure.

    Cooldowns are tracked per (key, model) — Groq rate limits are per model,
    so a key throttled on one model can still serve another (this is what lets
    call_voice's cross-model fallback work under rate-limit pressure).

    The Groq HTTP call runs OUTSIDE _llm_lock — the lock only guards key
    selection and cooldown bookkeeping. Without this, every LLM call in the
    process serialises and a voice reply can sit behind a slow text reply.

    Args:
        reasoning_effort: For reasoning models (gpt-oss), set to "low" to
            minimize reasoning token consumption. None = use model default.
        max_wait_s: Cap on how long to wait for a free key. None = default
            (_MAX_COOLDOWN_WAIT_S). Real-time callers (voice) pass a small
            bound so a saturated key pool fails fast instead of stalling.
    """
    if not _groq_clients:
        logger.warning(f"[{task}] No Groq keys configured")
        return ""

    if want_json and "json" not in system.lower():
        system += "\n\nIMPORTANT: You must reply in valid JSON format."

    # Default models
    if model is None:
        if image_urls:
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            model = "openai/gpt-oss-20b"

    global _groq_current_key_idx
    n = len(_groq_clients)
    wait_budget = max_wait_s if max_wait_s is not None else _MAX_COOLDOWN_WAIT_S
    deadline = time.time() + wait_budget

    while True:
        now = time.time()
        if now > deadline:
            logger.warning(f"[{task}] Gave up waiting for a free key after {wait_budget:.0f}s")
            return ""

        # Pick the next key that isn't cooling for THIS model. The lock is
        # only held for the pick — not for the HTTP call below.
        client = None
        client_idx = -1
        wait_s = 0.0
        with _llm_lock:
            for _ in range(n):
                idx = _groq_current_key_idx % n
                candidate = _groq_clients[idx]

                cooldown_until = _groq_cooldowns.get(f"{candidate.api_key}|{model}", 0)
                if cooldown_until > now:
                    remaining = int(cooldown_until - now)
                    logger.debug(f"[{task}] Key {idx+1} cooling on {model} ({remaining}s left), trying next")
                    _groq_current_key_idx = (idx + 1) % n
                    continue

                client = candidate
                client_idx = idx
                # Advance past the picked key — round-robin spreads load across
                # keys and concurrent callers naturally land on different keys.
                _groq_current_key_idx = (idx + 1) % n
                break

            if client is None:
                soonest = min(_groq_cooldowns.get(f"{c.api_key}|{model}", 0) for c in _groq_clients)
                wait_s = max(soonest - time.time(), 0) + 1
                wait_s = min(wait_s, max(deadline - time.time(), 0))

        if client is None:
            # Everything is cooling on this model — wait outside the lock so
            # calls for other models aren't blocked.
            if wait_s <= 0:
                logger.warning(f"[{task}] No time left, giving up")
                return ""
            logger.info(f"[{task}] All keys cooling on {model} — waiting {wait_s:.0f}s for soonest key...")
            time.sleep(wait_s)
            continue

        try:
            # Don't use response_format — gpt-oss models fail with it.
            # Instead, we parse JSON from the text response.
            # Build kwargs — only add reasoning_effort for reasoning models
            create_kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            # gpt-oss models support reasoning_effort to control thinking tokens
            if reasoning_effort and "gpt-oss" in model:
                create_kwargs["reasoning_effort"] = reasoning_effort

            if image_urls:
                content = [{"type": "text", "text": user}]
                for url in image_urls[:2]:
                    content.append({"type": "image_url", "image_url": {"url": url}})
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    **create_kwargs,
                )
            else:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **create_kwargs,
                )
            logger.debug(f"[{task}] Using key {client_idx+1}/{n} (model: {model})")
            return resp.choices[0].message.content

        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower() or "exceeded" in err.lower():
                is_tpd = "TPD" in err or "tokens per day" in err.lower()
                parsed = _parse_groq_wait_time(err)
                wait_time = parsed + 5 if parsed > 0 else (14400 if is_tpd else 60)
                # Cap cooldown at 24 hours max to prevent runaway values
                wait_time = min(wait_time, 86400)
                with _llm_lock:
                    _groq_cooldowns[f"{client.api_key}|{model}"] = time.time() + wait_time
                    _save_cooldowns(_groq_cooldowns)
                limit_type = "TPD" if is_tpd else "RPM/RPD"
                logger.warning(f"[{task}] Key {client_idx+1} {limit_type} limited on {model}. Rotating. Cool {wait_time}s.")
            elif "404" in err or "model_not_found" in err or "does not exist" in err:
                # Model not found — don't try other keys, they'll all fail
                logger.error(f"[{task}] Model not found: {model}")
                return ""
            else:
                logger.warning(f"[{task}] Groq call failed: {e}")
            # Loop back — the rotation index already moved past this key, so
            # the next iteration tries a different key (or waits if all cool).


# ── Convenience wrappers ──────────────────────────────────────────────────────

def call_smart(task: str, system: str, user: str, **kwargs) -> str:
    """Call with the smart model (gpt-oss-20b). For complex reply generation.
    Free tier: 200K TPD, 1K RPD, 8K TPM per key.
    Uses reasoning_effort='low' by default to minimize thinking tokens."""
    if 'reasoning_effort' not in kwargs:
        kwargs['reasoning_effort'] = 'low'
    return _call_llm(task, system, user, model="openai/gpt-oss-20b", **kwargs)


def call_fast(task: str, system: str, user: str, **kwargs) -> str:
    """Call with gpt-oss-20b but smaller max_tokens for simple tasks:
    greetings, background analysis, memory extraction, topic/style learning.
    With 10 keys: 2M TPD, 10K RPD — more than enough capacity."""
    # Default to smaller max_tokens for simple tasks (saves tokens)
    # NOTE: gpt-oss-20b is a reasoning model — it uses ~200 tokens for
    # internal reasoning before generating the response. max_tokens must
    # be high enough to cover both reasoning AND the actual response.
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = 500
    if 'reasoning_effort' not in kwargs:
        kwargs['reasoning_effort'] = 'low'
    return _call_llm(task, system, user, model="openai/gpt-oss-20b", **kwargs)


# Voice reply model chain. Groq's free/dev tier retired every non-reasoning
# text model in 2026 (llama-3.1-8b-instant, llama-3.3-70b-versatile,
# llama-4-scout, qwen3-32b, groq/compound — all shut down or enterprise-gated),
# so gpt-oss-20b (~940 tok/s, Groq's fastest text model) is the primary and
# gpt-oss-120b (~500 tok/s) the fallback on a separate rate-limit pool.
_VOICE_PRIMARY_MODEL = "openai/gpt-oss-20b"
_VOICE_FALLBACK_MODEL = "openai/gpt-oss-120b"
_VOICE_MAX_WAIT_S = 12.0  # voice is real-time — never sit in the full cooldown queue


def call_voice(task: str, system: str, user: str, **kwargs) -> str:
    """Call for voice replies — optimized for fast, token-efficient responses.

    Uses gpt-oss-20b with reasoning_effort='low' to minimize thinking tokens.
    If the primary fails (rate-limited/error), falls back to gpt-oss-120b —
    per-model cooldowns mean keys throttled on 20b can still serve 120b, so
    the fallback usually engages instantly instead of waiting out a cooldown.
    max_wait_s is bounded so a saturated key pool fails fast and the caller's
    own fallback can kick in, rather than stalling a live voice turn for 90s.

    Note: gpt-oss models count reasoning tokens against max_tokens, so the cap
    must cover a low-effort think (~50-150 tokens) PLUS the reply. 240 leaves
    comfortable headroom for a 1-3 sentence reply while still chopping
    monologues that shouldn't be spoken aloud anyway.
    """
    if 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = 340  # ~150 reasoning headroom + ~190 reply tokens
    if 'temperature' not in kwargs:
        kwargs['temperature'] = 0.8
    if 'reasoning_effort' not in kwargs:
        kwargs['reasoning_effort'] = 'low'
    kwargs.setdefault('max_wait_s', _VOICE_MAX_WAIT_S)
    model = kwargs.pop('model', None) or _VOICE_PRIMARY_MODEL
    resp = _call_llm(task, system, user, model=model, **kwargs)
    if not resp and model != _VOICE_FALLBACK_MODEL:
        logger.debug(f"[{task}] {model} returned empty — falling back to {_VOICE_FALLBACK_MODEL}")
        resp = _call_llm(task, system, user, model=_VOICE_FALLBACK_MODEL, **kwargs)
    return resp


def _stream_deltas(stream):
    """Yield content deltas from a Groq streaming response.

    gpt-oss models emit reasoning tokens on delta.reasoning — those are
    skipped; only delta.content (the actual reply) is yielded.
    """
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield piece
    except Exception as e:
        logger.debug(f"[voice] Stream read error: {e}")


def call_voice_stream(task: str, system: str, user: str, **kwargs):
    """Streaming variant of call_voice — returns a sync iterator of text
    deltas, or None if no stream could be started (caller falls back to
    call_voice / call_smart for the full response).

    Tries each non-cooling key once per model (primary → fallback); never
    waits — voice is real-time so a saturated pool must fail fast.
    """
    if not _groq_clients:
        return None

    max_tokens = kwargs.pop('max_tokens', 340)
    temperature = kwargs.pop('temperature', 0.8)
    reasoning_effort = kwargs.pop('reasoning_effort', 'low')

    global _groq_current_key_idx
    n = len(_groq_clients)

    for model in (_VOICE_PRIMARY_MODEL, _VOICE_FALLBACK_MODEL):
        tried = set()
        while len(tried) < n:
            client = None
            now = time.time()
            with _llm_lock:
                for _ in range(n):
                    idx = _groq_current_key_idx % n
                    candidate = _groq_clients[idx]
                    _groq_current_key_idx = (idx + 1) % n
                    if candidate.api_key in tried:
                        continue
                    if _groq_cooldowns.get(f"{candidate.api_key}|{model}", 0) > now:
                        continue
                    tried.add(candidate.api_key)
                    client = candidate
                    break
            if client is None:
                break  # every key cooling on this model — try the next model

            create_kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if reasoning_effort and "gpt-oss" in model:
                create_kwargs["reasoning_effort"] = reasoning_effort

            try:
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **create_kwargs,
                )
                logger.debug(f"[{task}] Streaming voice reply (model: {model})")
                return _stream_deltas(stream)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate limit" in err.lower() or "exceeded" in err.lower():
                    is_tpd = "TPD" in err or "tokens per day" in err.lower()
                    parsed = _parse_groq_wait_time(err)
                    wait_time = parsed + 5 if parsed > 0 else (14400 if is_tpd else 60)
                    wait_time = min(wait_time, 86400)
                    with _llm_lock:
                        _groq_cooldowns[f"{client.api_key}|{model}"] = time.time() + wait_time
                        _save_cooldowns(_groq_cooldowns)
                elif "404" in err or "model_not_found" in err or "does not exist" in err:
                    break  # model dead on all keys — try next model
                # else transient — try the next key
    return None


def call_vision(task: str, system: str, user: str, image_urls: list, **kwargs) -> str:
    """Call with the vision model for image/meme reactions."""
    return _call_llm(task, system, user, image_urls=image_urls,
                     model="meta-llama/llama-4-scout-17b-16e-instruct", **kwargs)


def parse_json_response(raw: str) -> Optional[dict]:
    """Safely parse a JSON response from the LLM. Handles truncated JSON."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try to fix truncated JSON by adding closing braces/quotes
        if '{' in text:
            snippet = text[text.index('{'):]
            # Count unmatched braces
            opens = snippet.count('{')
            closes = snippet.count('}')
            # Try adding missing closing braces
            attempt = snippet + ('}' * (opens - closes))
            # Also try closing any unclosed string
            if attempt.count('"') % 2 != 0:
                attempt += '"}'
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                # Try a more aggressive fix: close the last string and braces
                attempt2 = snippet
                if attempt2.count('"') % 2 != 0:
                    attempt2 += '"'
                attempt2 += '}' * (opens - attempt2.count('}'))
                try:
                    return json.loads(attempt2)
                except json.JSONDecodeError:
                    pass
    return None


def parse_json_array(raw: str) -> list:
    """Safely parse a JSON array response from the LLM."""
    if not raw or not raw.strip():
        return []
    try:
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


# ── Native function calling (tool use) ─────────────────────────────────────────

def call_with_tools(
    task: str,
    system: str,
    user: str,
    tools: list,
    tool_choice: str = "auto",
    model: str = "openai/gpt-oss-120b",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    previous_messages: list = None,
) -> dict:
    """
    Call Groq with native function calling support.

    Uses the same multi-key rotation as _call_llm.
    Returns: {"content": str|None, "tool_calls": list|None, "role": "assistant", "finish_reason": str}
    Returns {"content": "", "tool_calls": None, "role": "assistant", "finish_reason": "error"} on failure.
    """
    if not _groq_clients:
        logger.warning(f"[{task}] No Groq keys configured")
        return {"content": "", "tool_calls": None, "role": "assistant", "finish_reason": "error"}

    # Build the messages list: system + previous_messages (if any) + user
    messages = [{"role": "system", "content": system}]
    if previous_messages:
        messages.extend(previous_messages)
    messages.append({"role": "user", "content": user})

    global _groq_current_key_idx
    n = len(_groq_clients)

    with _llm_lock:
        deadline = time.time() + _MAX_COOLDOWN_WAIT_S

        while True:
            now = time.time()
            if now > deadline:
                logger.warning(f"[{task}] Gave up waiting for a free key after {_MAX_COOLDOWN_WAIT_S}s")
                return {"content": "", "tool_calls": None, "role": "assistant", "finish_reason": "error"}

            found_available = False
            for _ in range(n):
                idx = _groq_current_key_idx % n
                client = _groq_clients[idx]

                # Per-(key, model) cooldowns — same scheme as _call_llm
                cooldown_until = _groq_cooldowns.get(f"{client.api_key}|{model}", 0)
                if cooldown_until > now:
                    remaining = int(cooldown_until - now)
                    logger.debug(f"[{task}] Key {idx+1} cooling on {model} ({remaining}s left), trying next")
                    _groq_current_key_idx = (idx + 1) % n
                    continue

                found_available = True

                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    logger.debug(f"[{task}] Using key {idx+1}/{n} (model: {model})")
                    msg = resp.choices[0].message
                    # Serialize tool_calls to a plain dict format
                    if msg.tool_calls:
                        serialized_tool_calls = [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                                "type": "function",
                            }
                            for tc in msg.tool_calls
                        ]
                    else:
                        serialized_tool_calls = None
                    return {
                        "content": msg.content,
                        "tool_calls": serialized_tool_calls,
                        "role": "assistant",
                        "finish_reason": resp.choices[0].finish_reason,
                    }

                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate limit" in err.lower() or "exceeded" in err.lower():
                        is_tpd = "TPD" in err or "tokens per day" in err.lower()
                        parsed = _parse_groq_wait_time(err)
                        wait_time = parsed + 5 if parsed > 0 else (14400 if is_tpd else 60)
                        # Cap cooldown at 24 hours max to prevent runaway values
                        wait_time = min(wait_time, 86400)
                        _groq_cooldowns[f"{client.api_key}|{model}"] = time.time() + wait_time
                        _save_cooldowns(_groq_cooldowns)
                        _groq_current_key_idx = (idx + 1) % n
                        limit_type = "TPD" if is_tpd else "RPM/RPD"
                        logger.warning(f"[{task}] Key {idx+1} {limit_type} limited on {model}. Rotating. Cool {wait_time}s.")
                    else:
                        logger.warning(f"[{task}] Groq call failed: {e}")
                        return {"content": "", "tool_calls": None, "role": "assistant", "finish_reason": "error"}
                    _groq_current_key_idx = (idx + 1) % n
                    break

            if not found_available:
                soonest = min(_groq_cooldowns.get(f"{c.api_key}|{model}", 0) for c in _groq_clients)
                wait_s = max(soonest - time.time(), 0) + 1
                wait_s = min(wait_s, max(deadline - time.time(), 0))
                if wait_s <= 0:
                    logger.warning(f"[{task}] No time left, giving up")
                    return {"content": "", "tool_calls": None, "role": "assistant", "finish_reason": "error"}
                logger.info(f"[{task}] All keys cooling — waiting {wait_s:.0f}s for soonest key...")
                time.sleep(wait_s)
