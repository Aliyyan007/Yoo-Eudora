"""Human-like timing utilities to make the bot feel natural."""
import random
import asyncio
from loguru import logger


async def human_typing_delay(reply_text: str, min_delay: float = 1.5,
                             max_delay: float = 8.0, typing_speed_cps: float = 12.0):
    """Simulate the time a human would take to type a message."""
    typing_time = len(reply_text) / typing_speed_cps
    base_delay = random.uniform(min_delay, min_delay + 2.0)
    total = base_delay + typing_time
    total = min(total, max_delay)
    total *= random.uniform(0.85, 1.15)
    total = max(total, min_delay)
    logger.debug(f"Typing delay: {total:.1f}s for {len(reply_text)} chars")
    await asyncio.sleep(total)


async def random_silence_check(silence_chance: float = 0.15) -> bool:
    """Roll the dice on whether to stay silent. Returns True if should stay silent."""
    return random.random() < silence_chance


def jittered_delay(base_minutes: float, jitter_minutes: float) -> float:
    """Returns a delay in seconds with random jitter applied."""
    jitter = random.uniform(-jitter_minutes, jitter_minutes)
    total_minutes = base_minutes + jitter
    return max(total_minutes, 0) * 60


def maybe_add_typo(text: str, typo_chance: float = 0.08) -> str:
    """Occasionally introduce a small typo to seem more human."""
    if random.random() > typo_chance or len(text) < 4:
        return text
    words = text.split()
    if not words:
        return text
    candidates = [i for i, w in enumerate(words) if len(w) >= 3]
    if not candidates:
        return text
    idx = random.choice(candidates)
    word = list(words[idx])
    if random.random() < 0.5:
        pos = random.randint(0, len(word) - 2)
        word[pos], word[pos + 1] = word[pos + 1], word[pos]
    else:
        pos = random.randint(0, len(word) - 1)
        word.insert(pos, word[pos])
    words[idx] = ''.join(word)
    return ' '.join(words)
