"""
Multi-message reply module for the Eudora persona.

Features:
1. Algorithmically determine how many messages to send based on conversation nature
2. Send multiple messages with natural typing delays
3. Reply to multiple users in the same response
4. Break long responses into multiple messages (like real humans do)

Algorithmic approach:
- Excited/long conversations: 2-3 messages
- Casual chat: 1 message (default)
- Storytelling/explanations: 2-4 messages
- Quick reactions: 1 message
- Questions: 1-2 messages

The AI generates the content, this module decides how to split and deliver it.
"""
import re
import random
from typing import List, Optional, Tuple
from loguru import logger


def determine_message_count(
    text: str,
    mood: str = "chill",
    is_question: bool = False,
    is_story: bool = False,
    is_excited: bool = False,
) -> int:
    """
    Algorithmically determine how many messages to send based on conversation nature.

    Factors:
    - Text length: longer text → more messages
    - Mood: excited/hyped → more messages
    - Question: 1-2 messages
    - Story/explanation: 2-4 messages
    - Casual: 1 message (default)
    """
    # Base count
    count = 1

    # Length-based splitting
    if len(text) > 150:
        count = 2
    if len(text) > 300:
        count = 3

    # Mood-based adjustment
    if mood in ("hyped", "excited", "giddy", "playful"):
        count = max(count, random.choice([1, 2, 2]))
    elif mood in ("annoyed", "angry"):
        count = 1  # Short and sharp when annoyed
    elif mood in ("sad", "lonely"):
        count = max(count, random.choice([1, 1, 2]))

    # Story/explanation → more messages
    if is_story:
        count = max(count, random.choice([2, 3, 3]))

    # Question → 1-2 messages
    if is_question:
        count = min(count, 2)

    # Excited → more messages
    if is_excited:
        count = max(count, 2)

    # Cap at 4 (don't spam)
    count = min(count, 4)

    return count


def split_into_messages(text: str, num_messages: int) -> List[str]:
    """
    Algorithmically split a single text into multiple messages.

    Splitting strategy:
    1. Split on sentence boundaries (. ! ?)
    2. Split on natural pause points (, ; -)
    3. Split on line breaks
    4. If no natural splits, split by word count

    Returns a list of message strings.
    """
    if num_messages <= 1:
        return [text]

    # Try splitting on line breaks first
    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) >= num_messages:
        # Merge excess lines into the last message
        result = lines[:num_messages - 1]
        result.append(" ".join(lines[num_messages - 1:]))
        return result

    # Try splitting on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= num_messages:
        # Distribute sentences across messages
        result = []
        per_msg = len(sentences) // num_messages
        remainder = len(sentences) % num_messages
        idx = 0
        for i in range(num_messages):
            take = per_msg + (1 if i < remainder else 0)
            result.append(" ".join(sentences[idx:idx + take]))
            idx += take
        return result

    # Try splitting on commas/semicolons
    parts = re.split(r'(?<=[,;])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= num_messages:
        result = []
        per_msg = len(parts) // num_messages
        remainder = len(parts) % num_messages
        idx = 0
        for i in range(num_messages):
            take = per_msg + (1 if i < remainder else 0)
            result.append(" ".join(parts[idx:idx + take]))
            idx += take
        return result

    # Fallback: split by word count
    words = text.split()
    if len(words) < num_messages:
        # Not enough words to split — try character-based split
        if len(text) >= num_messages * 10:  # Only if text is long enough
            chunk_size = len(text) // num_messages
            result = []
            for i in range(num_messages):
                start = i * chunk_size
                end = start + chunk_size if i < num_messages - 1 else len(text)
                result.append(text[start:end])
            return result
        return [text]

    per_msg = len(words) // num_messages
    remainder = len(words) % num_messages
    result = []
    idx = 0
    for i in range(num_messages):
        take = per_msg + (1 if i < remainder else 0)
        result.append(" ".join(words[idx:idx + take]))
        idx += take

    return result


def detect_conversation_nature(text: str, trigger: str = "") -> dict:
    """
    Algorithmically detect the nature of a conversation to inform message count.

    Returns a dict with:
    - is_question: bool
    - is_story: bool
    - is_excited: bool
    - is_explanation: bool
    - is_reaction: bool
    """
    combined = (text + " " + trigger).lower()

    is_question = "?" in combined or any(
        w in combined for w in ["what", "why", "how", "when", "where", "who", "which", "can u", "could u", "do u"]
    )

    is_story = any(
        w in combined for w in ["so i", "today i", "yesterday i", "this happened", "story", "let me tell", "basically"]
    ) or len(combined) > 200

    is_excited = any(
        w in combined for w in ["omg", "no way", "fr??", "fr?", "really??", "actually??", "wait what", "brooo", "yooo"]
    )

    is_explanation = any(
        w in combined for w in ["because", "the reason", "so basically", "it's because", "that's because", "well,"]
    )

    is_reaction = any(
        w in combined for w in ["lol", "lmao", "💀", "fr", "true", "based", "real", "crazy", "wild", "damn"]
    ) and len(combined) < 50

    return {
        "is_question": is_question,
        "is_story": is_story,
        "is_excited": is_excited,
        "is_explanation": is_explanation,
        "is_reaction": is_reaction,
    }


def get_typing_delay(text: str) -> float:
    """
    Algorithmically determine typing delay based on message length.
    Simulates human typing speed (~12 chars/second with some variance).
    """
    base = len(text) / 12.0  # 12 chars per second
    variance = random.uniform(0.5, 1.5)
    delay = base * variance
    # Cap between 1-8 seconds
    return max(1.0, min(delay, 8.0))


def get_inter_message_delay() -> float:
    """
    Get the delay between multiple messages (simulates human pause between texts).
    """
    return random.uniform(1.5, 4.0)
