"""
Context module — builds message history windows for AI processing.
Formats recent Discord messages into a string the LLM can understand.
"""
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Optional


def format_message_history(messages: List[dict], max_messages: int = 15) -> str:
    """
    Format a list of message dicts into a readable conversation string.

    Each message dict should have:
        - author: str (username)
        - content: str (message text)
        - timestamp: datetime or str
        - is_me: bool (whether this message was sent by the bot)
        - reply_to: Optional[str] (who this message replies to)

    Returns a formatted string like:
        [12:05] Alice: hey everyone
        [12:06] Bob: hru?
        [12:06] (me): pretty good wbu
    """
    if not messages:
        return "(no recent messages)"

    # Take the last N messages
    recent = messages[-max_messages:]

    lines = []
    for msg in recent:
        author = msg.get("author", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp")
        is_me = msg.get("is_me", False)
        reply_to = msg.get("reply_to")

        # Format timestamp
        if isinstance(timestamp, datetime):
            time_str = timestamp.strftime("%H:%M")
        elif isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = "??"

        # Format author name
        display_name = "(me)" if is_me else author

        # Add reply indicator
        reply_indicator = ""
        if reply_to:
            reply_indicator = f" (replying to {reply_to}) "

        line = f"[{time_str}] {display_name}:{reply_indicator} {content}"
        lines.append(line)

    return "\n".join(lines)


def message_to_context_entry(message, bot_user_id: int) -> dict:
    """
    Convert a discord.py-self message object to a context dict.
    """
    is_me = message.author.id == bot_user_id
    author_name = message.author.display_name or message.author.name

    reply_to = None
    if hasattr(message, 'reference') and message.reference:
        if hasattr(message.reference, 'resolved') and message.reference.resolved:
            reply_to = message.reference.resolved.author.display_name or message.reference.resolved.author.name
        elif hasattr(message.reference, 'cached_message') and message.reference.cached_message:
            reply_to = message.reference.cached_message.author.display_name or message.reference.cached_message.author.name

    return {
        "author": author_name,
        "content": message.content,
        "timestamp": message.created_at,
        "is_me": is_me,
        "reply_to": reply_to,
    }


def is_old_message(message, max_age_minutes: int = 30) -> bool:
    """Check if a message is too old to warrant a reply."""
    if not hasattr(message, 'created_at'):
        return False

    now = datetime.now(message.created_at.tzinfo) if message.created_at.tzinfo else datetime.now()
    age = now - message.created_at
    return age > timedelta(minutes=max_age_minutes)
