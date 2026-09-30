"""
AI module — unified exports for the AI persona system.
"""
from . import llm
from . import prompts
from . import memory
from . import d1_memory
from . import mood
from . import search
from . import reply

# Convenience re-exports
from .mood import MoodEngine, MOODS
from .reply import (
    generate_reply,
    generate_reply_with_search,
    generate_proactive_message,
    humanize,
    is_duplicate,
    extract_memory,
    extract_channel_topic,
    extract_channel_style,
    self_reflect,
    get_overused_words,
    clean_for_speech,
)
from .d1_memory import (
    get_user_memory_text,
    update_user_memory,
    get_channel_topic,
    update_channel_topic,
    get_channel_style,
    update_channel_style,
    get_channel_lessons,
    update_channel_lessons,
)
from .search import search as web_search
from .llm import get_key_count, call_fast, call_smart
