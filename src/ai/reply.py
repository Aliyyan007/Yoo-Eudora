"""
Reply generation, humanization post-processor, burst replies, and dedup logic.
"""
import re
import random
from collections import Counter, deque
from typing import List, Optional
from loguru import logger

from . import llm
from . import prompts
from . import d1_memory as mem_module  # D1-backed memory (falls back to JSON if D1 unavailable)


# â”€â”€ Stopwords for word analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_STOP = {
    "a", "an", "the", "i", "you", "it", "is", "to", "in", "of", "and", "or", "for",
    "my", "me", "he", "she", "we", "so", "at", "on", "if", "no", "not", "be", "do",
    "its", "im", "ur", "u", "r", "that", "this", "with", "are", "was", "have", "just",
}


def _extract_words(text: str) -> List[str]:
    """Lowercase words from text, excluding stopwords."""
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in _STOP and len(w) > 1]


def get_overused_words(recent_replies: List[str], threshold: int = 2) -> List[str]:
    """Return words that appear >= threshold times across recent replies."""
    counts = Counter()
    for r in recent_replies:
        counts.update(_extract_words(r))
    return [w for w, c in counts.items() if c >= threshold]


def _similarity_ratio(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two strings."""
    wa = set(_extract_words(a))
    wb = set(_extract_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# â”€â”€ Humanization post-processor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ABBREVS = [
    (r'\byou\b',          'u',      0.35),
    (r'\byour\b',         'ur',     0.30),
    (r'\bbecause\b',      'bc',     0.45),
    (r'\bgoing to\b',     'gonna',  0.50),
    (r'\bwant to\b',      'wanna',  0.40),
    (r'\bkind of\b',      'kinda',  0.45),
    (r'\bsort of\b',      'sorta',  0.40),
    (r'\bright now\b',    'rn',     0.50),
    (r'\bto be honest\b', 'tbh',    0.55),
    (r'\bnot gonna lie\b','ngl',    0.55),
    (r"\bi don't know\b", 'idk',    0.50),
    (r"\bI don't know\b", 'idk',    0.50),
    (r'\bfor real\b',     'fr',     0.45),
    (r'\bthough\b',       'tho',    0.40),
    (r'\bsomething\b',    'smth',   0.30),
    (r'\bprobably\b',     'prolly', 0.30),
    (r'\bhow are you\b',  'hru',    0.40),
    (r'\bwhat about you\b','wbu',   0.40),
]


def humanize(text: str) -> str:
    """
    Apply probabilistic casual transforms to make a reply feel less AI-generated.
    Not every transform fires every time, so output varies naturally.

    Based on SignalSweep's 9-gate humanization framework:
    - Burstiness: vary sentence length and structure
    - Lexical diversity: use varied vocabulary
    - Punctuation fingerprinting: avoid em-dash/semicolon abuse
    - Imperfection injection: typos, corrections, lowercase
    """
    if not text:
        return text

    r = random.random

    # 1. Standalone "I" -> "i" (60% of the time)
    if r() < 0.60:
        text = re.sub(r'\bI\b', 'i', text)

    # 2. Word contractions / abbreviations
    for pattern, replacement, chance in _ABBREVS:
        if r() < chance:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Drop trailing period (55% of the time)
    if r() < 0.55 and text.endswith('.'):
        text = text[:-1]

    # 4. Lowercase first letter (35% of the time)
    if r() < 0.35 and len(text) > 1 and text[0].isupper():
        text = text[0].lower() + text[1:]

    # 5. Occasionally append a filler (12% chance)
    if r() < 0.12 and len(text) < 80:
        text += random.choice([' lol', ' lmao', ' fr', ' ngl', ' tho', ' rn'])

    # 6. Anti-detection: Remove AI tells (em-dashes, semicolons, "moreover", etc.)
    # AI text overuses em-dashes and semicolons â€” humans rarely use them in chat
    text = text.replace('â€”', '-')   # em-dash -> hyphen
    text = text.replace('â€“', '-')   # en-dash -> hyphen
    if r() < 0.80:
        text = text.replace(';', ',')  # semicolons -> commas (80% of the time)
    # Remove AI transition words
    ai_transitions = ['moreover', 'furthermore', 'additionally', 'nevertheless',
                      'consequently', 'thus', 'hence', 'accordingly', 'indeed']
    words = text.split()
    words = [w for w in words if w.lower().strip('.,!?') not in ai_transitions]
    text = ' '.join(words)

    # 7. Imperfection injection: occasional realistic typo (8% chance, not on short msgs)
    if r() < 0.08 and len(text) > 15:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
        if candidates:
            idx = random.choice(candidates)
            word = list(words[idx])
            # Swap two adjacent characters (most common human typo)
            pos = random.randint(0, len(word) - 2)
            word[pos], word[pos + 1] = word[pos + 1], word[pos]
            words[idx] = ''.join(word)
            text = ' '.join(words)

    # 8. Occasional double space (3% chance â€” very common human typing error)
    if r() < 0.03 and ' ' in text:
        pos = text.find(' ')
        text = text[:pos] + '  ' + text[pos + 1:]

    # 9. Vary punctuation: sometimes add "..." for trailing thought (10% chance)
    if r() < 0.10 and not text.endswith(('?', '!', '...', 'ðŸ˜‚', 'ðŸ’€', 'ðŸ˜­')):
        text += '...'

    # 10. Occasionally lowercase everything (15% chance for short casual msgs)
    if r() < 0.15 and len(text) < 60:
        text = text.lower()

    return text.strip()


# â”€â”€ Spoken-language cleanup for voice / TTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Text abbreviations that look fine in a text chat but sound wrong when read
# aloud by TTS. These are expanded to full words so the synthesised voice sounds
# natural. Order matters â€” longer patterns are checked first to avoid partial
# replacements (e.g. "tbh" before "tho").
_SPEECH_ABBREVS = [
    # Greetings / check-ins
    (r'\bhru\b',           'how are you'),
    (r'\bhbu\b',           'how about you'),
    (r'\bwbu\b',           'what about you'),
    (r'\bwbu\?*',          'what about you'),
    # Common chat abbreviations
    (r'\bngl\b',           'not gonna lie'),
    (r'\btbh\b',           'to be honest'),
    (r'\bidk\b',           "i don't know"),
    (r"\bidrk\b",          "i don't really know"),
    (r'\bfr\b',            'for real'),
    (r'\bnfr\b',           'nah for real'),
    (r'\brn\b',            'right now'),
    (r'\bbc\b',            'because'),
    (r'\bimo\b',           'in my opinion'),
    (r'\bimho\b',          'in my honest opinion'),
    (r'\btysm\b',          'thank you so much'),
    (r'\btysm\b',          'thank you so much'),
    (r'\bty\b',            'thank you'),
    (r'\bnp\b',            'no problem'),
    (r'\bym\b',            'you know'),
    (r'\baf\b',            'as heck'),
    (r'\basf\b',           'as heck'),
    (r'\basfh\b',          'as heck'),
    (r'\bahh?\b',          'ah'),
    # Single-letter / ultra-short
    (r'(?<!\w)u(?!\w)',    'you'),
    (r'(?<!\w)ur(?!\w)',   'your'),
    (r'(?<!\w)ur\s+',      'your '),
    (r'(?<!\w)r(?!\w)',    'are'),
    (r'(?<!\w)n(?!\w)',    'and'),
    # Word shortenings
    (r'\bsmth\b',          'something'),
    (r'\bsm\b',            'some'),
    (r'\bprolly\b',        'probably'),
    (r'\btho\b',           'though'),
    (r'\baltho\b',         'although'),
    (r'\bpls\b',           'please'),
    (r'\bplz\b',           'please'),
    (r'\bthx\b',           'thanks'),
    (r'\btx\b',            'thanks'),
    (r'\bcuz\b',           'because'),
    (r'\bcoz\b',           'because'),
    (r'\bcos\b',           'because'),
    (r'\bomg\b',           'oh my god'),
    (r'\bomfg\b',          'oh my god'),
    (r'\blol\b',           'haha'),  # TTS reads "lol" awkwardly; "haha" sounds natural
    (r'\blmao\b',          'haha'),
    (r'\blmfao\b',         'haha'),
    (r'\brofl\b',          'haha'),
    (r'\bbrb\b',           'be right back'),
    (r'\bgtg\b',           'got to go'),
    (r'\bg2g\b',           'got to go'),
    (r'\bbtw\b',           'by the way'),
    (r'\bfyi\b',           'for your information'),
    (r'\biykyk\b',         'if you know you know'),
    (r'\bwdym\b',          'what do you mean'),
    (r'\bwdym\?*',         'what do you mean'),
    (r'\bistg\b',          "i swear to god"),
    (r'\bfrfr\b',          'for real for real'),
    (r'\bbffr\b',          'be for real'),
    (r'\bik\b',            "i know"),
    (r'\bikr\b',           "i know right"),
    (r'\birl\b',           'in real life'),
    (r'\btbh\b',           'to be honest'),
    # "w/" â†’ "with"
    (r'\bw/\b',            'with'),
    (r'\bw\/',             'with '),
    # "b/c" â†’ "because"
    (r'\bb/c\b',           'because'),
]

# Markdown / formatting artifacts that TTS can't speak â€” strip entirely
_SPEECH_STRIP_RE = re.compile(
    r'[*_~`#>|]|'                    # markdown emphasis / code / headers / quote
    r'\[[^\]]+\]\([^)]+\)|'          # markdown links [text](url) -> keep text
    r'^\s*[-*]\s+'                   # bullet points
)


def clean_for_speech(text: str) -> str:
    """Post-process LLM text so it sounds natural when read aloud by TTS.

    Expands text abbreviations into full words (e.g. "ngl" -> "not gonna lie"),
    strips markdown / emojis / formatting that a synthesiser can't speak, and
    tidies up spacing. This is the OPPOSITE of ``humanize`` (which *adds* chat
    abbreviations for text channels) â€” voice responses need full words.
    """
    if not text:
        return text

    # 0. Strip model control tokens ("<|constrain|>", "<|end_of_text|>") â€”
    # they occasionally leak from the LLM and must never reach TTS
    text = re.sub(r'<\|[^|]*\|>', '', text)

    # 1. Keep markdown link text but drop the URL
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 2. Strip markdown / formatting characters
    text = _SPEECH_STRIP_RE.sub(' ', text)

    # 3. Remove emojis (emoji + variation selectors + ZWJ sequences)
    text = re.sub(
        r'[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]'
        r'[\U0000FE00-\U0000FE0F\U0000200D]*',
        '',
        text,
    )

    # 4. Expand text abbreviations â†’ full words (case-insensitive, keep as-is)
    for pattern, replacement in _SPEECH_ABBREVS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 5. Collapse multiple spaces / whitespace left by removals
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # 6. Remove stray brackets or braces that survived
    text = re.sub(r'[{}]', '', text).strip()

    return text


def split_into_bursts(text: str) -> list:
    """
    Split a longer reply into 2-3 short burst messages for natural sending.
    Humans often send multiple short messages instead of one long one.

    Returns a list of message strings. If the text is short, returns [text].
    """
    if not text or len(text) < 60:
        return [text] if text else []

    # Try to split on sentence boundaries
    import re as _re
    sentences = _re.split(r'(?<=[.!?])\s+', text.strip())

    if len(sentences) <= 1:
        # No sentence boundaries â€” try splitting on commas or conjunctions
        parts = _re.split(r'\s+(?:but|and|so|bc|because|plus)\s+', text, flags=_re.IGNORECASE)
        if len(parts) > 1:
            sentences = parts

    if len(sentences) <= 1:
        return [text]

    # Group sentences into 2-3 bursts
    bursts = []
    if len(sentences) == 2:
        bursts = sentences
    elif len(sentences) >= 3:
        # First burst = first sentence, second = rest (or split further)
        bursts = [sentences[0], ' '.join(sentences[1:])]
        if len(bursts[1]) > 80 and len(sentences) > 2:
            bursts = [sentences[0], sentences[1], ' '.join(sentences[2:])]

    # Filter empty bursts
    bursts = [b.strip() for b in bursts if b and b.strip()]
    return bursts if bursts else [text]


# â”€â”€ Reply generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_reply(
    transcript: str,
    username: str,
    user_id: str,
    trigger_message: str,
    mood: str,
    recent_replies: list = None,
    rules_text: str = "",
    channel_style: str = "",
    my_profile_text: str = "",
    channel_lessons: list = None,
    channel_topic: str = "",
    channel_name: str = "",
    discord_topic: str = "",
    image_urls: list = None,
    my_name: str = "",
    mentioned_users: list = None,
) -> Optional[dict]:
    """
    Generate a reply using the LLM. Returns parsed JSON dict or None.
    Uses vision model if image_urls provided, smart model otherwise.
    """
    prompt = prompts.build_reply_prompt(
        transcript, username, user_id, trigger_message, mood,
        recent_replies, rules_text, channel_style, my_profile_text,
        channel_lessons, channel_topic, channel_name, discord_topic,
        my_name, mentioned_users,
    )

    if image_urls:
        raw = llm.call_vision(
            task="reply",
            system=prompts.REPLY_SYSTEM_VISION,
            user=prompt,
            image_urls=image_urls,
            temperature=0.9,
            max_tokens=1100,
            want_json=True,
        )
    else:
        # Try smart model first, fall back to fast model if it fails
        raw = llm.call_smart(
            task="reply",
            system=prompts.REPLY_SYSTEM,
            user=prompt,
            temperature=0.92,
            max_tokens=1100,
            want_json=True,
        )
        if not raw or not raw.strip():
            logger.debug("Smart model returned empty, falling back to fast model")
            raw = llm.call_fast(
                task="reply",
                system=prompts.REPLY_SYSTEM,
                user=prompt,
                temperature=0.92,
                max_tokens=450,
                want_json=True,
            )

    if not raw or not raw.strip():
        # Retry up to 2 more times with different keys
        for attempt in range(2):
            logger.warning(f"LLM returned empty response â€” retry {attempt + 1}/2")
            # Force key rotation before retry
            llm.rotate_key()
            raw = llm.call_smart(
                task="reply",
                system=prompts.REPLY_SYSTEM,
                user=prompt,
                temperature=0.92,
                max_tokens=1100,
                want_json=True,
            )
            if raw and raw.strip():
                break
            # Try fast model too
            raw = llm.call_fast(
                task="reply",
                system=prompts.REPLY_SYSTEM,
                user=prompt,
                temperature=0.92,
                max_tokens=450,
                want_json=True,
            )
            if raw and raw.strip():
                break

    if not raw or not raw.strip():
        logger.warning("LLM returned empty response after all retries")
        return None

    data = llm.parse_json_response(raw)
    if not data:
        logger.warning(f"Failed to parse LLM response as JSON: {raw[:100]}")
        # Try to extract the reply field from partial/truncated JSON
        import re as _re
        reply_match = _re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        if reply_match:
            extracted = reply_match.group(1).replace('\\n', '\n').replace('\\"', '"')
            logger.info(f"Extracted reply from partial JSON: {extracted[:60]}")
            return {"reply": extracted, "reaction": None, "burst_reply": None,
                    "new_status": None, "new_mood": None, "search_query": None}
        # Last resort: if it's just text, use it as the reply (but not if it looks like JSON)
        if raw.strip() and not raw.strip().startswith('{'):
            return {"reply": raw.strip()[:200], "reaction": None, "burst_reply": None,
                    "new_status": None, "new_mood": None, "search_query": None}
        return None

    # Strip leaked model control tokens + mass-ping tokens from text fields.
    # @everyone/@here/<@&role> are REAL pings on user accounts — the only
    # legit source is a PingController-chosen prefix, never generated text.
    for _k in ("reply", "burst_reply", "new_status"):
        if isinstance(data.get(_k), str):
            data[_k] = sanitize_mass_mentions(
                re.sub(r'<\|[^|]*\|>', '', data[_k])).strip()
    return data


def generate_reply_with_search(prompt: str, search_result: str) -> Optional[dict]:
    """Second-pass: given a search result, generate an informed casual reply."""
    augmented = (
        f"{prompt}\n\n"
        f"[WEB SEARCH RESULT â€” use this to give an accurate answer, but phrase it casually]\n"
        f"{search_result[:500]}\n"
        f"[END SEARCH RESULT]\n\n"
        f"Now write your reply knowing the above. Casual tone. Up to 50 words."
    )
    raw = llm.call_smart(
        task="reply_search",
        system=prompts.REPLY_SYSTEM,
        user=augmented,
        temperature=0.85,
        max_tokens=1100,
        want_json=True,
    )
    return llm.parse_json_response(raw)


def is_duplicate(reply_text: str, last_sent: str) -> bool:
    """Check if a reply is too similar to the last one we sent."""
    if not last_sent:
        return False
    if reply_text.lower() == last_sent.lower():
        return True
    if len(reply_text) > 4 and reply_text.lower()[:8] == last_sent.lower()[:8]:
        return True
    return _similarity_ratio(reply_text, last_sent) > 0.65


# â”€â”€ Background analysis functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_memory(transcript: str, username: str) -> list:
    """Extract facts about a user from conversation."""
    prompt = prompts.build_memory_prompt(transcript, username)
    raw = llm.call_fast(
        task="memory",
        system=prompts.MEMORY_EXTRACT_SYSTEM,
        user=prompt,
        temperature=0.3,
        max_tokens=500,
        want_json=False,
    )
    return llm.parse_json_array(raw)


def extract_channel_topic(transcript: str) -> str:
    """Summarize what a channel is about."""
    raw = llm.call_fast(
        task="topic",
        system=prompts.TOPIC_SUMMARY_SYSTEM,
        user=transcript[-1500:],
        temperature=0.3,
        max_tokens=450,
        want_json=False,
    )
    if raw:
        return raw.strip().strip('"').strip("'")
    return ""


def extract_channel_style(transcript: str) -> str:
    """Learn how people talk in a channel."""
    raw = llm.call_fast(
        task="style",
        system=prompts.STYLE_EXTRACT_SYSTEM,
        user=f"Chat transcript:\n{transcript[:2000]}",
        temperature=0.3,
        max_tokens=500,
        want_json=False,
    )
    return raw.strip() if raw else ""


def self_reflect(transcript: str) -> list:
    """Analyze own messages to extract improvement lessons."""
    raw = llm.call_fast(
        task="reflect",
        system=prompts.SELF_REFLECTION_SYSTEM,
        user=f"Chat transcript:\n{transcript[:2000]}",
        temperature=0.4,
        max_tokens=500,
        want_json=False,
    )
    return llm.parse_json_array(raw)


_MASS_MENTION_RE = re.compile(r"@everyone|@here|<@&\d+>", re.IGNORECASE)


def sanitize_mass_mentions(text: str) -> str:
    """Strip mass-ping tokens (@everyone, @here, <@&role>) from generated
    text. User accounts ping FOR REAL — no allowed_mentions guard — so the
    only legit mass ping is a PingController-chosen prefix, never model text.
    Single-user <@id> mentions are kept (normal conversation)."""
    return _MASS_MENTION_RE.sub("", text or "")


def generate_proactive_message(channel_topic: str = "", for_user: str = None) -> str:
    """Generate a proactive message to start a conversation.
    for_user: display name of the user the message will ping — keeps the
    text generic (no assumed facts) and stops the model freehand-writing
    '@name' mentions that render as garbage."""
    topic_hint = f"[channel topic: {channel_topic}]\n" if channel_topic else ""
    ping_hint = (
        f"[this message pings '{for_user}' — you know nothing personal about "
        "them: no assumed hobbies, location or facts. ask an open question or "
        "keep it about the shared space. NEVER write '@', usernames or mentions "
        "yourself — the ping is added for you.]\n"
        if for_user else
        "[NEVER write '@', usernames or mentions in the message — pings are "
        "added separately.]\n"
    )
    raw = llm.call_fast(
        task="proactive",
        system=prompts.PROACTIVE_SYSTEM,
        user=f"{topic_hint}{ping_hint}Start a casual conversation. Single short message only.",
        temperature=1.0,
        max_tokens=450,
        want_json=False,
    )
    if raw:
        text = raw.strip().strip('"').strip("'")
        # Strip model control tokens that occasionally leak ("<|constrain|>",
        # "<|end_of_text|>") â€” they render as literal garbage in the channel
        text = re.sub(r'<\|[^|]*\|>', '', text)
        # Freehand '@name' mentions render as literal garbage — pings are
        # added by the caller, drop any the model wrote anyway. Also strip
        # resolved-mention forms (<@id>, <@!id>, <@&role>) and mass-pings —
        # anything that would actually ping on a user account.
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'<@!?\d+>', '', text)
        text = sanitize_mass_mentions(text)
        text = ' '.join(text.split())  # collapse the gaps left behind
        text = text.split("\n")[0].strip()
        return text if len(text) > 2 else ""
    return ""
