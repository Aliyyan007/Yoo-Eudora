"""AI auto-correction of ASR transcripts.

After transcription, the text may contain errors (wrong words, phonetic
confusions). This module uses a fast LLM call to correct obvious errors
while preserving the user's intent.

Examples:
  "I'm going to the bark" → "I'm going to the park"
  "What do you mean by that" → "What do you mean by that" (no change)
  "COMPUTED ON PRETENDANCE" → "Can you sit on the bench?" (context-based)

The correction is optional and fast — uses gpt-oss-20b with reasoning_effort=low
and very low max_tokens (50). If the LLM call fails or times out, the original
transcript is used unchanged.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

from loguru import logger


# System prompt for transcript correction
_CORRECTION_SYSTEM = """You correct speech recognition transcript errors. The input is an ASR transcript that may have wrong words due to phonetic confusion. Fix obvious errors to match what the person most likely said. Rules:
- Keep it the SAME length and meaning — don't add or remove information
- Only fix obvious word errors (e.g. "bark" → "park", "pre-tend-ance" → "pretentious")
- Don't change correct text — if it's already fine, return it unchanged
- Preserve the speaker's intent and tone
- Return ONLY the corrected text, nothing else
- If the transcript is garbage/unintelligible, return it unchanged
- Don't add punctuation if there is none, don't remove it if there is"""

# Don't correct very short transcripts (1-2 words) — too little context
_MIN_WORDS_TO_CORRECT = 3

# Max time to wait for correction
_CORRECTION_TIMEOUT_S = 8.0


async def auto_correct_transcript(transcript: str) -> str:
    """Correct ASR errors in a transcript using a fast LLM call.

    Args:
        transcript: The raw ASR transcript text

    Returns:
        Corrected transcript, or original if correction fails/times out
    """
    if not transcript or len(transcript.strip()) < 2:
        return transcript

    word_count = len(transcript.strip().split())
    if word_count < _MIN_WORDS_TO_CORRECT:
        return transcript

    # Skip correction for ALL-CAPS text from streaming ASR — the Groq Whisper
    # API two-pass will replace it with proper-case text anyway, so correcting
    # the uppercase version is a waste of time.
    if transcript.isupper():
        return transcript

    try:
        from ..ai import llm

        # Use call_voice (gpt-oss-20b, reasoning_effort=low) for fast correction
        corrected = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.call_voice(
                    "transcript_correction",
                    _CORRECTION_SYSTEM,
                    f"Correct this speech transcript: \"{transcript}\"",
                    want_json=False,
                    max_tokens=80,  # transcripts are short
                    temperature=0.1,  # low temp for accuracy
                )
            ),
            timeout=_CORRECTION_TIMEOUT_S,
        )

        if corrected and corrected.strip():
            # Clean up the response — remove quotes, extra whitespace
            corrected = corrected.strip().strip('"').strip("'").strip()
            # Remove any "Corrected:" prefix if the LLM added one
            corrected = re.sub(r'^(corrected|fixed|result):\s*', '', corrected, flags=re.IGNORECASE)

            if corrected and len(corrected) > 0:
                # Sanity check: don't accept corrections that are wildly different
                # (much longer or shorter than original)
                orig_len = len(transcript.strip())
                corr_len = len(corrected)
                if corr_len > orig_len * 2 or corr_len < orig_len * 0.3:
                    logger.debug(f"[correct] Rejected (length mismatch): '{corrected}' vs '{transcript}'")
                    return transcript

                if corrected != transcript:
                    logger.info(f"[correct] '{transcript}' → '{corrected}'")
                return corrected

    except asyncio.TimeoutError:
        logger.debug(f"[correct] Timed out, using original: '{transcript[:50]}'")
    except Exception as e:
        logger.debug(f"[correct] Failed ({e}), using original: '{transcript[:50]}'")

    return transcript


def _looks_clean(text: str) -> bool:
    """Check if a transcript looks clean enough to skip correction.
    A transcript is 'clean' if it has proper capitalization and no
    obvious ASR artifacts."""
    # If it has mixed case (not all upper), it's probably from Whisper
    # and already decent quality
    has_upper = any(c.isupper() for c in text)
    has_lower = any(c.islower() for c in text)
    return has_upper and has_lower
