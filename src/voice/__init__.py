"""
Voice module — real-time voice conversation for the AI persona.

Pipeline: discord-native-voice receive → Silero VAD → sherpa-onnx ASR → Groq LLM → Fish Audio TTS → discord voice send

Components:
- vad.py: Silero VAD for voice activity detection
- asr.py: sherpa-onnx streaming ASR for speech-to-text
- tts.py: Fish Audio streaming TTS for human-like voice output
- pipeline.py: orchestrates the full listen → think → speak pipeline
- manager.py: manages voice connections, auto-join/leave, silence detection
"""
from .manager import VoiceManager

__all__ = ["VoiceManager"]
