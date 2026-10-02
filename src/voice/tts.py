"""Fish Audio streaming TTS — human-like cloned voice output.

Uses Fish Audio's /v1/tts/live WebSocket API with msgpack protocol.
Returns Ogg/Opus 48kHz mono packets ready for Discord voice send.

Requires FISH_AUDIO_API_KEY and a voice_id (from fish.audio dashboard).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import msgpack
import websockets
from loguru import logger


@dataclass
class TTSConfig:
    api_key: str
    voice_id: str
    model: str = "s2.1-pro-free"
    latency: str = "low"
    base_url: str = "wss://api.fish.audio"
    connect_timeout: float = 5.0
    temperature: float = 0.7
    top_p: float = 0.7
    speed: float = 1.0
    volume_db: float = 0.0
    chunk_length: int = 200
    repetition_penalty: float = 1.2


class FishAudioTTS:
    """Streaming TTS via Fish Audio WebSocket API."""

    def __init__(self, cfg: TTSConfig):
        self._cfg = cfg
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._packet_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self.last_error: Optional[str] = None

    async def open(self) -> None:
        """Open a TTS stream session."""
        url = f"{self._cfg.base_url.rstrip('/')}/v1/tts/live"
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "model": self._cfg.model,
        }
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, additional_headers=headers),
                timeout=self._cfg.connect_timeout,
            )
        except (asyncio.TimeoutError, OSError, websockets.WebSocketException) as e:
            raise ConnectionError(f"Fish Audio connect failed: {e}") from e

        voice_id = self._cfg.voice_id
        if not voice_id:
            raise ValueError("No voice_id configured for Fish Audio TTS")

        request_body = {
            "text": "",
            "reference_id": voice_id,
            "format": "opus",
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "repetition_penalty": self._cfg.repetition_penalty,
            "chunk_length": self._cfg.chunk_length,
            "prosody": {
                "speed": self._cfg.speed,
                "volume": self._cfg.volume_db,
            },
        }
        if self._cfg.latency != "normal":
            request_body["latency"] = self._cfg.latency

        start_payload = {"event": "start", "request": request_body}

        try:
            await self._ws.send(msgpack.packb(start_payload, use_bin_type=True))
        except Exception as e:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise ConnectionError(f"Fish Audio start failed: {e}") from e

        self._reader_task = asyncio.create_task(self._read_loop())
        logger.info(f"Fish Audio TTS stream opened (voice={voice_id})")

    async def _read_loop(self) -> None:
        assert self._ws is not None
        audio_count = 0
        try:
            async for msg in self._ws:
                if isinstance(msg, str):
                    logger.debug(f"[tts] String message: {msg[:100]}")
                    continue
                try:
                    evt = msgpack.unpackb(msg, raw=False)
                except Exception:
                    logger.warning("[tts] Failed to unpack msgpack message")
                    continue
                if not isinstance(evt, dict):
                    continue
                ev = evt.get("event")
                if ev == "audio":
                    audio = evt.get("audio")
                    if isinstance(audio, (bytes, bytearray)):
                        audio_count += 1
                        if audio_count <= 3:
                            logger.info(f"[tts] Audio packet #{audio_count}: {len(audio)} bytes")
                        await self._packet_queue.put(bytes(audio))
                    else:
                        logger.warning(f"[tts] Audio event but data is {type(audio).__name__}, not bytes")
                elif ev == "finish":
                    logger.info(f"[tts] Finish event (reason={evt.get('reason')}, audio_count={audio_count})")
                    if evt.get("reason") != "stop":
                        self.last_error = str(evt.get("message") or evt)
                        logger.error(f"Fish Audio error: {self.last_error}")
                    await self._packet_queue.put(None)
                    return
                else:
                    logger.debug(f"[tts] Unknown event: {ev} — {evt}")
        except websockets.ConnectionClosed as e:
            logger.info(f"[tts] WebSocket closed (code={e.code}, audio_count={audio_count})")
        except Exception as e:
            logger.warning(f"Fish Audio read loop error: {e!r} (audio_count={audio_count})")
        logger.info(f"[tts] Read loop ended (total audio packets: {audio_count})")
        await self._packet_queue.put(None)

    async def push_text(self, text: str) -> None:
        """Push text to be spoken. Can be called multiple times per turn."""
        if self._ws is None:
            raise RuntimeError("TTS not open; call open() first")
        # Skip empty/whitespace-only chunks (Fish Audio rejects them)
        if not text or not text.strip():
            return
        await self._ws.send(msgpack.packb({"event": "text", "text": text}, use_bin_type=True))

    async def flush(self) -> None:
        """Flush pending text to generate audio."""
        if self._ws is None:
            return
        try:
            await self._ws.send(msgpack.packb({"event": "flush"}, use_bin_type=True))
        except websockets.ConnectionClosed:
            pass

    async def end_turn(self) -> None:
        """Signal end of this speaking turn."""
        if self._ws is None:
            return
        try:
            await self._ws.send(msgpack.packb({"event": "stop"}, use_bin_type=True))
        except websockets.ConnectionClosed:
            pass

    async def packets(self) -> AsyncIterator[bytes]:
        """Yield Opus audio packets as they arrive."""
        while True:
            pkt = await self._packet_queue.get()
            if pkt is None:
                return
            yield pkt

    async def close(self) -> None:
        """Close the TTS stream."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        await self._packet_queue.put(None)

    def is_open(self) -> bool:
        return self._ws is not None and not self._ws.closed
