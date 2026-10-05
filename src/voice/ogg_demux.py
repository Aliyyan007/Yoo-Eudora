"""Streaming Ogg page parser → raw Opus audio packets.

Per RFC 3533 (Ogg) and RFC 7845 (Ogg encapsulation of Opus):
  - Each page begins with magic 'OggS', then a 27-byte header,
    then a segment table of N bytes, then N segments concatenated.
  - A packet is a sequence of consecutive segments where every segment
    is exactly 255 bytes except the last (which is < 255).
  - Packets may span pages: a final segment of 255 bytes signals
    continuation; the remainder lives in the next page.
  - The first audio stream packet is OpusHead, the second OpusTags;
    both are non-audio and must be skipped before yielding to consumers.
"""
from __future__ import annotations

from collections import deque
from typing import Iterator


class OggDemuxer:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._packet_carry = bytearray()  # leftover from a continued packet
        self._pending_audio: deque[bytes] = deque()  # extracted but not yet handed out
        self._non_audio_skipped = 0  # OpusHead + OpusTags

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._buf.extend(data)
        self._parse_pages()

    def packets(self) -> Iterator[bytes]:
        while self._pending_audio:
            yield self._pending_audio.popleft()

    def flush(self) -> Iterator[bytes]:
        # Try one more parse pass in case data was waiting.
        self._parse_pages()
        while self._pending_audio:
            yield self._pending_audio.popleft()

    def _parse_pages(self) -> None:
        while True:
            if len(self._buf) < 27:
                return
            if bytes(self._buf[:4]) != b"OggS":
                # Drop bytes until we find 'OggS' or run out
                idx = self._buf.find(b"OggS")
                if idx == -1:
                    self._buf.clear()
                    return
                del self._buf[:idx]
                if len(self._buf) < 27:
                    return

            n_segs = self._buf[26]
            header_len = 27 + n_segs
            if len(self._buf) < header_len:
                return
            seg_table = self._buf[27:27 + n_segs]
            body_len = sum(seg_table)
            page_len = header_len + body_len
            if len(self._buf) < page_len:
                return

            body = bytes(self._buf[header_len:page_len])

            # Walk segments and assemble packets
            seg_off = 0
            for seglen in seg_table:
                self._packet_carry.extend(body[seg_off:seg_off + seglen])
                seg_off += seglen
                if seglen < 255:
                    self._emit_packet(bytes(self._packet_carry))
                    self._packet_carry.clear()
            # If the page ended with a 255-segment, _packet_carry remains
            # and will continue with the next page.

            del self._buf[:page_len]

    def _emit_packet(self, packet: bytes) -> None:
        if self._non_audio_skipped < 2:
            # First two packets of an Opus stream are OpusHead + OpusTags
            self._non_audio_skipped += 1
            return
        self._pending_audio.append(packet)
