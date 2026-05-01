"""Unified audio pipeline — orchestrates capture, VAD, AEC, and playback.

All components run in-process. No TCP connections, no Wyoming protocol.
State machine: IDLE -> LISTENING -> PROCESSING -> RESPONDING -> IDLE
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Any, Callable, Awaitable

import numpy as np

from chaosvector_audio.capture import AudioChunk, CaptureConfig, CaptureManager
from chaosvector_audio.playback import PlaybackConfig, PlaybackManager, PlaybackPriority
from chaosvector_audio.aec import AECConfig, EchoCanceller
from chaosvector_audio.vad import VADConfig, VoiceActivityDetector, SpeechState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state machine
# ---------------------------------------------------------------------------

class PipelineState(Enum):
    IDLE = auto()         # Mic streaming, waiting for wake word
    LISTENING = auto()    # Wake word detected, collecting utterance
    PROCESSING = auto()   # Utterance complete, running STT + intent
    RESPONDING = auto()   # Playing TTS response


class AudioPipeline:
    """Single-process audio pipeline replacing the Wyoming TCP stack.

    Lifecycle:
        pipeline = AudioPipeline(...)
        await pipeline.start()
        ...
        await pipeline.stop()

    Integration points (set via register_*):
        - wake_detector:  (AudioChunk) -> bool
        - stt_handler:    (list[AudioChunk]) -> str
        - intent_handler: (str) -> str | bytes   (text or audio response)
        - tts_handler:    (str) -> np.ndarray     (text -> int16 audio)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        pre_roll_ms: int = 500,
        vad_aggressiveness: int = 2,
    ) -> None:
        self._state = PipelineState.IDLE

        # Sub-components
        self._capture = CaptureManager(CaptureConfig(
            device=device,
            sample_rate=sample_rate,
            channels=channels,
            pre_roll_ms=pre_roll_ms,
        ))
        self._playback = PlaybackManager(PlaybackConfig())
        self._aec = EchoCanceller(AECConfig())
        self._vad = VoiceActivityDetector(VADConfig(
            aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
        ))

        # Wire AEC reference from playback
        self._playback.set_reference_callback(self._aec.feed_reference)

        # Integration callbacks (set by host application)
        self._wake_detector: Callable[[AudioChunk], bool] | None = None
        self._stt_handler: Callable[[list[AudioChunk]], Awaitable[str]] | None = None
        self._intent_handler: Callable[[str], Awaitable[str | bytes]] | None = None
        self._tts_handler: Callable[[str], Awaitable[np.ndarray]] | None = None

        self._task: asyncio.Task | None = None

    # -- registration --------------------------------------------------------

    def register_wake_detector(self, fn: Callable[[AudioChunk], bool]) -> None:
        self._wake_detector = fn

    def register_stt(self, fn: Callable[[list[AudioChunk]], Awaitable[str]]) -> None:
        self._stt_handler = fn

    def register_intent(self, fn: Callable[[str], Awaitable[str | bytes]]) -> None:
        self._intent_handler = fn

    def register_tts(self, fn: Callable[[str], Awaitable[np.ndarray]]) -> None:
        self._tts_handler = fn

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self._capture.open()
        await self._playback.start()
        self._task = asyncio.create_task(self._run())
        log.info("pipeline started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._playback.stop()
        await self._capture.close()
        self._set_state(PipelineState.IDLE)
        log.info("pipeline stopped")

    # -- state management ----------------------------------------------------

    def _set_state(self, new: PipelineState) -> None:
        if new != self._state:
            log.info("state: %s -> %s", self._state.name, new.name)
            self._state = new

    @property
    def state(self) -> PipelineState:
        return self._state

    # -- main loop -----------------------------------------------------------

    async def _run(self) -> None:
        """Core pipeline loop — runs until cancelled."""
        async for chunk in self._capture.chunks():
            try:
                await self._process_chunk(chunk)
            except Exception:
                log.exception("error processing audio chunk")

    async def _process_chunk(self, chunk: AudioChunk) -> None:
        # Apply AEC to capture frame
        processed_samples = self._aec.process_capture_frame(chunk.samples)
        chunk = AudioChunk(
            samples=processed_samples,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            timestamp=chunk.timestamp,
            rms=chunk.rms,
        )

        if self._state == PipelineState.IDLE:
            await self._handle_idle(chunk)
        elif self._state == PipelineState.LISTENING:
            await self._handle_listening(chunk)
        # PROCESSING and RESPONDING are handled by their respective tasks

    # -- state handlers ------------------------------------------------------

    async def _handle_idle(self, chunk: AudioChunk) -> None:
        """IDLE: stream audio through wake word detector."""
        if self._wake_detector is None:
            return

        if self._aec.should_suppress_stt():
            return  # Don't trigger wake word during echo tail

        if self._wake_detector(chunk):
            log.info("wake word detected")
            self._set_state(PipelineState.LISTENING)

            # Barge-in: stop any current playback
            self._playback.barge_in()
            self._playback.duck()

            # Play wake beep
            beep = _generate_beep(frequency=880, duration_ms=80, sample_rate=22050)
            await self._playback.enqueue(
                beep, sample_rate=22050, priority=PlaybackPriority.WAKE_BEEP, label="wake-beep"
            )

            # Grab pre-roll audio
            self._utterance_chunks: list[AudioChunk] = self._capture.drain_pre_roll()
            self._vad.reset()

    async def _handle_listening(self, chunk: AudioChunk) -> None:
        """LISTENING: collect utterance frames until end-of-speech."""
        self._utterance_chunks.append(chunk)

        _, end_of_speech = self._vad.process_frame(chunk.samples)

        if end_of_speech:
            log.info("end of speech — %d chunks collected", len(self._utterance_chunks))
            self._set_state(PipelineState.PROCESSING)
            self._playback.unduck()
            # Kick off processing as a separate task so the capture loop continues
            asyncio.create_task(self._process_utterance(self._utterance_chunks))
            self._utterance_chunks = []

    async def _process_utterance(self, chunks: list[AudioChunk]) -> None:
        """PROCESSING: run STT then intent, then respond."""
        try:
            # -- STT --
            transcript = ""
            if self._stt_handler is not None:
                transcript = await self._stt_handler(chunks)
                log.info("STT result: %s", transcript)
            else:
                log.warning("no STT handler registered")
                self._set_state(PipelineState.IDLE)
                return

            if not transcript.strip():
                log.info("empty transcript, returning to IDLE")
                self._set_state(PipelineState.IDLE)
                return

            # -- Intent --
            response: str | bytes | None = None
            if self._intent_handler is not None:
                response = await self._intent_handler(transcript)

            if response is None:
                self._set_state(PipelineState.IDLE)
                return

            # -- TTS / Playback --
            self._set_state(PipelineState.RESPONDING)

            if isinstance(response, bytes):
                # Raw audio returned from intent handler
                audio = np.frombuffer(response, dtype=np.int16)
            elif isinstance(response, str) and self._tts_handler is not None:
                audio = await self._tts_handler(response)
            else:
                log.warning("no TTS handler and response is text — dropping")
                self._set_state(PipelineState.IDLE)
                return

            await self._playback.enqueue(
                audio, sample_rate=22050, priority=PlaybackPriority.TTS, label="response"
            )

            # Wait for playback to finish before returning to IDLE
            while self._playback.is_playing:
                await asyncio.sleep(0.05)

        except Exception:
            log.exception("error in utterance processing")
        finally:
            self._aec.notify_playback_stopped()
            self._set_state(PipelineState.IDLE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _generate_beep(
    frequency: float = 880,
    duration_ms: int = 80,
    sample_rate: int = 22050,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a short sine-wave beep as int16."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    # Apply 5 ms fade-in/out to avoid clicks
    fade_samples = int(sample_rate * 0.005)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return (wave * 32767).astype(np.int16)
