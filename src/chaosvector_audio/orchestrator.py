"""Pipeline orchestrator — ties wake, STT, intent, LLM, TTS, and playback together.

This is the top-level daemon that replaces satellite.py. It imports the existing
intent classifier and Ollama client from pi-fi-software and wires them into the
new audio pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chaosvector_audio.capture import CaptureConfig, CaptureManager, AudioChunk
from chaosvector_audio.playback import PlaybackConfig, PlaybackManager, PlaybackPriority
from chaosvector_audio.vad import VADConfig, VoiceActivityDetector
from chaosvector_audio.wake import WakeConfig, WakeWordClient
from chaosvector_audio.stt import STTConfig, transcribe
from chaosvector_audio.tts import TTSConfig, synthesize
from chaosvector_audio.llm import LLMConfig, LLMClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified config for the ChaosVector Audio pipeline."""
    # Audio
    mic_device: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 20
    pre_roll_ms: int = 500
    playback_device: str | None = None
    playback_rate: int = 22050

    # Wake word
    wake_host: str = "127.0.0.1"
    wake_port: int = 10400
    wake_names: list[str] = field(default_factory=lambda: ["hey_jarvis"])
    wake_energy_threshold: float = 200.0
    wake_gain: float = 1.0

    # VAD
    vad_aggressiveness: int = 2
    silence_frames: int = 20  # ~400ms at 20ms frames
    min_speech_frames: int = 3
    listen_timeout: float = 10.0

    # STT
    stt_host: str = "10.1.1.240"
    stt_port: int = 10301
    stt_timeout: float = 10.0

    # TTS
    tts_host: str = "10.1.1.240"
    tts_port: int = 10210
    tts_voice: str = "af_heart"
    tts_timeout: float = 10.0

    # Ollama / LLM
    ollama_url: str = "http://10.1.1.228:8080"
    ollama_model: str = "gemma4-e4b"
    ollama_api_format: str = "openai"
    ollama_system_prompt_file: str = ""
    ollama_timeout: float = 15.0
    ollama_max_tokens: int = 120

    # HA
    ha_url: str = "http://10.1.1.53:8123"
    ha_token: str = ""

    # Follow-up
    follow_up_timeout: float = 5.0

    # Chime blanking
    chime_blanking_ms: int = 100

    # Pi-Fi software path (for importing intent classifier etc.)
    pifi_path: str = "/home/chaos/pi-fi-software/voice"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Main pipeline orchestrator — IDLE → LISTENING → PROCESSING → RESPONDING loop."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Audio components
        self._capture = CaptureManager(CaptureConfig(
            device=config.mic_device,
            sample_rate=config.sample_rate,
            channels=config.channels,
            chunk_duration_ms=config.chunk_ms,
            pre_roll_ms=config.pre_roll_ms,
        ))
        self._playback = PlaybackManager(PlaybackConfig(
            device=config.playback_device,
            sample_rate=config.playback_rate,
            channels=1,
        ))
        self._vad = VoiceActivityDetector(VADConfig(
            aggressiveness=config.vad_aggressiveness,
            sample_rate=config.sample_rate,
            frame_duration_ms=config.chunk_ms,
            silence_frames_threshold=config.silence_frames,
            min_speech_frames=config.min_speech_frames,
        ))
        self._wake = WakeWordClient(WakeConfig(
            host=config.wake_host,
            port=config.wake_port,
            names=config.wake_names,
            energy_threshold=config.wake_energy_threshold,
            gain=config.wake_gain,
        ))

        # Service configs
        self._stt_config = STTConfig(
            host=config.stt_host, port=config.stt_port, timeout=config.stt_timeout,
        )
        self._tts_config = TTSConfig(
            host=config.tts_host, port=config.tts_port,
            voice=config.tts_voice, timeout=config.tts_timeout,
        )

        # LLM client (self-contained, no pi-fi dependency)
        system_prompt = ""
        if config.ollama_system_prompt_file:
            try:
                system_prompt = Path(config.ollama_system_prompt_file).read_text().strip()
            except Exception:
                pass
        self._llm = LLMClient(LLMConfig(
            url=config.ollama_url,
            timeout=config.ollama_timeout,
            max_tokens=config.ollama_max_tokens,
            system_prompt=system_prompt,
        ))

        # Lazy-loaded components
        self._classifier = None

        # State
        self._wake_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._running = False
        self._interaction_count = 0
        self._beep = _generate_beep()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start all components and enter the main loop."""
        # Import pi-fi components
        self._load_pifi_modules()

        await self._capture.open()
        await self._playback.start()
        await self._wake.start(self._wake_audio_queue)

        # Connect LLM
        await self._llm.connect()

        self._running = True
        log.info("orchestrator started")

    async def stop(self) -> None:
        self._running = False
        await self._wake.stop()
        await self._playback.stop()
        await self._capture.close()
        await self._llm.disconnect()
        log.info("orchestrator stopped")

    async def run(self) -> None:
        """Main loop — runs until cancelled or stop() called."""
        try:
            while self._running:
                await self._idle_loop()
        except asyncio.CancelledError:
            pass

    # -- IDLE ----------------------------------------------------------------

    async def _idle_loop(self) -> None:
        """IDLE state: stream audio to wake word, wait for detection."""
        log.info("=== IDLE ===")
        self._vad.reset()

        # Feed audio to wake word detector
        feed_task = asyncio.create_task(self._feed_wake_audio())
        try:
            name, rms = await self._wake.wait_for_wake()
        finally:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass

        self._interaction_count += 1
        log.info("WAKE #%d: '%s' (rms=%.1f)", self._interaction_count, name, rms)

        # Play wake beep
        await self._playback.enqueue(
            self._beep, sample_rate=22050,
            priority=PlaybackPriority.WAKE_BEEP, label="wake-beep",
        )
        await asyncio.sleep(0.12)

        # Transition to LISTENING
        utterance = await self._listen()
        if not utterance:
            self._wake.force_reconnect()
            await asyncio.sleep(0.3)
            return

        # Transition to PROCESSING
        transcript = await self._process_stt(utterance)
        if not transcript:
            self._wake.force_reconnect()
            await asyncio.sleep(0.3)
            return

        # Transition to RESPONDING
        await self._respond(transcript)

        # Clean up for next cycle
        self._wake.force_reconnect()
        await asyncio.sleep(0.3)

    # -- LISTENING -----------------------------------------------------------

    async def _listen(self) -> list[AudioChunk] | None:
        """Collect utterance via VAD. Returns chunks or None on timeout/empty."""
        log.info("=== LISTENING ===")
        pre_roll = self._capture.drain_pre_roll()
        utterance: list[AudioChunk] = list(pre_roll)
        listen_start = time.monotonic()

        # Chime blanking
        blanking_chunks = int(self.config.chime_blanking_ms / self.config.chunk_ms)
        blanked = 0

        async for chunk in self._capture.chunks():
            if blanked < blanking_chunks:
                blanked += 1
                continue
            utterance.append(chunk)
            _, end_of_speech = self._vad.process_frame(chunk.samples)
            if end_of_speech:
                break
            if time.monotonic() - listen_start > self.config.listen_timeout:
                log.warning("listen timeout")
                break

        total_ms = sum(len(c.samples) for c in utterance) / self.config.sample_rate * 1000
        log.info("LISTENING done: %d chunks, %.0fms audio", len(utterance), total_ms)

        if total_ms < 200:  # too short to be real speech
            return None
        return utterance

    # -- PROCESSING ----------------------------------------------------------

    async def _process_stt(self, chunks: list[AudioChunk]) -> str | None:
        """Run STT on collected audio."""
        log.info("=== STT ===")
        transcript = await transcribe(chunks, self._stt_config)
        if transcript:
            log.info("Transcript: \"%s\"", transcript)
        return transcript

    # -- RESPONDING ----------------------------------------------------------

    async def _respond(self, transcript: str) -> None:
        """Classify intent and generate response."""
        log.info("=== RESPONDING ===")

        # Classify intent
        intent_type = "general"
        if self._classifier is not None:
            try:
                intents = self._classifier.classify_compound(transcript)
                if intents:
                    intent_type = intents[0].type.value
                    log.info("Intent: %s (confidence=%.2f)", intent_type, intents[0].confidence)
            except Exception as e:
                log.warning("intent classification failed: %s", e)

        # For now: route everything through LLM for conversational response
        # TODO: Add HA device control, music, timers, etc.
        log.info("Routing to LLM (intent=%s)", intent_type)
        await self._respond_llm(transcript)

    async def _respond_llm(self, transcript: str) -> None:
        """Stream response from LLM with sentence-level TTS."""
        if not self._llm.is_available:
            log.warning("LLM not available")
            await self._speak("I heard you, but I can't reach the language model right now.")
            return

        log.info("Streaming from LLM...")

        # Producer-consumer: LLM yields sentences, we synthesize and play each
        sentence_count = 0
        try:
            async for sentence in self._llm.generate_stream(transcript):
                sentence_count += 1
                log.info("LLM sentence %d: \"%s\"", sentence_count, sentence[:80])

                result = await synthesize(sentence, self._tts_config)
                if result is not None:
                    await self._playback.enqueue(
                        result.audio,
                        sample_rate=result.sample_rate,
                        channels=result.channels,
                        priority=PlaybackPriority.TTS,
                        label=f"response-{sentence_count}",
                    )
                else:
                    log.warning("TTS failed for sentence %d", sentence_count)
        except Exception as e:
            log.error("LLM streaming error: %s", e, exc_info=True)

        # Wait for all playback to finish
        while self._playback.is_playing:
            await asyncio.sleep(0.05)

        if sentence_count == 0:
            log.warning("LLM produced no response")
            await self._speak("Sorry, I didn't get a response from the language model.")

    async def _speak(self, text: str) -> None:
        """Synthesize and play a single text response."""
        result = await synthesize(text, self._tts_config)
        if result is not None:
            await self._playback.enqueue(
                result.audio,
                sample_rate=result.sample_rate,
                channels=result.channels,
                priority=PlaybackPriority.TTS,
                label="speak",
            )
            while self._playback.is_playing:
                await asyncio.sleep(0.05)

    # -- helpers -------------------------------------------------------------

    async def _feed_wake_audio(self) -> None:
        """Feed capture chunks to wake word queue."""
        async for chunk in self._capture.chunks():
            raw = chunk.samples.astype(np.int16).tobytes()
            try:
                self._wake_audio_queue.put_nowait(raw)
            except asyncio.QueueFull:
                pass

    def _load_pifi_modules(self) -> None:
        """Import intent classifier from pi-fi-software."""
        import sys
        pifi_path = self.config.pifi_path
        if pifi_path not in sys.path:
            sys.path.insert(0, pifi_path)

        # Intent classifier has no relative imports — direct import works
        try:
            from intent_classifier import IntentClassifier
            self._classifier = IntentClassifier()
            log.info("intent classifier loaded")
        except ImportError as e:
            log.warning("intent classifier unavailable: %s", e)



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _generate_beep(
    frequency: float = 880, duration_ms: int = 80,
    sample_rate: int = 22050, amplitude: float = 0.3,
) -> np.ndarray:
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    fade = int(sample_rate * 0.005)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return (wave * 32767).astype(np.int16)
