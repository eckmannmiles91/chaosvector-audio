"""Pipeline orchestrator — ties wake, STT, intent, LLM, TTS, and playback together.

This is the top-level daemon that replaces satellite.py. It imports the existing
intent classifier from pi-fi-software and integrates:
- Local intent handling (time, weather, calendar via context engine)
- HA device control (WebSocket intent execution)
- LLM streaming (Gemma 4 via llama-server)
- Echo gate (suppress wake during TTS playback)
- Barge-in (wake during playback stops TTS)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np

from chaosvector_audio.capture import CaptureConfig, CaptureManager, AudioChunk
from chaosvector_audio.playback import PlaybackConfig, PlaybackManager, PlaybackPriority
from chaosvector_audio.vad import VADConfig, VoiceActivityDetector
from chaosvector_audio.wake import WakeConfig, WakeWordClient
from chaosvector_audio.stt import STTConfig, transcribe
from chaosvector_audio.tts import TTSConfig, synthesize
from chaosvector_audio.llm import LLMConfig, LLMClient
from chaosvector_audio.context import ContextConfig, ContextClient, get_local_time
from chaosvector_audio.ha import HAConfig, HAClient
from chaosvector_audio.feedback import FeedbackLogger
from chaosvector_audio.speaker import SpeakerConfig, identify_speaker
from chaosvector_audio.stt_filters import correct_stt, is_stt_garbage
from chaosvector_audio.sounds import ThinkingIndicator

log = logging.getLogger(__name__)


# Device command detection (same regex as satellite.py)
_DEVICE_CMD_RE = re.compile(
    r"\b(?:turn\s+(?:on|off)|switch\s+(?:on|off)|toggle"
    r"|(?:open|close|lock|unlock)\s+the"
    r"|dim\s+the|brighten\s+the"
    r"|set\s+(?:the\s+)?(?:thermostat|temperature|temp)\b"
    r"|(?:turn|switch)\s+(?:the\s+)?(?:heat|heating|cool(?:ing)?|ac|air)\s+(?:on|off))",
    re.IGNORECASE,
)

# Broader check for anything that might be a device/HA command
_HA_CANDIDATE_RE = re.compile(
    r"\b(?:turn|switch|toggle|open|close|lock|unlock|dim|brighten|set|increase|decrease"
    r"|lights?|lamp|fan|thermostat|temperature|temp|heat|cool|ac|air"
    r"|garage|door|blind|curtain|shade|volume)\b",
    re.IGNORECASE,
)


def _might_be_device_cmd(text: str) -> bool:
    """Quick check if text could be a device command worth sending to HA."""
    return bool(_HA_CANDIDATE_RE.search(text))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified config for the ChaosVector Audio pipeline."""
    # Audio — default to XVF3800 hardware AEC output
    mic_device: str | None = None  # None = PipeWire default; set to XVF3800 ALSA name for hardware AEC
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

    # LLM
    ollama_url: str = "http://10.1.1.228:8080"
    ollama_system_prompt_file: str = ""
    ollama_timeout: float = 15.0
    ollama_max_tokens: int = 120

    # Home Assistant
    ha_ws_url: str = "ws://10.1.1.53:8123/api/websocket"
    ha_http_url: str = "http://10.1.1.53:8123"
    ha_token: str = ""
    ha_pipeline: str | None = None
    ha_intent_timeout: float = 10.0

    # Context engine
    context_url: str = "http://10.1.1.176:8400"

    # Follow-up
    follow_up_timeout: float = 5.0

    # Chime blanking
    chime_blanking_ms: int = 100

    # Echo gate: suppress wake detection for this many ms after playback ends
    echo_gate_ms: int = 300

    # Speaker verification
    speaker_url: str = "http://10.1.1.228:8500"
    speaker_enabled: bool = True

    # Feedback logging
    feedback_dir: str = "/var/lib/pi-fi/feedback"

    # Sounds directory
    sounds_dir: str = "/home/chaos/pi-fi-software/voice/sounds"

    # Volume adaptation
    volume_adapt: bool = True
    volume_adapt_min: float = 0.25
    volume_adapt_max: float = 0.85
    volume_adapt_rms_low: int = 500
    volume_adapt_rms_high: int = 4000

    # Brief mode
    brief_mode: bool = True
    brief_min_frequency: int = 3
    brief_top_n: int = 20

    # AVR ducking
    avr_enabled: bool = False
    avr_device_name: str = ""
    avr_restore_delay: float = 1.0

    # Pi-Fi software path (for importing intent classifier + managers)
    pifi_path: str = "/home/chaos/pi-fi-software/voice"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Main pipeline orchestrator — IDLE → LISTENING → PROCESSING → RESPONDING loop.

    Features:
    - Local intent handling (time/weather/calendar via context engine)
    - HA device control (WebSocket)
    - LLM streaming with sentence-level TTS
    - Echo gate (suppress wake during/after TTS)
    - Barge-in (wake during playback stops TTS)
    """

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

        # LLM client
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

        # Context engine
        self._context = ContextClient(ContextConfig(url=config.context_url))

        # Home Assistant
        self._ha = HAClient(HAConfig(
            ws_url=config.ha_ws_url,
            http_url=config.ha_http_url,
            token=config.ha_token,
            pipeline=config.ha_pipeline,
            intent_timeout=config.ha_intent_timeout,
        ))

        # Feedback logger
        self._feedback = FeedbackLogger(config.feedback_dir)

        # Speaker verification
        self._speaker_config = SpeakerConfig(
            url=config.speaker_url, enabled=config.speaker_enabled,
        )

        # Thinking indicator
        self._thinking = ThinkingIndicator(self._playback, config.sounds_dir)

        # Lazy-loaded from pi-fi-software
        self._classifier = None
        self._timer_mgr = None
        self._reminder_mgr = None
        self._alarm_mgr = None

        # Brief mode + routines
        self._frequent_commands: set[str] = set()
        self._routines: list[dict] = []

        # State
        self._wake_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._speaker_name: str | None = None
        self._last_wake_rms: float = 0.0
        self._speech_rms: float = 0.0
        self._stt_ms: float = 0.0
        self._stt_start: float = 0.0
        self._running = False
        self._interaction_count = 0
        self._beep = _load_wake_sound(config.pifi_path)
        self._beep_rate = 44100  # wake.wav is 44100Hz
        self._responding = False  # True while TTS is playing (echo gate)
        self._last_playback_end: float = 0.0  # for echo gate tail

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start all components."""
        self._load_pifi_modules()

        await self._capture.open()
        await self._playback.start()
        await self._wake.start(self._wake_audio_queue)
        await self._llm.connect()
        await self._context.connect()
        await self._ha.connect()

        # Restore persisted timers/reminders/alarms
        await self._restore_scheduled()

        # Sync routines and frequent commands from context engine
        await self._sync_routines_and_brief()

        self._running = True
        log.info("orchestrator started")

    async def stop(self) -> None:
        self._running = False
        await self._wake.stop()
        await self._playback.stop()
        await self._capture.close()
        await self._llm.disconnect()
        await self._context.disconnect()
        await self._ha.disconnect()
        log.info("orchestrator stopped")

    async def run(self) -> None:
        """Main loop — runs until cancelled."""
        try:
            while self._running:
                await self._idle_loop()
        except asyncio.CancelledError:
            pass

    # -- IDLE ----------------------------------------------------------------

    async def _idle_loop(self) -> None:
        """IDLE: stream audio to wake word, wait for detection.
        Echo gate: skip wake detection while playing or within tail window."""
        log.info("=== IDLE ===")
        self._vad.reset()

        # Feed audio to wake word detector (with echo gate)
        feed_task = asyncio.create_task(self._feed_wake_audio())
        try:
            name, rms = await self._wake.wait_for_wake()
        finally:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass

        # Echo gate check: reject wake if we're still in the echo tail
        if self._is_echo_active():
            log.info("wake rejected (echo gate active)")
            return

        self._interaction_count += 1
        self._last_wake_rms = rms
        log.info("WAKE #%d: '%s' (rms=%.1f)", self._interaction_count, name, rms)

        # Barge-in: stop any current playback
        if self._playback.is_playing:
            log.info("barge-in: stopping playback")
            self._playback.barge_in()

        # Play wake sound and wait for it to finish (max 500ms)
        await self._playback.enqueue(
            self._beep, sample_rate=self._beep_rate,
            priority=PlaybackPriority.WAKE_BEEP, label="wake-beep",
        )
        beep_wait = 0
        while self._playback.is_playing and beep_wait < 0.5:
            await asyncio.sleep(0.02)
            beep_wait += 0.02
        await asyncio.sleep(0.05)  # small gap after beep

        # LISTENING
        utterance = await self._listen()
        if not utterance:
            self._wake.force_reconnect()
            await asyncio.sleep(0.3)
            return

        # STT
        transcript = await self._process_stt(utterance)
        if not transcript:
            self._wake.force_reconnect()
            await asyncio.sleep(0.3)
            return

        # RESPOND
        wants_followup = await self._respond(transcript)

        # Mark playback end for echo gate
        self._last_playback_end = time.monotonic()

        # Follow-up mode: listen again after conversational responses
        # Requires AEC mic source (ec_source) to avoid hearing own TTS
        while wants_followup and self._running:
            log.info("=== FOLLOW-UP (%.0fs window) ===", self.config.follow_up_timeout)
            # Brief pause for echo gate after TTS
            await asyncio.sleep(0.3)

            # Listen with follow-up timeout (shorter than normal)
            utterance = await self._listen_followup()
            if not utterance:
                log.info("follow-up: no speech, returning to IDLE")
                break

            transcript = await self._process_stt(utterance)
            if not transcript:
                break

            wants_followup = await self._respond(transcript)
            self._last_playback_end = time.monotonic()

        # Clean up
        self._wake.force_reconnect()
        await asyncio.sleep(0.3)

    # -- LISTENING -----------------------------------------------------------

    async def _listen(self) -> list[AudioChunk] | None:
        """Collect utterance via VAD."""
        log.info("=== LISTENING ===")
        pre_roll = self._capture.drain_pre_roll()
        utterance: list[AudioChunk] = list(pre_roll)
        listen_start = time.monotonic()

        blanking_chunks = int(self.config.chime_blanking_ms / self.config.chunk_ms)
        blanked = 0

        min_listen_s = 1.0  # don't accept end-of-speech before 1s

        async for chunk in self._capture.chunks():
            if blanked < blanking_chunks:
                blanked += 1
                continue
            utterance.append(chunk)
            _, end_of_speech = self._vad.process_frame(chunk.samples)
            elapsed = time.monotonic() - listen_start
            if end_of_speech and elapsed > min_listen_s:
                break
            if elapsed > self.config.listen_timeout:
                log.warning("listen timeout")
                break

        total_ms = sum(len(c.samples) for c in utterance) / self.config.sample_rate * 1000
        log.info("LISTENING done: %d chunks, %.0fms audio", len(utterance), total_ms)

        # Compute average speech RMS for volume adaptation
        if utterance:
            rms_sum = sum(c.rms for c in utterance)
            self._speech_rms = (rms_sum / len(utterance)) * 32768  # convert to int16 scale
        else:
            self._speech_rms = 0.0

        if total_ms < 200:
            return None
        return utterance

    async def _listen_followup(self) -> list[AudioChunk] | None:
        """Listen for follow-up speech without wake word.

        Simple approach: wait for playback to end, then use normal VAD
        listening with the standard _listen() method but with a shorter
        timeout. The VAD handles speech detection the same way as after
        a wake word.
        """
        # Wait for playback + echo to fully clear
        while self._playback.is_playing:
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.5)  # let room reverb settle

        # Drain any stale audio from the capture queue
        while True:
            try:
                import queue as _q
                self._capture._thread_queue.get_nowait()
            except _q.Empty:
                break

        # Use normal listen with follow-up timeout
        log.info("follow-up: listening for %.0fs...", self.config.follow_up_timeout)
        self._vad.reset()
        utterance: list[AudioChunk] = []
        listen_start = time.monotonic()
        speech_started = False

        async for chunk in self._capture.chunks():
            elapsed = time.monotonic() - listen_start

            _, end_of_speech = self._vad.process_frame(chunk.samples)

            if not speech_started:
                if self._vad._state.name == "SPEECH":
                    speech_started = True
                    utterance.append(chunk)
                elif elapsed > self.config.follow_up_timeout:
                    return None  # timed out waiting for speech
            else:
                utterance.append(chunk)
                if end_of_speech and elapsed > 1.5:  # min 1.5s after speech starts
                    break
                if elapsed > self.config.follow_up_timeout + self.config.listen_timeout:
                    break

        if not utterance:
            return None

        total_ms = sum(len(c.samples) for c in utterance) / self.config.sample_rate * 1000
        log.info("FOLLOW-UP listen: %d chunks, %.0fms audio", len(utterance), total_ms)

        if total_ms < 300:
            return None
        return utterance

    # -- STT -----------------------------------------------------------------

    async def _process_stt(self, chunks: list[AudioChunk]) -> str | None:
        """Run STT with name corrections and hallucination filtering."""
        log.info("=== STT ===")
        self._stt_start = time.monotonic()
        transcript = await transcribe(chunks, self._stt_config)
        self._stt_ms = (time.monotonic() - self._stt_start) * 1000

        if not transcript:
            return None

        # Apply name corrections
        transcript = correct_stt(transcript)

        # Filter hallucinations
        if is_stt_garbage(transcript):
            log.info("STT garbage filtered: \"%s\"", transcript)
            return None

        log.info("Transcript: \"%s\"", transcript)

        # Start speaker verification in background (uses first 3s of audio)
        speaker_audio = np.concatenate([c.samples for c in chunks[:150]])  # ~3s
        asyncio.create_task(self._identify_speaker(speaker_audio))

        return transcript

    async def _identify_speaker(self, audio: np.ndarray) -> None:
        """Background speaker identification."""
        self._speaker_name = await identify_speaker(audio, self._speaker_config)

    # -- RESPONDING ----------------------------------------------------------

    async def _respond(self, transcript: str) -> bool:
        """Classify intent and route to appropriate handler.
        Returns True if follow-up listening is appropriate."""
        log.info("=== RESPONDING ===")
        self._responding = True
        follow_up = False
        respond_start = time.monotonic()
        response_text = ""
        route = ""

        # AVR ducking
        await self._duck_avr()

        # Volume adaptation
        self._apply_volume_adaptation()

        try:
            # Check routines BEFORE intent classification (higher priority)
            routine = self._match_routine(transcript)
            if routine:
                log.info("Routine matched: '%s'", routine.get("trigger", ""))
                await self._run_routine(routine)
                route = "routine"
                return False  # no follow-up after routines

            # Classify
            intent_type = "general"
            context_query = None
            if self._classifier is not None:
                try:
                    intents = self._classifier.classify_compound(transcript)
                    if intents:
                        intent_type = intents[0].type.value
                        context_query = getattr(intents[0], "context_query", None)
                        log.info("Intent: %s (confidence=%.2f, context=%s)",
                                 intent_type, intents[0].confidence, context_query)
                except Exception as e:
                    log.warning("intent classification failed: %s", e)

            # Route based on intent type
            if intent_type == "simple_local":
                response_text = await self._handle_simple_local(transcript, context_query)
                route = f"context:{context_query}" if context_query else "local"
            elif intent_type == "general" and _DEVICE_CMD_RE.search(transcript):
                response_text = await self._handle_ha_device(transcript)
                route = "ha"
            elif intent_type == "general":
                await self._handle_general(transcript)
                follow_up = True
                route = "llm"
            elif intent_type == "complex":
                await self._respond_llm(transcript)
                follow_up = True
                route = "llm"
            else:
                await self._respond_llm(transcript)
                follow_up = True
                route = "llm"

        finally:
            self._responding = False
            # Restore normal volume and AVR
            self._playback.set_volume(self._playback.config.volume)
            await self._restore_avr()

            # Log interaction
            total_ms = (time.monotonic() - respond_start) * 1000
            try:
                self._feedback.log_interaction(
                    transcript=transcript,
                    intent_type=intent_type,
                    response_text=response_text or "",
                    speaker=self._speaker_name,
                    route=route,
                    wake_rms=self._last_wake_rms,
                    stt_ms=getattr(self, "_stt_ms", 0),
                    total_ms=total_ms + getattr(self, "_stt_ms", 0),
                    context_query=context_query,
                )
            except Exception as e:
                log.debug("feedback log failed: %s", e)

        return follow_up

    async def _handle_simple_local(self, transcript: str, context_query: str | None) -> str:
        """Handle simple_local intents: time, weather, calendar, presence.
        Returns response text."""
        # Time is pure local — instant
        if context_query == "time" or "time" in transcript.lower():
            response = get_local_time()
            log.info("Local time: %s", response)
            await self._speak(response)
            return response

        # Everything else: ask context engine
        if context_query and self._context.is_available:
            answer = await self._context.get_answer(context_query)
            if answer:
                log.info("Context answer (%s): %s", context_query, answer[:80])
                await self._speak(answer)
                return answer

        # Fallback: try LLM
        log.info("simple_local fallback to LLM (context_query=%s)", context_query)
        await self._respond_llm(transcript)
        return ""

    async def _handle_ha_device(self, transcript: str) -> str:
        """Handle device commands via HA. Returns response text."""
        if not self._ha.is_available:
            log.warning("HA not available, falling back to LLM")
            await self._respond_llm(transcript)
            return ""

        log.info("Sending to HA: \"%s\"", transcript)
        response = await self._ha.run_intent(transcript)

        if response:
            # Brief mode: play chime instead of TTS for frequent commands
            if self._is_brief_response(transcript, response):
                from chaosvector_audio.sounds import load_sound
                sound = load_sound("confirm", self.config.sounds_dir)
                if sound is not None:
                    audio, rate = sound
                    await self._playback.enqueue(audio, sample_rate=rate,
                                                 priority=PlaybackPriority.TTS, label="confirm")
                    await self._wait_playback(timeout=3.0)
                    log.info("brief mode: chime for '%s'", transcript[:40])
                    return response
            await self._speak(response)
            return response
        else:
            log.info("HA returned no response, trying LLM")
            await self._respond_llm(transcript)
            return ""

    async def _handle_general(self, transcript: str) -> None:
        """Handle general intents: try HA for device-like commands, else LLM."""
        # Only try HA if it looks like it could be a device command
        if self._ha.is_available and _might_be_device_cmd(transcript):
            response = await self._ha.run_intent(transcript)
            if response and response.lower() not in (
                "sorry, i couldn't understand that",
                "sorry, i'm not sure how to help with that",
            ):
                await self._speak(response)
                return

        # Not a device command or HA didn't handle it — use LLM with context
        enriched = await self._enrich_prompt(transcript)
        await self._respond_llm(enriched)

    async def _enrich_prompt(self, transcript: str) -> str:
        """Enrich the user's transcript with relevant context for the LLM.

        Fetches relevant context from the context engine and prepends it
        so the LLM can give informed answers about weather, calendar, etc.
        """
        if not self._context.is_available:
            return transcript

        try:
            ctx = await self._context.get_relevant_context(
                transcript, speaker=self._speaker_name,
            )
            if ctx:
                # Build context block
                parts = []
                if "weather" in ctx:
                    parts.append(f"[Weather] {ctx['weather']}")
                if "calendar" in ctx:
                    parts.append(f"[Calendar] {ctx['calendar']}")
                if "presence" in ctx:
                    parts.append(f"[Presence] {ctx['presence']}")
                if "time" in ctx:
                    parts.append(f"[Time] {ctx['time']}")
                if "memories" in ctx:
                    for m in ctx["memories"][:3]:
                        parts.append(f"[Memory] {m.get('fact', '')}")

                if self._speaker_name:
                    parts.append(f"[Speaker: {self._speaker_name}]")

                if parts:
                    context_block = "\n".join(parts)
                    log.info("Context enrichment: %d items", len(parts))
                    return f"[Context]\n{context_block}\n\n{transcript}"
        except Exception as e:
            log.debug("context enrichment failed: %s", e)

        return transcript

    async def _respond_llm(self, transcript: str) -> None:
        """Stream response from LLM with sentence-level TTS.
        Supports barge-in: wake word during playback stops response."""
        if not self._llm.is_available:
            log.warning("LLM not available")
            await self._speak("I heard you, but I can't reach the language model right now.")
            return

        log.info("Streaming from LLM...")

        # Play thinking sound (cancelled when first audio plays)
        await self._thinking.start()

        barged = False
        barge_feed = None
        sentence_count = 0

        try:
            async for sentence in self._llm.generate_stream(transcript):
                # Check for barge-in after first sentence is playing
                if barge_feed is not None and self._wake.has_pending_wake():
                    log.info("barge-in detected during LLM stream")
                    self._playback.barge_in()
                    barged = True
                    break

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
                    # Stop thinking sound before first audio plays
                    if barge_feed is None:
                        await self._thinking.stop()
                    # Start barge-in listener AFTER first sentence is queued
                    if barge_feed is None:
                        self._wake.has_pending_wake()  # clear stale
                        barge_feed = asyncio.create_task(self._feed_wake_audio())
                else:
                    log.warning("TTS failed for sentence %d", sentence_count)
        except Exception as e:
            log.error("LLM streaming error: %s", e, exc_info=True)

        if not barged:
            # Keep barge-in feed running during playback wait
            if barge_feed is None:
                self._wake.has_pending_wake()  # clear stale
                barge_feed = asyncio.create_task(self._feed_wake_audio())
            barged = await self._wait_playback_with_bargein(timeout=30.0)

        # Now clean up the barge-in feed
        if barge_feed is not None:
            barge_feed.cancel()
            try:
                await barge_feed
            except asyncio.CancelledError:
                pass

        if sentence_count == 0 and not barged:
            log.warning("LLM produced no response")
            await self._speak("Sorry, I didn't get a response.")

    async def _speak(self, text: str) -> None:
        """Synthesize and play a single text response.
        Supports barge-in during playback."""
        result = await synthesize(text, self._tts_config)
        if result is not None:
            # Start barge-in listener during playback
            self._wake.has_pending_wake()  # clear stale
            barge_feed = asyncio.create_task(self._feed_wake_audio())
            await self._playback.enqueue(
                result.audio,
                sample_rate=result.sample_rate,
                channels=result.channels,
                priority=PlaybackPriority.TTS,
                label="speak",
            )
            await self._wait_playback_with_bargein(timeout=15.0)
            barge_feed.cancel()
            try:
                await barge_feed
            except asyncio.CancelledError:
                pass

    async def _wait_playback(self, timeout: float = 15.0) -> None:
        """Wait for playback to finish with a timeout to prevent hangs."""
        waited = 0.0
        while self._playback.is_playing and waited < timeout:
            await asyncio.sleep(0.05)
            waited += 0.05
        if waited >= timeout:
            log.warning("playback wait timed out after %.1fs", timeout)

    async def _wait_playback_with_bargein(self, timeout: float = 30.0) -> bool:
        """Wait for playback, checking for barge-in every 100ms.
        Returns True if barge-in occurred."""
        waited = 0.0
        while self._playback.is_playing and waited < timeout:
            if self._wake.has_pending_wake():
                log.info("barge-in: stopping playback")
                self._playback.barge_in()
                return True
            await asyncio.sleep(0.1)
            waited += 0.1
        return False

    # -- Brief mode, routines, timers ----------------------------------------

    _DEVICE_CONFIRMATIONS = {"Done.", "Got it.", "On it.", "All set.", "You got it.",
                             "Turned on the light", "Turned off the light",
                             "Turned on the lights", "Turned off the lights"}

    def _is_brief_response(self, transcript: str, response: str) -> bool:
        """Check if this device response should be a chime instead of TTS."""
        if not self.config.brief_mode:
            return False
        if response not in self._DEVICE_CONFIRMATIONS:
            return False
        return self._is_frequent_command(transcript)

    def _is_frequent_command(self, text: str) -> bool:
        """Check if text matches a known frequent command."""
        if not self._frequent_commands:
            return False
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        if text_lower in {c.lower() for c in self._frequent_commands}:
            return True
        for cmd in self._frequent_commands:
            cmd_words = set(cmd.lower().split())
            if cmd_words and text_words:
                overlap = len(text_words & cmd_words)
                total = max(len(text_words), len(cmd_words))
                if overlap / total >= 0.8:
                    return True
        return False

    def _match_routine(self, text: str) -> dict | None:
        """Check if spoken text matches a saved routine trigger."""
        if not self._routines:
            return None
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        best = None
        best_score = 0.0
        for r in self._routines:
            trigger_lower = r["trigger"].lower().strip()
            trigger_words = set(trigger_lower.split())
            if text_lower == trigger_lower:
                return r
            if trigger_words and text_words:
                overlap = len(text_words & trigger_words)
                total = max(len(text_words), len(trigger_words))
                ratio = overlap / total
                if ratio >= 0.8 and ratio > best_score:
                    best = r
                    best_score = ratio
        return best

    async def _run_routine(self, routine: dict) -> None:
        """Execute a routine's steps sequentially."""
        steps = routine.get("steps", [])
        log.info("Running routine '%s' (%d steps)", routine.get("trigger", ""), len(steps))
        for i, step in enumerate(steps):
            log.info("Routine step %d/%d: %s", i + 1, len(steps), step)
            # Each step is processed as a normal transcript
            await self._respond(step)
            if i < len(steps) - 1:
                await asyncio.sleep(0.3)

    async def _restore_scheduled(self) -> None:
        """Restore persisted reminders and alarms from disk."""
        if self._reminder_mgr is not None:
            try:
                count = await self._reminder_mgr.load_and_schedule(self._on_reminder_fire)
                if count:
                    log.info("restored %d reminder(s)", count)
            except Exception as e:
                log.debug("reminder restore failed: %s", e)

        if self._alarm_mgr is not None:
            try:
                count = await self._alarm_mgr.load_and_schedule(self._on_alarm_fire)
                if count:
                    log.info("restored %d alarm(s)", count)
            except Exception as e:
                log.debug("alarm restore failed: %s", e)

    async def _on_timer_fire(self, label: str, duration: int) -> None:
        """Callback when a timer expires."""
        log.info("Timer fired: %s", label)
        await self._speak(f"Your {label} timer is done." if label else "Timer is done.")

    async def _on_reminder_fire(self, label: str) -> None:
        """Callback when a reminder fires."""
        log.info("Reminder fired: %s", label)
        await self._speak(f"Reminder: {label}" if label else "You have a reminder.")

    async def _on_alarm_fire(self, label: str) -> None:
        """Callback when an alarm fires."""
        log.info("Alarm fired: %s", label)
        from chaosvector_audio.sounds import load_sound
        sound = load_sound("alarm", self.config.sounds_dir)
        if sound is not None:
            audio, rate = sound
            await self._playback.enqueue(audio, sample_rate=rate,
                                         priority=PlaybackPriority.NOTIFICATION, label="alarm")
        await self._speak(f"Alarm: {label}" if label else "Your alarm is going off.")

    async def _sync_routines_and_brief(self) -> None:
        """Sync routines and frequent commands from context engine."""
        if not self._context.is_available:
            return

        # Routines
        try:
            routines = await self._context._session.get(
                f"{self._context.config.url}/routine/list",
                timeout=aiohttp.ClientTimeout(total=3.0),
            )
            async with routines as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._routines = data.get("routines", [])
                    if self._routines:
                        log.info("loaded %d routines", len(self._routines))
        except Exception as e:
            log.debug("routine sync failed: %s", e)

        # Brief mode frequent commands
        if self.config.brief_mode:
            try:
                import aiohttp as _aio
                async with self._context._session.get(
                    f"{self._context.config.url}/stats/frequent",
                    params={"days": 7},
                    timeout=_aio.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._frequent_commands = {
                            item["stt_text"]
                            for item in data[:self.config.brief_top_n]
                            if item.get("count", 0) >= self.config.brief_min_frequency
                            and item.get("intent_route") in (
                                "ha_device", "ha_device_fast", "ha_entity_fuzzy", "ha",
                            )
                        }
                        if self._frequent_commands:
                            log.info("brief mode: %d frequent commands", len(self._frequent_commands))
            except Exception as e:
                log.debug("brief mode sync failed: %s", e)

    async def _duck_avr(self) -> None:
        """Mute AVR input during voice interaction."""
        if not self.config.avr_enabled or not self.config.avr_device_name:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "wpctl", "set-mute", self.config.avr_device_name, "1",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as e:
            log.debug("AVR duck failed: %s", e)

    async def _restore_avr(self) -> None:
        """Unmute AVR input after voice interaction."""
        if not self.config.avr_enabled or not self.config.avr_device_name:
            return
        await asyncio.sleep(self.config.avr_restore_delay)
        try:
            proc = await asyncio.create_subprocess_exec(
                "wpctl", "set-mute", self.config.avr_device_name, "0",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as e:
            log.debug("AVR restore failed: %s", e)

    # -- Volume adaptation ---------------------------------------------------

    def _apply_volume_adaptation(self) -> None:
        """Scale playback volume based on how loud the user spoke."""
        if not self.config.volume_adapt:
            return
        rms = self._speech_rms if self._speech_rms > 0 else self._last_wake_rms
        if rms <= 0:
            return
        lo = self.config.volume_adapt_rms_low
        hi = self.config.volume_adapt_rms_high
        t = max(0.0, min(1.0, (rms - lo) / max(hi - lo, 1)))
        vol = self.config.volume_adapt_min + t * (self.config.volume_adapt_max - self.config.volume_adapt_min)
        log.info("Volume adapt: RMS=%.0f → volume=%.2f", rms, vol)
        self._playback.set_volume(vol)

    # -- Echo gate -----------------------------------------------------------

    def _is_echo_active(self) -> bool:
        """True if we're within the echo tail window after playback."""
        if self._responding:
            return True
        if self._last_playback_end == 0.0:
            return False
        elapsed_ms = (time.monotonic() - self._last_playback_end) * 1000
        return elapsed_ms < self.config.echo_gate_ms

    # -- helpers -------------------------------------------------------------

    async def _feed_wake_audio(self) -> None:
        """Feed capture chunks to wake word queue.
        Suppresses during echo tail (post-playback) but NOT during playback
        itself — barge-in needs to hear the wake word during TTS."""
        async for chunk in self._capture.chunks():
            # Only suppress during echo tail AFTER playback ends (not during)
            if not self._responding and self._is_echo_active():
                continue
            raw = chunk.samples.astype(np.int16).tobytes()
            try:
                self._wake_audio_queue.put_nowait(raw)
            except asyncio.QueueFull:
                pass

    def _load_pifi_modules(self) -> None:
        """Import intent classifier and managers from pi-fi-software."""
        import sys
        pifi_path = self.config.pifi_path
        if pifi_path not in sys.path:
            sys.path.insert(0, pifi_path)

        try:
            from intent_classifier import IntentClassifier
            self._classifier = IntentClassifier()
            log.info("intent classifier loaded")
        except ImportError as e:
            log.warning("intent classifier unavailable: %s", e)

        try:
            from timer_manager import TimerManager
            self._timer_mgr = TimerManager()
            log.info("timer manager loaded")
        except ImportError as e:
            log.debug("timer manager unavailable: %s", e)

        try:
            from reminder_manager import ReminderManager
            self._reminder_mgr = ReminderManager()
            log.info("reminder manager loaded")
        except ImportError as e:
            log.debug("reminder manager unavailable: %s", e)

        try:
            from alarm_manager import AlarmManager
            self._alarm_mgr = AlarmManager()
            log.info("alarm manager loaded")
        except ImportError as e:
            log.debug("alarm manager unavailable: %s", e)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_wake_sound(pifi_path: str) -> np.ndarray:
    """Load wake.wav from pi-fi sounds directory, fallback to synthetic beep."""
    import wave
    wav_path = Path(pifi_path) / "sounds" / "wake.wav"
    try:
        with wave.open(str(wav_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            log.info("loaded wake sound: %s (%d samples, %dHz)",
                     wav_path, len(audio), wf.getframerate())
            return audio
    except Exception as e:
        log.warning("wake.wav not found (%s), using synthetic beep", e)
        rate = 22050
        t = np.linspace(0, 0.15, int(rate * 0.15), endpoint=False)
        w = 0.7 * np.sin(2 * np.pi * 600 * t)
        fade = int(rate * 0.010)
        w[:fade] *= np.linspace(0, 1, fade)
        w[-fade:] *= np.linspace(1, 0, fade)
        return (w * 32767).astype(np.int16)
