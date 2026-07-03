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
from chaosvector_audio.stt_streaming import StreamingSTTConfig, StreamingSTTSession
from chaosvector_audio.family_knowledge import answer_family_question
from chaosvector_audio.wake_shadow import ShadowWakeDetector, ShadowWakeConfig
from chaosvector_audio.tts import TTSConfig, synthesize
from chaosvector_audio.llm import LLMConfig, LLMClient
from chaosvector_audio.context import ContextConfig, ContextClient, get_local_time
from chaosvector_audio.ha import HAConfig, HAClient
from chaosvector_audio.feedback import FeedbackLogger
from chaosvector_audio.speaker import SpeakerConfig, identify_speaker
from chaosvector_audio.stt_filters import correct_stt, is_stt_garbage
from chaosvector_audio.sounds import ThinkingIndicator, load_sound
from chaosvector_audio.tts_cache import TTSCache, prewarm_cache
from chaosvector_audio.health import HealthReporter, HealthStatus
from chaosvector_audio.wake_verify import WakeVerifier
from chaosvector_audio.stt_fast import FastSTTConfig, transcribe_fast

log = logging.getLogger(__name__)

# Conversation history timeout (clear LLM history after this many seconds idle)
_CONVERSATION_TIMEOUT = 1800  # 30 minutes


# Device command detection (same regex as satellite.py)
_DEVICE_CMD_RE = re.compile(
    r"\b(?:turn\s+(?:on|off)|switch\s+(?:on|off)|toggle"
    r"|(?:open|close|lock|unlock)\s+the"
    r"|dim\s+(?:the\s+)?\w+|brighten\s+(?:the\s+)?\w+"
    r"|set\s+(?:the\s+)?(?:thermostat|temperature|temp)\b"
    r"|(?:turn|switch)\s+(?:the\s+)?(?:heat|heating|cool(?:ing)?|ac|air)\s+(?:on|off)"
    r"|\b\d+\s*%)",  # "to 20%" anywhere in the command
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
    # Must match a device keyword AND an action keyword
    if not _HA_CANDIDATE_RE.search(text):
        return False
    # Require an action verb too — prevents "speed of light" matching
    return bool(_DEVICE_CMD_RE.search(text))


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
    ollama_url: str = "http://10.1.1.104:11434"
    ollama_model: str = "gemma4-12b-jarvis"
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

    # Wake verifier
    wake_verifier_path: str = "/home/chaos/chaosvector-audio/model/wake_verifier.pkl"
    wake_verifier_threshold: float = 0.5

    # Speech-to-Phrase fast path
    fast_stt_host: str = "10.1.1.53"
    fast_stt_port: int = 10302
    fast_stt_enabled: bool = True

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
            model=config.ollama_model,
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

        # TTS cache
        self._tts_cache = TTSCache(max_size=300)

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

        # Conversation tracking
        self._last_interaction_time: float = 0.0
        self._last_response_text: str = ""  # for adaptive follow-up timeout
        self._recent_interactions: list[tuple[str, str]] = []  # (transcript, response) last 3
        self._last_entities: list[str] = []  # entity names for pronoun resolution
        self._last_entity_ts: float = 0.0

        # Fast STT (Speech-to-Phrase)
        self._fast_stt_config = FastSTTConfig(
            host=config.fast_stt_host,
            port=config.fast_stt_port,
            enabled=config.fast_stt_enabled,
        )

        # Wake verifier (speaker-specific filter)
        self._wake_verifier = WakeVerifier(
            config.wake_verifier_path, config.wake_verifier_threshold,
        )

        # Health reporter
        self._health = HealthReporter(
            ha_url=config.ha_http_url, ha_token=config.ha_token,
        )
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

        # Pre-warm TTS cache (runs in background)
        asyncio.create_task(prewarm_cache(self._tts_cache, synthesize, self._tts_config))

        # Pre-synthesize common family knowledge answers (static, never change)
        asyncio.create_task(self._prewarm_static_answers())

        # Recurring precache: pull predicted answers from context engine and pre-synthesize TTS
        asyncio.create_task(self._precache_loop())

        # Health reporting to HA
        await self._health.start(self._get_health_status)

        # Start shadow wake detector (runs our ONNX model in parallel for comparison)
        self._shadow_wake = ShadowWakeDetector(ShadowWakeConfig(
            model_path="/home/chaos/chaosvector-audio/model/chaosvector-wake.onnx",
            threshold=0.7,
            trigger_level=2,
            energy_threshold=self.config.wake_energy_threshold,
        ))
        shadow_queue = asyncio.Queue(maxsize=200)
        self._shadow_audio_queue = shadow_queue
        asyncio.create_task(self._shadow_wake.start(shadow_queue))

        self._running = True
        log.info("orchestrator started")

    async def _prewarm_static_answers(self) -> None:
        """Pre-synthesize static answers that never change (family knowledge, etc)."""
        from chaosvector_audio.tts import synthesize
        static_answers = [
            "Sam is the oldest at 16.",
            "Eli is the youngest at 10.",
            "There are five kids: Sam, Zoey, Kinzleigh, Lexi, and Eli.",
            "The kids are Sam, Zoey, Kinzleigh, Lexi, and Eli.",
            "The family dog is Honey.",
            "Miles is the dad. He's 34 and into tech, cars, and alternative rock.",
            "Jennie is the mom. She's 37 and into reading and movies.",
        ]
        count = 0
        for text in static_answers:
            if not self._tts_cache.get(text):
                try:
                    result = await synthesize(text, self._tts_config)
                    if result and result.audio is not None:
                        self._tts_cache.put(text, result.audio, result.sample_rate, result.channels, result.duration_ms)
                        count += 1
                except Exception as e:
                    log.debug("static precache failed: %s", e)
        if count:
            log.info("static precache: %d family answers synthesized", count)

    async def _precache_loop(self) -> None:
        """Periodically fetch predicted answers from context engine and pre-synthesize TTS."""
        import aiohttp
        await asyncio.sleep(10)  # initial delay — let services settle
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._context.config.url}/precache",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            entries = await resp.json()
                            synthesized = 0
                            for entry in entries:
                                text = entry.get("answer", "")
                                if not text:
                                    continue
                                if not self._tts_cache.get(text):
                                    try:
                                        from chaosvector_audio.tts import synthesize
                                        result = await synthesize(text, self._tts_config)
                                        if result and result.audio is not None:
                                            self._tts_cache.put(
                                                text, result.audio, result.sample_rate,
                                                result.channels, result.duration_ms,
                                            )
                                            synthesized += 1
                                    except Exception as e:
                                        log.debug("precache synth failed: %s", e)
                            if synthesized:
                                log.info("precache: %d new entries synthesized (total %d)",
                                         synthesized, self._tts_cache.size)
                                self._tts_cache.save_to_disk()
            except Exception as e:
                log.debug("precache loop error: %s", e)
            await asyncio.sleep(60)  # refresh every 60 seconds

    async def stop(self) -> None:
        self._running = False
        # Save TTS cache to disk before shutdown
        self._tts_cache.save_to_disk()
        await self._health.stop()
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

        # Clear conversation history after idle timeout
        if (self._last_interaction_time > 0
                and time.monotonic() - self._last_interaction_time > _CONVERSATION_TIMEOUT):
            self._llm.clear_history()
            self._recent_interactions.clear()
            log.info("conversation history cleared (idle >30min)")

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

        self._last_wake_rms = rms
        log.info("WAKE candidate: '%s' (rms=%.1f)", name, rms)

        # Verify wake word using pre-roll (contains the wake audio) + 1s more
        if self._wake_verifier.is_available:
            pre_roll = self._capture.drain_pre_roll()
            # Also capture 1 more second to get the full wake word
            extra_chunks = []
            extra_start = time.monotonic()
            async for chunk in self._capture.chunks():
                extra_chunks.append(chunk)
                if time.monotonic() - extra_start > 1.0:
                    break
            all_chunks = (pre_roll or []) + extra_chunks
            if all_chunks:
                verify_audio = np.concatenate([c.samples for c in all_chunks])
                accepted, score = self._wake_verifier.verify(verify_audio)
                if not accepted:
                    log.info("wake rejected by verifier (score=%.3f)", score)
                    # Re-push chunks for pre-roll
                    return
                # Re-push extra chunks into pre-roll for STT use
                for c in extra_chunks:
                    self._capture._pre_roll.push(c)

        self._interaction_count += 1
        log.info("WAKE #%d: '%s' (rms=%.1f)", self._interaction_count, name, rms)

        # Save wake audio for future verifier training
        self._save_wake_audio()

        # Barge-in: stop any current playback
        if self._playback.is_playing:
            log.info("barge-in: stopping playback")
            self._playback.barge_in()

        # Play wake sound at low volume (15%, matches satellite.py)
        # Don't wait — play in background while listening starts
        quiet_beep = (self._beep.astype(np.float64) * 0.15).astype(np.int16)
        # Mute shadow during beep + room reverb
        if hasattr(self, '_shadow_wake'):
            self._shadow_wake.mute(1.5)
        await self._playback.enqueue(
            quiet_beep, sample_rate=self._beep_rate,
            priority=PlaybackPriority.WAKE_BEEP, label="wake-beep",
        )

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
        # Adaptive timeout: short after confirmations, longer after detailed answers
        while wants_followup and self._running:
            followup_timeout = self._adaptive_followup_timeout()
            log.info("=== FOLLOW-UP (%.0fs window) ===", followup_timeout)
            # Brief pause for echo gate after TTS
            await asyncio.sleep(0.3)

            # Listen with adaptive follow-up timeout
            utterance = await self._listen_followup(timeout=followup_timeout)
            if not utterance:
                log.info("follow-up: no speech, returning to IDLE")
                break

            transcript = await self._process_stt(utterance)
            if not transcript:
                break

            # If user said the wake word again, treat as new interaction
            if re.search(r"\bhey\s+jarvis\b", transcript, re.I):
                log.info("follow-up: wake word detected, treating as new command")
                transcript = re.sub(r"\bhey\s+jarvis\b[,.]?\s*", "", transcript, flags=re.I).strip()
                if not transcript:
                    break  # just said "hey Jarvis" with no command

            wants_followup = await self._respond(transcript)
            self._last_playback_end = time.monotonic()

        # Clean up
        self._wake.force_reconnect()
        await asyncio.sleep(0.3)

    # -- LISTENING -----------------------------------------------------------

    async def _listen(self) -> list[AudioChunk] | None:
        """Collect utterance via VAD while streaming to STT in real-time.

        Opens the STT connection at the start and sends audio chunks as they
        arrive. When VAD detects end-of-speech, the transcript is ready almost
        instantly since the server already has all the audio.
        """
        log.info("=== LISTENING ===")

        # Start streaming STT session
        self._streaming_stt = StreamingSTTSession(StreamingSTTConfig(
            host=self._stt_config.host,
            port=self._stt_config.port,
            language=self._stt_config.language,
            timeout=self._stt_config.timeout,
        ))
        stt_connected = await self._streaming_stt.start()
        if not stt_connected:
            log.warning("Streaming STT failed to connect, will fall back to buffered")

        # Include late pre-roll (speech that started during wake word)
        # but skip the first 500ms (the wake word itself)
        pre_roll = self._capture.drain_pre_roll()
        utterance: list[AudioChunk] = []
        if pre_roll:
            skip_chunks = int(500 / self.config.chunk_ms)  # skip ~500ms
            late_chunks = pre_roll[skip_chunks:]
            if late_chunks:
                utterance.extend(late_chunks)
                # Send pre-roll to streaming STT
                if stt_connected:
                    for c in late_chunks:
                        await self._streaming_stt.send_chunk(c)

        listen_start = time.monotonic()

        blanking_chunks = int(self.config.chime_blanking_ms / self.config.chunk_ms)
        blanked = 0

        min_listen_s = 1.0  # don't accept end-of-speech before 1s

        async for chunk in self._capture.chunks():
            if blanked < blanking_chunks:
                blanked += 1
                continue
            utterance.append(chunk)
            # Stream chunk to STT in real-time
            if stt_connected:
                await self._streaming_stt.send_chunk(chunk)
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

    async def _listen_followup(self, timeout: float | None = None) -> list[AudioChunk] | None:
        """Listen for follow-up speech without wake word.

        Simple approach: wait for playback to end, then use normal VAD
        listening with the standard _listen() method but with a shorter
        timeout. The VAD handles speech detection the same way as after
        a wake word.
        """
        # Wait for playback + echo to fully clear
        while self._playback.is_playing:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)  # XVF3800 hardware AEC handles echo, just clear reverb tail

        # Drain any stale audio from the capture queue
        while True:
            try:
                import queue as _q
                self._capture._thread_queue.get_nowait()
            except _q.Empty:
                break

        # Start streaming STT session for follow-up (same as _listen)
        self._streaming_stt = StreamingSTTSession(StreamingSTTConfig(
            host=self._stt_config.host,
            port=self._stt_config.port,
            language=self._stt_config.language,
            timeout=self._stt_config.timeout,
        ))
        stt_connected = await self._streaming_stt.start()
        if not stt_connected:
            log.warning("Follow-up streaming STT failed, will fall back to buffered")

        # Use normal listen with follow-up timeout
        fu_timeout = timeout or self.config.follow_up_timeout
        log.info("follow-up: listening for %.0fs...", fu_timeout)
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
                    if stt_connected:
                        await self._streaming_stt.send_chunk(chunk)
                elif elapsed > fu_timeout:
                    return None  # timed out waiting for speech
            else:
                utterance.append(chunk)
                if stt_connected:
                    await self._streaming_stt.send_chunk(chunk)
                if end_of_speech and elapsed > 1.5:  # min 1.5s after speech starts
                    break
                if elapsed > fu_timeout + self.config.listen_timeout:
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
        """Get transcript from streaming STT (already has audio) + S2P in parallel."""
        log.info("=== STT ===")
        self._stt_start = time.monotonic()

        # Streaming STT: just call finish() — server already has all audio
        if hasattr(self, "_streaming_stt") and self._streaming_stt and self._streaming_stt._connected:
            fast_task = asyncio.create_task(transcribe_fast(chunks, self._fast_stt_config))
            full_task = asyncio.create_task(self._streaming_stt.finish())
        else:
            # Fallback: buffered send (streaming connection failed)
            log.info("Falling back to buffered STT")
            fast_task = asyncio.create_task(transcribe_fast(chunks, self._fast_stt_config))
            full_task = asyncio.create_task(transcribe(chunks, self._stt_config))

        # Wait for both to complete (both are fast, <1s typically)
        fast_result, full_result = await asyncio.gather(fast_task, full_task)

        self._stt_ms = (time.monotonic() - self._stt_start) * 1000

        # Prefer S2P if it matched (exact entity names, no garbling)
        if fast_result:
            log.info("STT parallel: S2P matched \"%s\", full got \"%s\" (%.0fms)",
                     fast_result, full_result or "", self._stt_ms)
            speaker_audio = np.concatenate([c.samples for c in chunks[:150]])
            asyncio.create_task(self._identify_speaker(speaker_audio))
            return fast_result

        # S2P didn't match — use full STT result
        transcript = full_result
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
            # Resolve pronouns ("turn it off" → "turn off the office lights")
            transcript = self._resolve_pronouns(transcript)

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

            # Compound commands: split "turn on lights and play music"
            if self._classifier is not None and intents and len(intents) > 1:
                log.info("Compound command: %d intents", len(intents))
                for ci in intents:
                    await self._respond(ci.text)
                route = "compound"
                return False

            # Route based on intent type
            if intent_type == "timer":
                response_text = await self._handle_timer(intents[0])
                route = "timer"
            elif intent_type == "reminder":
                response_text = await self._handle_reminder(intents[0])
                route = "reminder"
            elif intent_type == "alarm":
                response_text = await self._handle_alarm(intents[0])
                route = "alarm"
            elif intent_type == "simple_local":
                response_text = await self._handle_simple_local(transcript, context_query)
                route = f"context:{context_query}" if context_query else "local"
            elif intent_type == "general" and _DEVICE_CMD_RE.search(transcript):
                response_text = await self._handle_ha_device(transcript)
                route = "ha"
                self._track_entities(transcript)
            elif intent_type == "general":
                # Try family knowledge first — instant answers for static facts
                family_answer = answer_family_question(transcript)
                if family_answer:
                    log.info("Family knowledge: %s", family_answer)
                    await self._speak(family_answer)
                    response_text = family_answer
                    route = "family_knowledge"
                else:
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

        except Exception as e:
            log.error("respond error: %s", e)
            await self._play_error_sound()

        finally:
            self._responding = False
            # Restore normal volume and AVR
            self._playback.set_volume(self._playback.config.volume)
            await self._restore_avr()

            # Track interaction
            self._last_interaction_time = time.monotonic()
            self._last_response_text = response_text or ""
            if response_text:
                self._recent_interactions.append((transcript, response_text))
                if len(self._recent_interactions) > 3:
                    self._recent_interactions.pop(0)

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

        # Humidity — pull directly from HA weather entity
        if "humid" in transcript.lower():
            humidity = await self._get_humidity()
            if humidity is not None:
                response = f"The humidity is {humidity} percent."
                log.info("Humidity: %s", response)
                await self._speak(response)
                return response

        # Everything else: ask context engine
        if context_query and self._context.is_available:
            answer = await self._context.get_answer(context_query)
            if answer:
                # Clean up weather text (context engine quirks)
                if context_query in ("weather", "forecast"):
                    answer = answer.replace("partlycloudy", "partly cloudy")
                    answer = answer.replace("mostlycloudy", "mostly cloudy")
                    answer = answer.replace("mostlysunny", "mostly sunny")
                    answer = answer.replace("partlysunny", "partly sunny")

                # If user asked about tomorrow, strip the "today" portion
                if context_query == "forecast" and "tomorrow" in transcript.lower():
                    import re as _re
                    tomorrow_match = _re.search(r'(Tomorrow\b.+)', answer, _re.IGNORECASE)
                    if tomorrow_match:
                        answer = tomorrow_match.group(1)

                # If user asked about a specific person, extract just their info
                # and resolve "away" to a city name via GPS reverse geocoding
                if context_query == "presence":
                    import re as _re
                    name_match = _re.search(r"where(?:'?s|\s+is)\s+(\w+)", transcript, _re.IGNORECASE)
                    person = name_match.group(1) if name_match else None
                    # Filter out non-name words (the, everyone, everybody, etc.)
                    non_names = {"the", "everyone", "everybody", "all", "they", "people", "family"}
                    if person and person.lower() in non_names:
                        person = None

                    if person:
                        # Name aliases (STT output → context engine name)
                        _name_aliases = {
                            "kinzleigh": "kinz", "kinsley": "kinz", "kenzie": "kinz",
                        }
                        search_names = [person.lower()]
                        alias = _name_aliases.get(person.lower())
                        if alias:
                            search_names.append(alias)

                        # Find the sentence about this specific person
                        person_answer = None
                        for sentence in answer.split(". "):
                            if any(n in sentence.lower() for n in search_names):
                                person_answer = sentence.strip().rstrip(".") + "."
                                break
                        if person_answer:
                            answer = person_answer
                        # If "away", try to get city from GPS
                        if "away" in answer.lower():
                            # Use HA entity name (alias if available)
                            ha_name = alias if alias else person.lower()
                            city = await self._get_person_city(ha_name)
                            if city:
                                answer = answer.replace("is away", f"is in {city}").replace("is Away", f"is in {city}")
                    else:
                        # General "where is everyone" — resolve ALL away people
                        if "away" in answer.lower():
                            for sentence in answer.split(". "):
                                away_match = _re.search(r"(\w+)\s+is\s+away", sentence, _re.IGNORECASE)
                                if away_match:
                                    away_name = away_match.group(1)
                                    city = await self._get_person_city(away_name)
                                    if city:
                                        answer = answer.replace(f"{away_name} is away", f"{away_name} is in {city}")
                                        answer = answer.replace(f"{away_name} is Away", f"{away_name} is in {city}")

                log.info("Context answer (%s): %s", context_query, answer[:80])
                await self._speak(answer)
                return answer

        # Context engine write operations
        if self._context.is_available:
            text_lower = transcript.lower()

            # Shopping list
            if "shopping list" in text_lower or "grocery" in text_lower:
                if "add" in text_lower:
                    item = re.sub(r".*(?:add|put)\s+", "", transcript, flags=re.I).strip()
                    item = re.sub(r"\s+(?:to|on)\s+(?:the\s+)?(?:shopping|grocery).*", "", item, flags=re.I).strip()
                    if item:
                        result = await self._context_write("shopping/add", {"item": item})
                        if result:
                            await self._speak(result)
                            return result

            # Todo list
            if "to do" in text_lower or "todo" in text_lower:
                if "add" in text_lower:
                    item = re.sub(r".*(?:add)\s+", "", transcript, flags=re.I).strip()
                    item = re.sub(r"\s+(?:to|on)\s+(?:the\s+)?(?:to.?do).*", "", item, flags=re.I).strip()
                    if item:
                        result = await self._context_write("todo/add", {"item": item})
                        if result:
                            await self._speak(result)
                            return result

            # Memory ("remember that...")
            if re.match(r"remember\s+(?:that\s+)?", text_lower):
                fact = re.sub(r"^remember\s+(?:that\s+)?", "", transcript, flags=re.I).strip()
                if fact:
                    result = await self._context_write("memory/add", {"fact": fact})
                    if result:
                        await self._speak(result)
                        return result

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

        # Rewrite "dim X to Y%" → "set X brightness to Y percent"
        # HA Assist doesn't understand "dim" but understands "set brightness"
        dim_match = re.match(
            r"dim\s+(?:the\s+)?(.+?)\s+to\s+(\d+)\s*%?\.?$", transcript, re.I
        )
        if dim_match:
            device, level = dim_match.group(1), dim_match.group(2)
            transcript = f"set {device} brightness to {level} percent"
            log.info("Rewritten dim command: \"%s\"", transcript)

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

                # Recent interactions for conversational context
                if self._recent_interactions:
                    for user_text, resp_text in self._recent_interactions[-2:]:
                        parts.append(f"[Recent] User: {user_text[:60]} → {resp_text[:60]}")

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

                # Double-check for corruption (LLM filter may miss partial tokens)
                from chaosvector_audio.llm import _is_corrupted
                if _is_corrupted(sentence):
                    log.warning("corruption caught at orchestrator, aborting response")
                    break

                # Check TTS cache first
                cached = self._tts_cache.get(sentence)
                if cached:
                    log.info("TTS cache hit: \"%s\"", sentence[:40])
                    result = type('R', (), {
                        'audio': cached.audio, 'sample_rate': cached.sample_rate,
                        'channels': cached.channels, 'duration_ms': cached.duration_ms,
                    })()
                else:
                    result = await synthesize(sentence, self._tts_config)
                    if result is not None:
                        self._tts_cache.put(sentence, result.audio, result.sample_rate,
                                            result.channels, result.duration_ms)
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
        Checks TTS cache first for instant playback."""
        # Check cache first
        cached = self._tts_cache.get(text)
        if cached:
            log.info("TTS cache hit: \"%s\"", text[:60])
            result = type('R', (), {
                'audio': cached.audio, 'sample_rate': cached.sample_rate,
                'channels': cached.channels, 'duration_ms': cached.duration_ms,
            })()
        else:
            result = await synthesize(text, self._tts_config)
            # Store in cache for next time
            if result is not None:
                self._tts_cache.put(text, result.audio, result.sample_rate,
                                    result.channels, result.duration_ms)
        if result is not None:
            # Mute shadow wake detector during playback + 2s echo tail
            if hasattr(self, '_shadow_wake'):
                duration = (result.duration_ms / 1000.0) + 2.0
                self._shadow_wake.mute(duration)
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

    # -- Weather helpers -----------------------------------------------------

    async def _get_humidity(self) -> int | None:
        """Get current outdoor humidity from HA weather entity."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.ha_http_url}/api/states/weather.forecast_home",
                    headers={"Authorization": f"Bearer {self.config.ha_token}"},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("attributes", {}).get("humidity")
        except Exception as e:
            log.debug("humidity lookup failed: %s", e)
        return None

    # -- Location resolution -------------------------------------------------

    # Cache geocoding results for 10 minutes (people don't teleport between cities)
    _geocode_cache: dict[str, tuple[str, float]] = {}
    _GEOCODE_TTL = 600  # 10 minutes

    async def _get_person_city(self, name: str) -> str | None:
        """Get city name for a person who is 'away' via HA GPS + reverse geocoding."""
        # Check cache first
        cached = self._geocode_cache.get(name.lower())
        if cached and (time.monotonic() - cached[1]) < self._GEOCODE_TTL:
            return cached[0]

        try:
            # Look up person entity in HA
            person_id = f"person.{name.lower()}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.ha_http_url}/api/states/{person_id}",
                    headers={"Authorization": f"Bearer {self.config.ha_token}"},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            attrs = data.get("attributes", {})
            lat = attrs.get("latitude")
            lon = attrs.get("longitude")
            if lat is None or lon is None:
                return None

            # Reverse geocode via OpenStreetMap Nominatim (free, no API key)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
                    headers={"User-Agent": "ChaosVector-Audio/1.0"},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    if resp.status != 200:
                        return None
                    geo = await resp.json()

            address = geo.get("address", {})
            city = address.get("city") or address.get("town") or address.get("suburb") or address.get("village")
            if city:
                self._geocode_cache[name.lower()] = (city, time.monotonic())
            return city

        except Exception as e:
            log.debug("person city lookup failed: %s", e)
            return None

    # -- Wake audio collection -----------------------------------------------

    def _save_wake_audio(self) -> None:
        """Save pre-roll + 1s of post-wake audio for verifier training.
        Captures through the live pipeline path so embeddings match."""
        import wave as _wave
        from datetime import datetime

        try:
            save_dir = Path("/home/chaos/wake-training/live")
            save_dir.mkdir(parents=True, exist_ok=True)

            # Use pre-roll audio (contains the wake word)
            pre_roll = self._capture.drain_pre_roll()
            if not pre_roll:
                return

            audio = np.concatenate([c.samples for c in pre_roll])

            # Pad to at least 2.5s
            target = int(self.config.sample_rate * 2.5)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"wake_{ts}_{self._interaction_count:04d}.wav"
            with _wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio[:target].tobytes())

            log.debug("saved wake audio: %s", path.name)
        except Exception as e:
            log.debug("wake audio save failed: %s", e)

    # -- Pronoun resolution, entity tracking, error sound --------------------

    # Matches pronouns in device commands: "turn it off", "make it brighter", "set it to 50%"
    _PRONOUN_RE = re.compile(
        r"\b(it|that|them|those)\b",
        re.IGNORECASE,
    )

    # Relative brightness commands: "make it brighter/dimmer"
    _RELATIVE_CMD_RE = re.compile(
        r"(?:make|turn)\s+(?:it|that|them|those)\s+(brighter|dimmer|louder|quieter)",
        re.IGNORECASE,
    )

    def _resolve_pronouns(self, text: str) -> str:
        """Replace pronouns with last controlled entity if available."""
        if not self._last_entities:
            return text
        # Only resolve if entity context is fresh (<5 min)
        if time.monotonic() - self._last_entity_ts > 300:
            self._last_entities.clear()
            return text

        entity = self._last_entities[0]

        # Handle relative brightness: "make it brighter" → "set X brightness to Y%"
        rel_match = self._RELATIVE_CMD_RE.search(text)
        if rel_match:
            direction = rel_match.group(1).lower()
            if direction == "brighter":
                resolved = f"set {entity} brightness to 100 percent"
            elif direction == "dimmer":
                resolved = f"set {entity} brightness to 20 percent"
            else:
                resolved = text  # louder/quieter — pass through
            if resolved != text:
                log.info("relative command resolved: \"%s\" → \"%s\"", text, resolved)
                return resolved

        # General pronoun replacement in device commands
        if self._PRONOUN_RE.search(text):
            # Only replace if it looks like a device command
            if re.search(r"\b(?:turn|switch|toggle|set|dim|make)\b", text, re.I):
                resolved = self._PRONOUN_RE.sub(entity, text)
                log.info("pronoun resolved: \"%s\" → \"%s\"", text, resolved)
                return resolved
        return text

    def _adaptive_followup_timeout(self) -> float:
        """Adaptive follow-up duration based on response length.

        Short confirmations (Done, Got it) → 3s — user likely done
        Medium answers (weather, presence) → 5s — normal
        Long LLM responses → 8s — user needs time to process and ask follow-up
        """
        text = self._last_response_text
        if not text:
            return self.config.follow_up_timeout

        word_count = len(text.split())
        if word_count <= 5:
            return 3.0  # short confirmation
        elif word_count >= 30:
            return 8.0  # long detailed answer
        return self.config.follow_up_timeout  # default (5s)

    def _track_entities(self, transcript: str) -> None:
        """Extract entity names from device commands for pronoun resolution."""
        patterns = [
            r"(?:turn\s+(?:on|off)|toggle)\s+(?:the\s+)?(.+?)\.?$",
            r"(?:dim|set)\s+(?:the\s+)?(.+?)\s+(?:brightness|to)\s",
            r"(?:turn|switch)\s+(?:the\s+)?(.+?)\s+(?:on|off)",
        ]
        for pat in patterns:
            m = re.search(pat, transcript, re.I)
            if m:
                entity = m.group(1).strip()
                if entity.lower() not in ("it", "that", "them", "those"):
                    self._last_entities = [entity]
                    self._last_entity_ts = time.monotonic()
                    log.info("tracking entity: %s", entity)
                    return

    async def _play_error_sound(self) -> None:
        """Play error sound on failures."""
        sound = load_sound("error", self.config.sounds_dir)
        if sound is not None:
            audio, rate = sound
            await self._playback.enqueue(audio, sample_rate=rate,
                                         priority=PlaybackPriority.NOTIFICATION, label="error")
            await self._wait_playback(timeout=3.0)

    async def _context_write(self, endpoint: str, data: dict) -> str | None:
        """Write to context engine (shopping, todo, memory, preferences)."""
        if not self._context.is_available:
            return None
        try:
            async with self._context._session.post(
                f"{self._context.config.url}/{endpoint}",
                json=data,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("message", "Done.")
        except Exception as e:
            log.warning("context write (%s) failed: %s", endpoint, e)
        return None

    def _get_health_status(self) -> HealthStatus:
        """Return current health status for HA sensor."""
        return HealthStatus(
            wake_word=self._wake.is_connected,
            stt=True,  # per-request, always "available"
            tts=True,
            llm=self._llm.is_available,
            context_engine=self._context.is_available,
            ha=self._ha.is_available,
            speaker_verify=self._speaker_config.enabled,
            interactions=self._interaction_count,
            last_interaction=time.strftime("%H:%M:%S",
                time.localtime(self._last_interaction_time)) if self._last_interaction_time else "",
        )

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

    async def _handle_timer(self, intent) -> str:
        """Handle timer intents: set, check, cancel."""
        if self._timer_mgr is None:
            response = "Timers aren't available right now."
            await self._speak(response)
            return response

        action = intent.timer_action
        if action is None:
            response = "I didn't understand that timer command."
            await self._speak(response)
            return response

        action_val = action.value
        if action_val == "set":
            duration = intent.timer_duration_seconds or 60
            label = intent.timer_label
            response = await self._timer_mgr.set_timer(duration, label, self._on_timer_fire)
        elif action_val == "cancel":
            response = await self._timer_mgr.cancel_timer(intent.timer_label)
        elif action_val == "check":
            response = self._timer_mgr.check_timer(intent.timer_label) if hasattr(self._timer_mgr, "check_timer") else "No active timers."
        else:
            response = "I didn't understand that timer command."

        log.info("Timer: %s → %s", action_val, response)
        await self._speak(response)
        return response

    async def _handle_reminder(self, intent) -> str:
        """Handle reminder intents: set, check, cancel."""
        if self._reminder_mgr is None:
            response = "Reminders aren't available right now."
            await self._speak(response)
            return response

        action = intent.reminder_action
        if action is None:
            response = "I didn't understand that reminder command."
            await self._speak(response)
            return response

        action_val = action.value
        if action_val == "set":
            time_str = intent.reminder_time_str or "in 1 hour"
            label = intent.reminder_label or "reminder"
            response = await self._reminder_mgr.set_reminder(
                time_str, label, self._on_reminder_fire,
                recurring=getattr(intent, "reminder_recurring", False),
                recurring_days=getattr(intent, "reminder_recurring_days", None),
            )
        elif action_val == "cancel":
            response = await self._reminder_mgr.cancel_reminder(intent.reminder_label)
        elif action_val == "check":
            response = self._reminder_mgr.check_reminders() if hasattr(self._reminder_mgr, "check_reminders") else "No active reminders."
        else:
            response = "I didn't understand that reminder command."

        log.info("Reminder: %s → %s", action_val, response)
        await self._speak(response)
        return response

    async def _handle_alarm(self, intent) -> str:
        """Handle alarm intents: set, check, cancel, snooze."""
        if self._alarm_mgr is None:
            response = "Alarms aren't available right now."
            await self._speak(response)
            return response

        action = intent.alarm_action
        if action is None:
            response = "I didn't understand that alarm command."
            await self._speak(response)
            return response

        action_val = action.value
        if action_val == "set":
            time_str = intent.alarm_time_str or "in 1 hour"
            label = getattr(intent, "alarm_label", None)
            response = await self._alarm_mgr.set_alarm(time_str, label, self._on_alarm_fire)
        elif action_val == "cancel":
            label = getattr(intent, "alarm_label", None)
            response = await self._alarm_mgr.cancel_alarm(label)
        elif action_val == "snooze":
            response = await self._alarm_mgr.snooze() if hasattr(self._alarm_mgr, "snooze") else "No alarm to snooze."
        elif action_val == "check":
            response = self._alarm_mgr.check_alarms() if hasattr(self._alarm_mgr, "check_alarms") else "No active alarms."
        else:
            response = "I didn't understand that alarm command."

        log.info("Alarm: %s → %s", action_val, response)
        await self._speak(response)
        return response

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
        itself — barge-in needs to hear the wake word during TTS.
        VAD gate: only send audio that contains speech-like content.
        Rejects sniffles, clicks, and transient noise."""
        from chaosvector_audio.vad import VADConfig, VoiceActivityDetector, SpeechState
        wake_vad = VoiceActivityDetector(VADConfig(
            aggressiveness=3,       # max filtering
            sample_rate=self.config.sample_rate,
            frame_duration_ms=self.config.chunk_ms,
            silence_frames_threshold=5,   # quick reset
            min_speech_frames=3,    # need 3 consecutive speech frames (~60ms) to pass
        ))
        speech_frame_count = 0

        async for chunk in self._capture.chunks():
            # Only suppress during echo tail AFTER playback ends (not during)
            if not self._responding and self._is_echo_active():
                speech_frame_count = 0
                continue

            # VAD gate: only forward audio that looks like speech
            is_speech = wake_vad.is_speech(chunk.samples)
            if is_speech:
                speech_frame_count += 1
            else:
                # Allow a few silent frames through once speech started
                # (natural pauses in "hey... Jarvis")
                if speech_frame_count > 0:
                    speech_frame_count = max(0, speech_frame_count - 1)

            # Only send to wake detector if we've seen sustained speech
            # (3+ consecutive frames = ~60ms, filters sniffles/clicks)
            if speech_frame_count < 3:
                continue

            raw = chunk.samples.astype(np.int16).tobytes()
            try:
                self._wake_audio_queue.put_nowait(raw)
            except asyncio.QueueFull:
                pass
            # Feed shadow detector
            if hasattr(self, '_shadow_audio_queue'):
                try:
                    self._shadow_audio_queue.put_nowait(raw)
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
