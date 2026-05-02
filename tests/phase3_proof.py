"""Phase 3 Proof: full voice loop — wake → STT → TTS → playback.

Proves the complete pipeline works end-to-end:
  IDLE → wake → LISTENING → VAD → STT (ChaosVector) → TTS (ChaosVector) → playback → IDLE

The "intent handler" here is a simple echo: it repeats back what you said.

Run on Pi 5:
    /home/chaos/pi-fi-software/voice/.venv/bin/python tests/phase3_proof.py

Say "hey Jarvis", speak a sentence, hear it echoed back. Ctrl+C to exit.
"""

import asyncio
import logging
import sys
import time

import numpy as np

sys.path.insert(0, "/home/chaos/chaosvector-audio/src")

from chaosvector_audio.capture import CaptureConfig, CaptureManager, AudioChunk
from chaosvector_audio.playback import PlaybackConfig, PlaybackManager, PlaybackPriority
from chaosvector_audio.vad import VADConfig, VoiceActivityDetector
from chaosvector_audio.wake import WakeConfig, WakeWordClient
from chaosvector_audio.stt import STTConfig, transcribe
from chaosvector_audio.tts import TTSConfig, synthesize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("phase3")


def generate_beep(freq: float = 880, duration_ms: int = 80, rate: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration_ms / 1000, int(rate * duration_ms / 1000), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    fade = int(rate * 0.005)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return (wave * 32767).astype(np.int16)


async def main() -> None:
    print("=== ChaosVector Audio — Phase 3 Proof ===")
    print("Full loop: wake → STT → TTS → playback")
    print("Say 'hey Jarvis', speak a sentence, hear it echoed back.")
    print("Ctrl+C to exit.\n")

    # --- Components ---
    capture = CaptureManager(CaptureConfig(
        sample_rate=16000, channels=1, chunk_duration_ms=20, pre_roll_ms=500,
    ))
    playback = PlaybackManager(PlaybackConfig(sample_rate=22050, channels=1))
    vad = VoiceActivityDetector(VADConfig(
        aggressiveness=2, sample_rate=16000, frame_duration_ms=20,
        silence_frames_threshold=20, min_speech_frames=3,
    ))
    wake = WakeWordClient(WakeConfig(
        host="127.0.0.1", port=10400, names=["hey_jarvis"],
        energy_threshold=200.0, gain=1.0,
    ))

    stt_config = STTConfig(host="10.1.1.240", port=10301)
    tts_config = TTSConfig(host="10.1.1.240", port=10210, voice="af_heart")

    wake_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

    # --- Start ---
    await capture.open()
    await playback.start()
    await wake.start(wake_audio_queue)

    beep = generate_beep()
    interaction_count = 0

    try:
        while True:
            # === IDLE ===
            log.info("=== IDLE — say 'hey Jarvis' ===")
            vad.reset()

            feed_task = asyncio.create_task(_feed_wake_audio(capture, wake_audio_queue))
            try:
                name, rms = await wake.wait_for_wake()
            finally:
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

            interaction_count += 1
            log.info("WAKE #%d: '%s' (rms=%.1f)", interaction_count, name, rms)

            # Wake beep
            await playback.enqueue(
                beep, sample_rate=22050, priority=PlaybackPriority.WAKE_BEEP, label="wake-beep"
            )
            await asyncio.sleep(0.15)

            # === LISTENING ===
            log.info("=== LISTENING ===")
            pre_roll = capture.drain_pre_roll()
            utterance_chunks: list[AudioChunk] = list(pre_roll)
            listen_start = time.monotonic()

            # Chime blanking (100ms)
            blanking_chunks = 5
            blanked = 0

            async for chunk in capture.chunks():
                if blanked < blanking_chunks:
                    blanked += 1
                    continue
                utterance_chunks.append(chunk)
                _, end_of_speech = vad.process_frame(chunk.samples)
                if end_of_speech:
                    break
                if time.monotonic() - listen_start > 10.0:
                    log.warning("listen timeout")
                    break

            listen_ms = (time.monotonic() - listen_start) * 1000
            log.info("LISTENING done: %d chunks, %.0fms", len(utterance_chunks), listen_ms)

            # === PROCESSING (STT) ===
            log.info("=== PROCESSING (STT) ===")
            transcript = await transcribe(utterance_chunks, stt_config)

            if not transcript or not transcript.strip():
                log.info("Empty transcript — back to IDLE")
                wake.force_reconnect()
                await asyncio.sleep(0.3)
                continue

            log.info("Transcript: \"%s\"", transcript)

            # === RESPONDING (TTS + playback) ===
            log.info("=== RESPONDING ===")
            response_text = f"You said: {transcript}"
            result = await synthesize(response_text, tts_config)

            if result is not None:
                await playback.enqueue(
                    result.audio,
                    sample_rate=result.sample_rate,
                    channels=result.channels,
                    priority=PlaybackPriority.TTS,
                    label="response",
                )
                # Wait for playback to finish
                while playback.is_playing:
                    await asyncio.sleep(0.05)
                log.info("Playback complete (%.0fms audio)", result.duration_ms)
            else:
                log.warning("TTS failed — no audio to play")

            # Force fresh wake connection
            wake.force_reconnect()
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await wake.stop()
        await playback.stop()
        await capture.close()

    print(f"\n=== Phase 3 complete — {interaction_count} interactions ===")


async def _feed_wake_audio(capture: CaptureManager, queue: asyncio.Queue) -> None:
    """Feed capture chunks as raw bytes into the wake word queue."""
    async for chunk in capture.chunks():
        raw = chunk.samples.astype(np.int16).tobytes()
        try:
            queue.put_nowait(raw)
        except asyncio.QueueFull:
            pass


if __name__ == "__main__":
    asyncio.run(main())
