"""Phase 2 Proof: wake word detection → listen → VAD end-of-speech.

Proves the state machine core works:
  IDLE (streaming to openWakeWord) → wake detected → beep →
  LISTENING (VAD) → end of speech → report duration & RMS

Run on Pi 5:
    /home/chaos/pi-fi-software/voice/.venv/bin/python tests/phase2_proof.py

Say the wake word, then speak a sentence. The script will detect end-of-speech
and print timing info. Ctrl+C to exit.
"""

import asyncio
import logging
import sys
import time

import numpy as np

# Add src to path for imports
sys.path.insert(0, "/home/chaos/chaosvector-audio/src")

from chaosvector_audio.capture import CaptureConfig, CaptureManager, AudioChunk
from chaosvector_audio.playback import PlaybackConfig, PlaybackManager, PlaybackPriority
from chaosvector_audio.vad import VADConfig, VoiceActivityDetector
from chaosvector_audio.wake import WakeConfig, WakeWordClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("phase2")


def generate_beep(freq: float = 880, duration_ms: int = 80, rate: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration_ms / 1000, int(rate * duration_ms / 1000), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    fade = int(rate * 0.005)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return (wave * 32767).astype(np.int16)


async def main() -> None:
    print("=== ChaosVector Audio — Phase 2 Proof ===")
    print("State machine: IDLE → wake → LISTENING → end-of-speech")
    print("Say your wake word, then speak. Ctrl+C to exit.\n")

    # --- Components ---
    capture = CaptureManager(CaptureConfig(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=20,  # 640 bytes per chunk, matches Wyoming expectations
        pre_roll_ms=500,
    ))

    playback = PlaybackManager(PlaybackConfig(
        sample_rate=22050,
        channels=1,
    ))

    vad = VoiceActivityDetector(VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        silence_frames_threshold=20,  # 400ms silence = end of speech
        min_speech_frames=3,
    ))

    wake = WakeWordClient(WakeConfig(
        host="127.0.0.1",
        port=10400,
        names=["hey_jarvis"],
        energy_threshold=200.0,  # lower threshold for testing
        gain=1.0,
    ))

    # Queue to feed audio bytes to wake word client
    wake_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

    # --- Start components ---
    await capture.open()
    await playback.start()
    await wake.start(wake_audio_queue)

    beep = generate_beep()
    interaction_count = 0

    try:
        while True:
            # === IDLE: stream audio to wake word detector ===
            log.info("=== IDLE — waiting for wake word ===")
            vad.reset()

            # Start feeding audio to wake in background
            feed_task = asyncio.create_task(
                _feed_wake_audio(capture, wake_audio_queue)
            )

            try:
                # Wait for wake word
                name, rms = await wake.wait_for_wake()
            finally:
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

            interaction_count += 1
            log.info("WAKE #%d: '%s' (rms=%.1f)", interaction_count, name, rms)

            # Play wake beep
            await playback.enqueue(
                beep, sample_rate=22050, priority=PlaybackPriority.WAKE_BEEP, label="wake-beep"
            )
            # Brief pause for beep to play
            await asyncio.sleep(0.15)

            # === LISTENING: collect utterance via VAD ===
            log.info("=== LISTENING — speak now ===")
            pre_roll = capture.drain_pre_roll()
            utterance_chunks: list[AudioChunk] = list(pre_roll)
            listen_start = time.monotonic()

            # Blanking: skip first 100ms after beep to avoid chime in utterance
            blanking_chunks = int(0.1 / 0.020)  # 5 chunks
            blanked = 0

            async for chunk in capture.chunks():
                if blanked < blanking_chunks:
                    blanked += 1
                    continue

                utterance_chunks.append(chunk)
                _, end_of_speech = vad.process_frame(chunk.samples)

                if end_of_speech:
                    break

                # Safety timeout: 10s max listen
                if time.monotonic() - listen_start > 10.0:
                    log.warning("listen timeout (10s)")
                    break

            listen_duration = time.monotonic() - listen_start
            total_samples = sum(len(c.samples) for c in utterance_chunks)
            audio_duration = total_samples / 16000

            log.info(
                "=== END OF SPEECH — %d chunks, %.2fs audio, %.2fs wall ===",
                len(utterance_chunks), audio_duration, listen_duration,
            )

            # Compute overall utterance RMS
            all_samples = np.concatenate([c.samples for c in utterance_chunks])
            utt_rms = np.sqrt(np.mean((all_samples.astype(np.float64) / 32768.0) ** 2))
            log.info("Utterance RMS=%.4f, samples=%d", utt_rms, len(all_samples))

            # Force fresh wake word connection (prevents stale TCP state)
            wake.force_reconnect()

            # Brief pause before next cycle
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await wake.stop()
        await playback.stop()
        await capture.close()

    print(f"\n=== Phase 2 complete — {interaction_count} interactions ===")


async def _feed_wake_audio(
    capture: CaptureManager, queue: asyncio.Queue
) -> None:
    """Feed capture audio chunks as raw bytes into the wake word queue."""
    count = 0
    async for chunk in capture.chunks():
        raw = chunk.samples.astype(np.int16).tobytes()
        count += 1
        if count % 50 == 1:  # log every 1s
            log.debug("feed_wake: chunk #%d, len=%d, rms=%.4f", count, len(raw), chunk.rms)
        try:
            queue.put_nowait(raw)
        except asyncio.QueueFull:
            pass  # drop if wake client is behind


if __name__ == "__main__":
    asyncio.run(main())
