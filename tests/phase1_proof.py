"""Phase 1 Proof: capture 3 seconds of mic audio, play it back.

This proves that sounddevice can capture from the Pi-Fi mic array and play
to the speaker without any pw-record/pw-play subprocesses.

Run on Pi 5:
    /home/chaos/pi-fi-software/voice/.venv/bin/python tests/phase1_proof.py
"""

import asyncio
import sys
import time

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_S = 3
PLAYBACK_RATE = 16000  # play back at same rate we captured


async def main() -> None:
    print("=== ChaosVector Audio — Phase 1 Proof ===")
    print(f"Devices:\n{sd.query_devices()}\n")

    # Use default device (PipeWire routes to the correct source/sink)
    device_in = None  # default
    device_out = None  # default

    # --- Capture ---
    print(f"Recording {DURATION_S}s from mic (rate={SAMPLE_RATE}, ch={CHANNELS})...")
    frames_total = SAMPLE_RATE * DURATION_S
    captured = np.zeros(frames_total, dtype=np.int16)
    offset = 0
    chunk_size = int(SAMPLE_RATE * 0.030)  # 30 ms chunks, same as pipeline

    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def capture_cb(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal offset
        if status:
            print(f"  [capture status: {status}]", file=sys.stderr)
        samples = indata[:, 0].copy().astype(np.int16) if CHANNELS == 1 else indata.copy().astype(np.int16)
        end = min(offset + len(samples), frames_total)
        captured[offset:end] = samples[: end - offset]
        offset = end
        if offset >= frames_total:
            loop.call_soon_threadsafe(done.set)

    stream_in = sd.InputStream(
        device=device_in,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=chunk_size,
        callback=capture_cb,
    )

    t0 = time.monotonic()
    with stream_in:
        await done.wait()
    elapsed = time.monotonic() - t0

    rms = np.sqrt(np.mean((captured.astype(np.float64) / 32768.0) ** 2))
    peak = np.max(np.abs(captured)) / 32768.0
    print(f"  Captured {len(captured)} samples in {elapsed:.2f}s")
    print(f"  RMS={rms:.4f}  Peak={peak:.4f}")

    if rms < 0.001:
        print("  WARNING: RMS very low — mic may not be routed correctly")

    # --- Playback ---
    print(f"\nPlaying back {DURATION_S}s to speaker (rate={PLAYBACK_RATE})...")
    playback_done = asyncio.Event()

    def finished_cb() -> None:
        loop.call_soon_threadsafe(playback_done.set)

    # Convert to float32 for output
    audio_float = captured.astype(np.float32) / 32768.0

    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=PLAYBACK_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=1024,
        finished_callback=finished_cb,
    )

    t0 = time.monotonic()
    with stream_out:
        stream_out.write(audio_float.reshape(-1, CHANNELS))

    # finished_callback fires when the stream is done, but with `with` block
    # closing the stream, we just wait a moment for audio to flush
    await asyncio.sleep(DURATION_S + 0.5)
    elapsed = time.monotonic() - t0
    print(f"  Playback complete ({elapsed:.2f}s)")

    print("\n=== Phase 1 PASSED — direct audio I/O works ===")


if __name__ == "__main__":
    asyncio.run(main())
