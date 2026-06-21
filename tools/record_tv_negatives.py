#!/usr/bin/env python3
"""Record TV audio for wake word negative training data.

Start this when watching TV, stop with Ctrl+C when done.
Saves 2.5s clips to /home/chaos/wake-training/tv_negatives/

Run on Pi 5:
    systemctl --user stop chaosvector-audio  # free the mic
    /home/chaos/pi-fi-software/voice/.venv/bin/python tools/record_tv_negatives.py
    systemctl --user start chaosvector-audio  # restart after

Or run alongside the pipeline (mic is shared via PipeWire):
    /home/chaos/pi-fi-software/voice/.venv/bin/python tools/record_tv_negatives.py &
"""

import signal
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
CLIP_DURATION = 2.5  # seconds, matches training format
OUT_DIR = Path("/home/chaos/wake-training/tv_negatives")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.wav")))

    print(f"Recording TV audio for negative training data")
    print(f"Output: {OUT_DIR}")
    print(f"Existing clips: {existing}")
    print(f"Press Ctrl+C to stop\n")

    clip_samples = int(SAMPLE_RATE * CLIP_DURATION)
    count = existing
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        try:
            audio = sd.rec(clip_samples, samplerate=SAMPLE_RATE,
                           channels=CHANNELS, dtype="int16")
            sd.wait()
            audio = audio.flatten()

            rms = np.sqrt(np.mean((audio.astype(np.float64) / 32768) ** 2))

            # Skip near-silence (no useful training data)
            if rms < 0.003:
                continue

            path = OUT_DIR / f"tv_neg_{count:04d}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            count += 1
            if count % 10 == 0:
                print(f"  {count} clips recorded (RMS={rms:.4f})", flush=True)

        except Exception as e:
            if running:
                print(f"  Error: {e}", file=sys.stderr)
                time.sleep(1)

    total_new = count - existing
    print(f"\nDone! {total_new} new clips recorded (total: {count})")


if __name__ == "__main__":
    main()
