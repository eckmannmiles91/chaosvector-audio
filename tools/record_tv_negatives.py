#!/usr/bin/env python3
"""Record negative samples for wake word training.

Captures continuous 2.5s clips from the mic and saves them to labeled
directories under /home/chaos/wake-training/.

Usage on Pi 5 (mic shared via PipeWire — no need to stop the pipeline):
    /home/chaos/pi-fi-software/voice/.venv/bin/python tools/record_tv_negatives.py --type ambient
    /home/chaos/pi-fi-software/voice/.venv/bin/python tools/record_tv_negatives.py --type tv

Output dirs:
    ambient → /home/chaos/wake-training/ambient_negatives/  (quiet room, house noise)
    tv      → /home/chaos/wake-training/tv_negatives/       (TV/music playing)

Stop with Ctrl+C. Progress is saved as you go.
"""

import argparse
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

# Per-type settings
TYPE_CONFIG = {
    "ambient": {
        "out_dir": "/home/chaos/wake-training/negative",
        "prefix": "negative",
        "rms_min": 0.0005,   # capture very quiet room sounds
        "rms_max": 0.08,     # skip anything that might be a voice command
        "description": "Quiet room / ambient house noise (no TV, no commands)",
    },
    "tv": {
        "out_dir": "/home/chaos/wake-training/negative_tv",
        "prefix": "tv",
        "rms_min": 0.003,    # skip near-silence
        "rms_max": None,     # no upper bound
        "description": "TV / music playing (anything that isn't a wake word)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Record negative training samples")
    parser.add_argument(
        "--type", choices=["ambient", "tv"], required=True,
        help="Sample type: 'ambient' for quiet room noise, 'tv' for TV/music audio",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    cfg = TYPE_CONFIG[args.type]
    out_dir = Path(args.output_dir or cfg["out_dir"])
    prefix = cfg["prefix"]
    rms_min = cfg["rms_min"]
    rms_max = cfg["rms_max"]

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    count = len(existing)

    print(f"\nRecording negative samples — type: {args.type}")
    print(f"  {cfg['description']}")
    print(f"  Output: {out_dir}")
    print(f"  Existing clips: {count}")
    print(f"  RMS gate: >{rms_min:.4f}" + (f" <{rms_max:.4f}" if rms_max else ""))
    print(f"  Press Ctrl+C to stop\n")

    clip_samples = int(SAMPLE_RATE * CLIP_DURATION)
    running = True
    skipped = 0
    saved = 0

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

            # RMS gates
            if rms < rms_min:
                skipped += 1
                continue
            if rms_max is not None and rms > rms_max:
                skipped += 1
                continue

            path = out_dir / f"{prefix}_{count:04d}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            count += 1
            saved += 1

            if saved % 10 == 0:
                print(f"  {count} clips total ({saved} this session, {skipped} skipped)"
                      f"  RMS={rms:.4f}", flush=True)

        except Exception as e:
            if running:
                print(f"  Error: {e}", file=sys.stderr)
                time.sleep(1)

    print(f"\nDone! {saved} new clips recorded (total: {count}, {skipped} skipped by RMS gate)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
