#!/usr/bin/env python3
"""Wake word recording tool — collects "hey Jarvis" samples for training.

Run on Pi 5:
    /home/chaos/pi-fi-software/voice/.venv/bin/python tools/record_wake_words.py --speaker miles

Collects 30 recordings with varying distance/volume prompts.
Saves to /home/chaos/wake-training/<speaker>/ as labeled WAV files.
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# Recording config
SAMPLE_RATE = 16000
CHANNELS = 1
CLIP_DURATION = 2.5  # seconds per clip
PAUSE_BETWEEN = 1.5  # seconds between clips

# Prompt variations — 30 recordings total
PROMPTS = [
    # Normal distance, normal volume (10)
    ("normal voice, normal distance", 10),
    # Across the room (8)
    ("step back 6-8 feet away, normal voice", 8),
    # Quiet / whisper (5)
    ("normal distance, quiet voice", 5),
    # Loud / call out (5)
    ("normal distance, louder than normal", 5),
    # Turned away (2)
    ("turn sideways, normal voice", 2),
]


def generate_beep(freq=600, duration_ms=100, rate=16000, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(rate * duration_ms / 1000), endpoint=False)
    w = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(rate * 0.005)
    w[:fade] *= np.linspace(0, 1, fade)
    w[-fade:] *= np.linspace(1, 0, fade)
    return w.astype(np.float32)


def record_clip(duration: float) -> np.ndarray:
    """Record a clip and return int16 audio."""
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="int16")
    sd.wait()
    return audio.flatten()


def save_wav(path: Path, audio: np.ndarray):
    """Save int16 audio as WAV."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def play_beep():
    """Play a short beep to signal recording start."""
    beep = generate_beep()
    sd.play(beep, SAMPLE_RATE)
    sd.wait()


def main():
    parser = argparse.ArgumentParser(description="Record wake word samples")
    parser.add_argument("--speaker", required=True, help="Speaker name (e.g. miles, jennie)")
    parser.add_argument("--output-dir", default="/home/chaos/wake-training",
                        help="Output directory")
    parser.add_argument("--wake-word", default="hey_jarvis",
                        help="Wake word label")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing clips to allow resuming
    existing = list(out_dir.glob(f"{args.wake_word}_*.wav"))
    start_idx = len(existing)

    total = sum(count for _, count in PROMPTS)

    print(f"\n{'='*60}")
    print(f"  ChaosVector Wake — Recording Session")
    print(f"  Speaker: {args.speaker}")
    print(f"  Wake word: \"hey Jarvis\"")
    print(f"  Clips to record: {total - start_idx} (of {total})")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    print()
    print("  Instructions:")
    print("  - Wait for the beep, then say \"hey Jarvis\" clearly")
    print("  - Each recording is 2.5 seconds")
    print("  - Follow the distance/volume prompts")
    print("  - Press Ctrl+C to stop early (progress is saved)")
    print()

    if start_idx > 0:
        print(f"  Resuming from clip {start_idx + 1} ({start_idx} already recorded)")
        print()

    input("  Press Enter to begin...\n")

    clip_idx = 0
    recorded = start_idx

    for prompt_text, count in PROMPTS:
        for i in range(count):
            clip_idx += 1
            if clip_idx <= start_idx:
                continue  # skip already recorded

            print(f"\n  [{recorded + 1}/{total}] {prompt_text}")
            print(f"  Say \"hey Jarvis\" after the beep...")
            time.sleep(0.5)

            # Beep
            play_beep()
            time.sleep(0.1)

            # Record
            audio = record_clip(CLIP_DURATION)
            rms = np.sqrt(np.mean((audio.astype(np.float64) / 32768) ** 2))

            # Save
            filename = f"{args.wake_word}_{recorded:03d}.wav"
            save_wav(out_dir / filename, audio)
            recorded += 1

            print(f"  ✓ Saved {filename} (RMS={rms:.4f})")

            if rms < 0.001:
                print("  ⚠ Very quiet — make sure you said the wake word")

            time.sleep(PAUSE_BETWEEN)

    print(f"\n{'='*60}")
    print(f"  Done! {recorded} clips saved to {out_dir}")
    print(f"{'='*60}")
    print()
    print("  Next steps:")
    print("  - Record other family members with --speaker <name>")
    print("  - Train the wake word model")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Recording stopped. Progress saved — run again to resume.")
        sys.exit(0)
