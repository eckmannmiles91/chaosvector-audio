# chaosvector-audio

Unified audio pipeline daemon for the Pi-Fi speaker. Replaces the fragile PipeWire filter-chain + Wyoming TCP protocol stitching with a single daemon that owns audio capture, playback, AEC, and routing.

## Why

- **No Wyoming TCP overhead** for co-located services
- **No wake word reconnection bugs** (everything in-process)
- **No pw-play subprocess race conditions** (direct PipeWire API via sounddevice)
- **Single process** = single point of monitoring
- **Pre-roll buffer managed internally** (no separate capture process)

## Architecture

```
Mic Array ──> CaptureManager ──> AEC ──> VAD ──> Pipeline State Machine
                                  ^                      │
                                  │              ┌───────┴────────┐
                                  │              │                │
                              EchoCanceller   Wake Det.      STT/Intent
                                  ^              │                │
                                  │              v                v
                              PlaybackManager <── TTS <── Response
```

State machine: `IDLE` -> `LISTENING` -> `PROCESSING` -> `RESPONDING` -> `IDLE`

## Install

```bash
pip install -e ".[aec,dev]"
```

## Run

```bash
chaosvector-audio --device default --sample-rate 16000 --pre-roll-ms 500
```

## Target

Raspberry Pi 5 with USB mic array (or XMOS XVF3800 for hardware AEC).
