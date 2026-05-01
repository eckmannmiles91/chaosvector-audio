# chaosvector-audio

## What this is

Unified audio pipeline daemon for the Pi-Fi speaker. Single Python process that owns the entire audio path: capture, playback, echo cancellation, and voice routing. Replaces the Wyoming TCP protocol stack and PipeWire filter-chain glue.

## Deployment target

- Raspberry Pi 5 (4 GB+)
- USB mic array (ReSpeaker or similar), or XMOS XVF3800 dev kit for hardware AEC
- PipeWire for audio backend (via sounddevice/portaudio)
- Python 3.11+

## Architecture

The pipeline is a state machine: IDLE -> LISTENING -> PROCESSING -> RESPONDING -> IDLE.

Key modules:
- `capture.py` — async audio capture with ring buffer pre-roll
- `playback.py` — priority-queued playback with barge-in support
- `aec.py` — echo cancellation (hardware passthrough or software via WebRTC AEC3)
- `vad.py` — voice activity detection with end-of-speech frame counting
- `pipeline.py` — orchestrates everything, owns the state machine

Integration callbacks are registered by the host application:
- `register_wake_detector(fn)` — synchronous, called per chunk
- `register_stt(fn)` — async, receives list of AudioChunks
- `register_intent(fn)` — async, receives transcript text
- `register_tts(fn)` — async, returns int16 numpy audio

## Design decisions

- No TCP sockets between components. Everything is in-process function calls.
- Pre-roll buffer lives in CaptureManager, drained on wake word detection.
- AEC reference signal comes via callback from PlaybackManager, not a loopback device.
- Playback uses a priority queue (wake beep > TTS > music) with barge-in cancellation.
- VAD uses WebRTC VAD with fallback to energy-based detection if webrtcvad not installed.

## Build and test

```bash
pip install -e ".[aec,dev]"
pytest
ruff check src/
```

## Known TODOs

- Software AEC is currently a simple echo gate (mute during tail). Real WebRTC AEC3 integration needs webrtc-audio-processing Python bindings.
- No tests yet — this is a scaffold.
- Config file loading (YAML/JSON) not implemented.
- Metrics/health endpoint for monitoring not implemented.
