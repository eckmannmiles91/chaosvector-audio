# chaosvector-audio

## What this is

Production voice pipeline daemon for the Pi-Fi smart speaker. Replaces satellite.py (2300 lines) with a modular, single-process architecture. In production since May 2026.

## Deployment

- **Pi 5** (10.1.1.235) — `/home/chaos/chaosvector-audio/`
- **Config:** `config.yaml` (YAML with `${ENV_VAR}` resolution)
- **Entry point:** `python -m chaosvector_audio --config config.yaml`
- **Systemd:** `chaosvector-audio.service` (user), `wyoming-openwakeword.service` (user), `chaosvector-moonshine.service` (user, shadow STT :10303)
- **HTTP API:** port 8300 (`/speak`, `/health`, `/status`)
- **Logs:** `/home/chaos/logs/chaosvector-audio.log` (daily rotation, 7 days)
- **Env:** Uses pi-fi-software venv at `/home/chaos/pi-fi-software/voice/.venv/`
- **Env vars:** `EnvironmentFile=/home/chaos/pi-fi-software/.env` (HA_TOKEN, etc.)

## Architecture

```
Capture (sounddevice) → VAD speech gate → openWakeWord (Wyoming TCP :10400)
  → Wake sound (15% volume!) → Listen (streaming STT + VAD end-of-speech)
  → Parallel STT: Speech-to-Phrase (:10302) + ChaosVector STT (:10301)
    → S2P early-cancel: if S2P matches, cancel Whisper to free compute
  → Intent classifier → Route:
      timer/reminder/alarm → local managers (set, cancel, check, snooze)
      simple_local → context engine / local time / humidity
      device cmd → rewrite dim→brightness, pronoun resolution → HA WebSocket
      general → context-enriched LLM streaming (E4B on OpenVINO iGPU :8081)
  → TTS: disk cache → memory LRU (300) → Kokoro remote (:10210) → Piper fallback
  → Playback (sounddevice, non-blocking, 15% wake beep volume)
  → Adaptive follow-up mode (3-8s based on response length)
```

## Key modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, YAML config loader |
| `orchestrator.py` | State machine, intent routing, all features |
| `capture.py` | Thread-safe audio capture (stdlib queue bridge) |
| `playback.py` | Priority-queued non-blocking playback with barge-in |
| `wake.py` | Wyoming TCP wake word client |
| `stt.py` | ChaosVector STT (Wyoming TCP, per-request) |
| `stt_streaming.py` | Streaming STT (audio sent in real-time during listen) |
| `stt_fast.py` | Speech-to-Phrase fast path with early-cancel |
| `tts.py` | TTS waterfall: remote Kokoro → local Piper |
| `tts_cache.py` | LRU cache (300 entries) + disk persistence + 2h time prewarm |
| `llm.py` | OpenAI/Ollama dual-API streaming, context-stripped history |
| `context.py` | Context engine HTTP client (60s refresh, disk cache fallback) |
| `ha.py` | HA WebSocket (fresh connection per intent) |
| `http_api.py` | HTTP API: /speak for proactive announcements, /health, /status |
| `stt_filters.py` | Name corrections + hallucination filter |
| `feedback.py` | JSONL interaction logging |
| `speaker.py` | Speaker verification (Resemblyzer HTTP) |
| `sounds.py` | Thinking indicator + sound loading |
| `health.py` | HA sensor (sensor.chaosvector_audio) |

## Features (July 2026)

- **Timer/reminder/alarm** — "set a 5 minute timer", "remind me to check the laundry in 30 minutes", "set an alarm for 7 AM"
- **Pronoun resolution** — "turn on the office light" then "make it dimmer" resolves "it" to office light
- **Adaptive follow-up** — 3s after confirmations, 5s default, 8s after long answers
- **TTS disk cache** — 300-entry LRU persists across restarts, 2h rolling time phrase coverage
- **S2P early-cancel** — cancels Whisper when Speech-to-Phrase matches fast
- **HTTP /speak** — HA automations push announcements: `POST :8300/speak {"message": "..."}`
- **Geocoding cache** — 10min TTL for person location lookups
- **Weather cleanup** — "partlycloudy" → "partly cloudy"
- **Context enrichment** — last device, time of day, recent interactions, speaker ID in LLM prompt
- **Streaming follow-up STT** — audio streamed in real-time during follow-up listening

## Critical lessons (DO NOT repeat)

1. **Wake beep at 15% volume** — full volume garbles STT first word
2. **Pre-roll = room noise before wake, NOT the wake word itself**
3. **Synthetic wake training ≠ real-world TV rejection** — always soak test
4. **VAD speech gate essential** — filters sniffles/clicks before wake detector
5. **Fresh TCP per request** — eliminates stale connection bugs
6. **HA doesn't understand "dim"** — rewrite to "set brightness"
7. **OpenVINO >> SYCL on Panther Lake** — 17 tok/s vs 10, no flash-attn
8. **Moonshine tiny drops short names** — "Eli" → "the", keep Whisper as primary
9. **Qwen 3 4B garbage via llama-server** — use Ollama if retrying
10. **TTS cache put() args must match TTSResult** — audio_bytes doesn't exist

## Config reference (config.yaml)

Key values that affect behavior:
- `wake_word.energy_threshold: 320` — reject quiet noise
- `wake_word.gain: 4.0` — boost audio to wake detector
- `chime_blanking_ms: 500` — skip audio after wake beep
- `echo_gate_ms: 300` — suppress wake after playback
- `vad.silence_frames: 25` — 500ms silence = end of speech
- `fast_stt.enabled: true` — parallel Speech-to-Phrase
- `volume_adapt.enabled: true` — scale TTS to match speech RMS
- `llm.url: http://10.1.1.240:8081` — E4B on OpenVINO iGPU (microchaos3)

## SSH access

- Pi 5: `ssh chaos@10.1.1.235`
- microchaos3 (LLM/STT/TTS): `ssh root@10.1.1.240`
- microchaos2 (old LLM, disabled): `ssh root@10.1.1.228`
- HA: `http://10.1.1.53:8123`
