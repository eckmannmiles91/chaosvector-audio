# chaosvector-audio

## What this is

Production voice pipeline daemon for the Pi-Fi smart speaker. Replaces satellite.py (2300 lines) with a modular, single-process architecture. In production since May 2026.

## Deployment

- **Pi 5** (10.1.1.235) — `/home/chaos/chaosvector-audio/`
- **Config:** `config.yaml` (YAML with `${ENV_VAR}` resolution)
- **Entry point:** `python -m chaosvector_audio --config config.yaml`
- **Systemd:** `chaosvector-audio.service` (user), `wyoming-openwakeword.service` (user)
- **Logs:** `/home/chaos/logs/chaosvector-audio.log` (daily rotation, 7 days)
- **Env:** Uses pi-fi-software venv at `/home/chaos/pi-fi-software/voice/.venv/`
- **Env vars:** `EnvironmentFile=/home/chaos/pi-fi-software/.env` (HA_TOKEN, etc.)

## Architecture

```
Capture (sounddevice) → VAD speech gate → openWakeWord (Wyoming TCP :10400)
  → Wake sound (15% volume!) → Listen (VAD end-of-speech)
  → Parallel STT: Speech-to-Phrase (:10302) + ChaosVector STT (:10301)
  → Intent classifier → Route:
      simple_local → context engine / local time / humidity
      device cmd → rewrite dim→brightness → HA WebSocket (fresh per request)
      general → context-enriched LLM streaming (E4B on iGPU :8080)
  → TTS: cache check → Kokoro remote (:10210) → Piper local fallback
  → Playback (sounddevice, 15% wake beep volume)
  → Follow-up mode (5s listen without re-wake)
```

## Key modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, YAML config loader |
| `orchestrator.py` | State machine, intent routing, all features |
| `capture.py` | Thread-safe audio capture (stdlib queue bridge) |
| `playback.py` | Priority-queued playback with barge-in |
| `wake.py` | Wyoming TCP wake word client |
| `stt.py` | ChaosVector STT (Wyoming TCP, per-request) |
| `stt_fast.py` | Speech-to-Phrase fast path (Wyoming TCP) |
| `tts.py` | TTS waterfall: remote Kokoro → local Piper |
| `tts_cache.py` | LRU cache (200 entries) + pre-warm on startup |
| `llm.py` | OpenAI/Ollama dual-API streaming client |
| `context.py` | Context engine HTTP client (60s refresh) |
| `ha.py` | HA WebSocket (fresh connection per intent) |
| `stt_filters.py` | Name corrections + hallucination filter |
| `feedback.py` | JSONL interaction logging |
| `speaker.py` | Speaker verification (Resemblyzer HTTP) |
| `sounds.py` | Thinking indicator + sound loading |
| `health.py` | HA sensor (sensor.chaosvector_audio) |
| `wake_livekit.py` | LiveKit WakeWord Wyoming wrapper (disabled) |
| `wake_verify.py` | Speaker-specific wake verifier (disabled) |

## Critical lessons (DO NOT repeat)

1. **Wake beep at 15% volume** — full volume garbles STT first word
2. **Pre-roll = room noise before wake, NOT the wake word itself**
3. **Synthetic wake training ≠ real-world TV rejection** — always soak test
4. **VAD speech gate essential** — filters sniffles/clicks before wake detector
5. **Fresh TCP per request** — eliminates stale connection bugs
6. **HA doesn't understand "dim"** — rewrite to "set brightness"
7. **iGPU speed is bandwidth-limited** — smaller model ≠ faster

## Config reference (config.yaml)

Key values that affect behavior:
- `wake_word.energy_threshold: 320` — reject quiet noise
- `wake_word.gain: 4.0` — boost audio to wake detector
- `chime_blanking_ms: 500` — skip audio after wake beep
- `echo_gate_ms: 300` — suppress wake after playback
- `vad.silence_frames: 25` — 500ms silence = end of speech
- `fast_stt.enabled: true` — parallel Speech-to-Phrase
- `volume_adapt.enabled: true` — scale TTS to match speech RMS

## Rollback to satellite.py

```bash
systemctl --user stop chaosvector-audio
systemctl --user enable --now pifi-voice.service
```

## SSH access

- Pi 5: `ssh chaos@10.1.1.235`
- microchaos2 (LLM): `ssh root@10.1.1.228`
- microchaos3 (STT/TTS): `ssh root@10.1.1.240`
- HA: `http://10.1.1.53:8123`
