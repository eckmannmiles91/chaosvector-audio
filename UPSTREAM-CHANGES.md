# Changes from Upstream

ChaosVector Audio replaces the Pi-Fi speaker's current architecture of PipeWire filter-chains + Wyoming TCP protocol + multiple independent services with a single unified daemon.

## Why we built this
- Wake word hangs after interactions due to stale Wyoming TCP connections
- pw-play subprocess race conditions cause silent TTS playback
- PipeWire filter-chain state drifts after days of uptime (faint feedback, AEC issues)
- Five separate processes (wake word, voice satellite, STT client, TTS client, AEC monitor) all need to coordinate over TCP — any connection drop breaks the chain

## What we changed

### In-Process Architecture
- **Current:** Wake word → TCP → Satellite → TCP → STT server → TCP → TTS server → subprocess (pw-play)
- **Ours:** Single process owns the entire pipeline. Wake detection, STT, intent classification, TTS, and playback are direct function calls. Zero TCP serialization overhead, zero connection management bugs.

### State Machine with Forced Recovery
- **Current:** State transitions depend on Wyoming event streams that can silently hang
- **Ours:** Built-in timeouts on every state transition. Forced disconnect/reconnect on IDLE entry (the exact fix that solved the wake word hang bug). State machine cannot get stuck.

### Direct Audio I/O
- **Current:** pw-play subprocess for playback (race condition: exits before audio finishes)
- **Ours:** Direct PipeWire/ALSA API for both capture and playback. No subprocess spawning, no stdin pipe races. Playback completion is deterministic.

### Priority Playback Queue
- **Current:** TTS and wake beep compete for the same playback path
- **Ours:** Priority queue: wake beep > notification > TTS > music. Barge-in (wake word during playback) is a first-class feature, not a race between cancellation and audio.

### Integrated AEC
- **Current:** Separate AEC monitor process checks PipeWire node health every 5 minutes
- **Ours:** Echo cancellation reference signal captured directly from the playback path. Hardware AEC passthrough for XVF3800. Echo gate suppresses STT during/after playback. All in one process.

### Pre-Roll Buffer
- **Current:** External ring buffer in satellite code, injected into STT stream
- **Ours:** Ring buffer owned by the capture manager. Always captures last N ms. Automatically prepended to STT audio on wake detection. No coordination needed.

## Bugs This Architecture Eliminates
1. Wake word stale connection hang (no TCP connections to go stale)
2. pw-play race condition / silent playback (no subprocess)
3. PipeWire filter-chain drift (direct audio API)
4. AEC reference loop misconfiguration (integrated reference capture)
5. Follow-up mode breaking wake word reconnect (single state machine)
6. STT provider missing errors (no HA pipeline fallback needed)
