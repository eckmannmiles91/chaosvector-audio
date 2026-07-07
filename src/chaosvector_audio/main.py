"""ChaosVector Audio — main entry point.

Loads config from YAML, resolves environment variables, starts the pipeline.

Usage:
    python -m chaosvector_audio --config /path/to/config.yaml
    python -m chaosvector_audio  # uses default config.yaml in working dir
"""

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
from pathlib import Path

import yaml

from chaosvector_audio.orchestrator import Orchestrator, PipelineConfig

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value):
    """Recursively resolve ${ENV_VAR} in strings."""
    if isinstance(value, str):
        def _replace(m):
            return os.environ.get(m.group(1), "")
        return _ENV_RE.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(path: str | Path) -> PipelineConfig:
    """Load PipelineConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_env(raw)

    audio = raw.get("audio", {})
    wake = raw.get("wake_word", {})
    vad = raw.get("vad", {})
    stt = raw.get("stt", {})
    fast = raw.get("fast_stt", {})
    tts = raw.get("tts", {})
    llm = raw.get("llm", {})
    ha = raw.get("home_assistant", {})
    ctx = raw.get("context_engine", {})
    spk = raw.get("speaker_verify", {})
    follow = raw.get("follow_up", {})
    vol = raw.get("volume_adapt", {})
    brief = raw.get("brief_mode", {})
    avr = raw.get("avr", {})
    wv = raw.get("wake_verifier", {})
    fb = raw.get("feedback", {})
    snd = raw.get("sounds", {})

    return PipelineConfig(
        mic_device=audio.get("mic_device"),
        sample_rate=audio.get("sample_rate", 16000),
        channels=audio.get("channels", 1),
        chunk_ms=audio.get("chunk_ms", 20),
        pre_roll_ms=audio.get("pre_roll_ms", 500),
        playback_device=audio.get("playback_device"),
        playback_rate=audio.get("playback_rate", 22050),

        wake_host=wake.get("host", "127.0.0.1"),
        wake_port=wake.get("port", 10400),
        wake_names=wake.get("names", ["hey_jarvis"]),
        wake_energy_threshold=wake.get("energy_threshold", 320),
        wake_gain=wake.get("gain", 4.0),

        vad_aggressiveness=vad.get("aggressiveness", 2),
        silence_frames=vad.get("silence_frames", 25),
        min_speech_frames=vad.get("min_speech_frames", 3),
        listen_timeout=vad.get("listen_timeout", 10.0),

        wake_beep=raw.get("wake_beep", False),
        chime_blanking_ms=raw.get("chime_blanking_ms", 500),
        echo_gate_ms=raw.get("echo_gate_ms", 300),
        backend_error_cooldown_s=raw.get("backend_error_cooldown_s", 90.0),

        stt_host=stt.get("host", "10.1.1.240"),
        stt_port=stt.get("port", 10301),
        stt_timeout=stt.get("timeout", 10.0),

        fast_stt_host=fast.get("host", "10.1.1.53"),
        fast_stt_port=fast.get("port", 10302),
        fast_stt_enabled=fast.get("enabled", True),

        tts_host=tts.get("host", "10.1.1.240"),
        tts_port=tts.get("port", 10210),
        tts_voice=tts.get("voice", "af_heart"),
        tts_timeout=tts.get("timeout", 10.0),

        ollama_url=llm.get("url", "http://10.1.1.240:8081"),
        ollama_model=llm.get("model", ""),
        ollama_system_prompt_file=llm.get("system_prompt_file", ""),
        ollama_timeout=llm.get("timeout", 15.0),
        ollama_max_tokens=llm.get("max_tokens", 120),

        ha_ws_url=ha.get("ws_url", "ws://10.1.1.53:8123/api/websocket"),
        ha_http_url=ha.get("http_url", "http://10.1.1.53:8123"),
        ha_token=ha.get("token", ""),
        ha_pipeline=ha.get("pipeline"),
        ha_intent_timeout=ha.get("intent_timeout", 10.0),

        context_url=ctx.get("url", "http://10.1.1.176:8400"),

        speaker_url=spk.get("url", "http://10.1.1.228:8500"),
        speaker_enabled=spk.get("enabled", True),

        follow_up_timeout=follow.get("timeout", 5.0),

        volume_adapt=vol.get("enabled", True),
        volume_adapt_min=vol.get("min", 0.25),
        volume_adapt_max=vol.get("max", 0.85),
        volume_adapt_rms_low=vol.get("rms_low", 500),
        volume_adapt_rms_high=vol.get("rms_high", 4000),

        brief_mode=brief.get("enabled", True),
        brief_min_frequency=brief.get("min_frequency", 3),
        brief_top_n=brief.get("top_n", 20),

        avr_enabled=avr.get("enabled", False),
        avr_device_name=avr.get("device_name", ""),
        avr_restore_delay=avr.get("restore_delay", 1.0),

        wake_verifier_path=wv.get("path", ""),
        wake_verifier_threshold=wv.get("threshold", 0.5),

        feedback_dir=fb.get("dir", "/var/lib/pi-fi/feedback"),
        sounds_dir=snd.get("dir", "/home/chaos/pi-fi-software/voice/sounds"),
        pifi_path=raw.get("pifi_path", "/home/chaos/pi-fi-software/voice"),
    )


async def run(config: PipelineConfig) -> None:
    orch = Orchestrator(config)
    await orch.start()

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())

    run_task = asyncio.create_task(orch.run())
    await stop.wait()

    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    await orch.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaosVector Audio — Voice Pipeline Daemon")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
