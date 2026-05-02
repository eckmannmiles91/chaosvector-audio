"""Phase 4 Proof: full voice assistant — wake → STT → intent → Ollama → TTS → playback.

Run on Pi 5:
    /home/chaos/pi-fi-software/voice/.venv/bin/python tests/phase4_proof.py

Say "hey Jarvis", ask a question, get a real LLM response spoken back.
"""

import asyncio
import logging
import signal
import sys

sys.path.insert(0, "/home/chaos/chaosvector-audio/src")

from chaosvector_audio.orchestrator import Orchestrator, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def main() -> None:
    print("=== ChaosVector Audio — Phase 4: Full Voice Assistant ===")
    print("Say 'hey Jarvis' and ask anything. Ctrl+C to exit.\n")

    config = PipelineConfig(
        # Audio
        mic_device=None,  # PipeWire default
        sample_rate=16000,
        channels=1,
        chunk_ms=20,
        pre_roll_ms=500,

        # Wake word
        wake_host="127.0.0.1",
        wake_port=10400,
        wake_names=["hey_jarvis"],
        wake_energy_threshold=200.0,

        # VAD
        vad_aggressiveness=2,
        silence_frames=20,

        # STT (ChaosVector STT on microchaos3)
        stt_host="10.1.1.240",
        stt_port=10301,

        # TTS (ChaosVector TTS on microchaos3)
        tts_host="10.1.1.240",
        tts_port=10210,
        tts_voice="af_heart",

        # LLM (Gemma 4 on microchaos2 via llama-server)
        ollama_url="http://10.1.1.228:8080",
        ollama_model="gemma4-e4b",
        ollama_api_format="openai",
        ollama_system_prompt_file="/home/chaos/pi-fi-software/voice/system_prompt_phase1b.txt",
        ollama_timeout=15.0,
        ollama_max_tokens=120,

        # Pi-Fi path for intent classifier import
        pifi_path="/home/chaos/pi-fi-software/voice",
    )

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

    print("\n=== Phase 4 complete ===")


if __name__ == "__main__":
    asyncio.run(main())
