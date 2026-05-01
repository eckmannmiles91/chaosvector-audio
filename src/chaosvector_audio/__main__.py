"""Entry point for chaosvector-audio daemon."""

import argparse
import asyncio
import logging
import signal
import sys

from chaosvector_audio import __version__
from chaosvector_audio.pipeline import AudioPipeline

log = logging.getLogger("chaosvector_audio")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chaosvector-audio",
        description="Unified audio pipeline daemon for Pi-Fi speaker",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file (optional)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="ALSA/PipeWire capture device name or index",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Capture sample rate in Hz (default: 16000)",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Capture channel count (default: 1)",
    )
    p.add_argument(
        "--pre-roll-ms",
        type=int,
        default=500,
        help="Pre-roll ring buffer length in ms (default: 500)",
    )
    p.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="WebRTC VAD aggressiveness 0-3 (default: 2)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    pipeline = AudioPipeline(
        device=args.device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        pre_roll_ms=args.pre_roll_ms,
        vad_aggressiveness=args.vad_aggressiveness,
    )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows: fall back to signal.signal
            signal.signal(sig, lambda *_: stop.set())

    log.info("starting chaosvector-audio %s", __version__)
    await pipeline.start()

    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        await pipeline.stop()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
