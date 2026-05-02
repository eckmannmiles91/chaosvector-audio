"""Wyoming TCP wake word client for openWakeWord integration.

Adapts the Wyoming protocol (Detect/AudioStart/AudioChunk/Detection) into
a simple async interface that the pipeline can use.

The wake word detector runs as an async task, consuming audio chunks from
an asyncio.Queue and signalling wake events via a callback.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import array as _array
from dataclasses import dataclass

from wyoming.audio import AudioChunk as WyAudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.wake import Detect, Detection, NotDetected

log = logging.getLogger(__name__)


@dataclass
class WakeConfig:
    host: str = "127.0.0.1"
    port: int = 10400
    names: list[str] | None = None  # wake word names to detect (None = all)
    energy_threshold: float = 350.0  # reject wake if RMS below this
    gain: float = 1.0  # gain applied to audio before sending to detector
    reconnect_delay: float = 2.0


class WakeWordClient:
    """Async Wyoming TCP client for openWakeWord.

    Usage:
        wake = WakeWordClient(config)
        wake.on_wake = my_callback  # async def my_callback(name, rms): ...
        await wake.start(audio_queue)
        ...
        await wake.stop()
    """

    def __init__(self, config: WakeConfig | None = None) -> None:
        self.config = config or WakeConfig()
        self.on_wake: asyncio.Event | None = None  # set externally
        self._client: AsyncTcpClient | None = None
        self._connected = False
        self._task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue | None = None
        self._wake_event = asyncio.Event()
        self._wake_name: str = ""
        self._wake_rms: float = 0.0
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- lifecycle -----------------------------------------------------------

    async def start(self, audio_queue: asyncio.Queue) -> None:
        """Start the detection loop. Audio chunks (bytes, 16kHz S16LE mono)
        should be put into audio_queue by the capture manager."""
        self._audio_queue = audio_queue
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._disconnect()

    # -- public interface ----------------------------------------------------

    async def wait_for_wake(self) -> tuple[str, float]:
        """Block until wake word is detected. Returns (name, rms)."""
        self._wake_event.clear()
        await self._wake_event.wait()
        return self._wake_name, self._wake_rms

    def force_reconnect(self) -> None:
        """Force disconnect so next cycle gets a fresh connection.
        Call this after each interaction to prevent stale TCP state."""
        asyncio.create_task(self._disconnect())

    # -- internal ------------------------------------------------------------

    async def _connect(self) -> None:
        self._client = AsyncTcpClient(self.config.host, self.config.port)
        await self._client.connect()
        self._connected = True
        log.info("wake word connected to %s:%d", self.config.host, self.config.port)

    async def _disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            self._connected = False

    async def _run_loop(self) -> None:
        """Outer loop: connect, run detection, handle errors, reconnect."""
        while self._running:
            try:
                if not self._connected:
                    await self._connect()
                await self._detection_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("wake word error: %s", e)
                await self._disconnect()
                await asyncio.sleep(self.config.reconnect_delay)

    async def _detection_cycle(self) -> None:
        """Run one detection cycle: send Detect+AudioStart, stream audio,
        wait for Detection or NotDetected."""
        if self._client is None:
            return

        # Start detection
        names = self.config.names or []
        await self._client.write_event(Detect(names=names).event())
        await self._client.write_event(
            AudioStart(rate=16000, width=2, channels=1).event()
        )

        # Rolling RMS buffer (~500ms = 25 chunks at 20ms)
        rms_buffer: collections.deque[float] = collections.deque(maxlen=25)
        stop = asyncio.Event()
        detection_name: str | None = None

        async def _send():
            gain = self.config.gain
            sent = 0
            while not stop.is_set() and self._running:
                try:
                    chunk_bytes = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    log.debug("wake _send: queue timeout (no audio)")
                    continue
                if chunk_bytes is None:
                    stop.set()
                    break
                rms_buffer.append(_chunk_rms(chunk_bytes))
                if gain != 1.0:
                    chunk_bytes = _apply_gain(chunk_bytes, gain)
                await self._client.write_event(
                    WyAudioChunk(
                        rate=16000, width=2, channels=1, audio=chunk_bytes
                    ).event()
                )
                sent += 1
                if sent % 50 == 1:
                    log.debug("wake _send: %d chunks sent, last_rms=%.1f", sent, rms_buffer[-1])

        async def _read():
            nonlocal detection_name
            while not stop.is_set():
                event = await self._client.read_event()
                if event is None:
                    self._connected = False
                    stop.set()
                    return
                if Detection.is_type(event.type):
                    detection = Detection.from_event(event)
                    detection_name = detection.name
                    stop.set()
                    return
                if NotDetected.is_type(event.type):
                    stop.set()
                    return

        send_task = asyncio.create_task(_send())
        read_task = asyncio.create_task(_read())

        try:
            await asyncio.wait(
                {send_task, read_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in (send_task, read_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            # Send AudioStop
            if self._connected and self._client is not None:
                try:
                    await self._client.write_event(AudioStop().event())
                except Exception:
                    pass

        if detection_name:
            wake_rms = (sum(rms_buffer) / len(rms_buffer)) if rms_buffer else 0.0
            # Energy gate
            if wake_rms < self.config.energy_threshold:
                log.debug(
                    "wake rejected (rms=%.1f < threshold=%.1f)",
                    wake_rms, self.config.energy_threshold,
                )
                return  # loop will restart detection cycle
            log.info("wake word detected: %s (rms=%.1f)", detection_name, wake_rms)
            self._wake_name = detection_name
            self._wake_rms = wake_rms
            self._wake_event.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_rms(chunk: bytes) -> float:
    """Compute RMS energy of a 16-bit PCM audio chunk."""
    n = len(chunk) // 2
    if n == 0:
        return 0.0
    samples = _array.array("h")
    samples.frombytes(chunk)
    return math.sqrt(sum(s * s for s in samples) / n)


def _apply_gain(chunk: bytes, gain: float) -> bytes:
    """Apply gain with clipping protection."""
    samples = _array.array("h")
    samples.frombytes(chunk)
    for i in range(len(samples)):
        v = int(samples[i] * gain)
        samples[i] = max(-32768, min(32767, v))
    return samples.tobytes()
