"""Health reporting — publishes component status to Home Assistant.

Uses HA REST API to update a sensor entity with pipeline health status.
Creates/updates sensor.chaosvector_audio_health.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp

log = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    state: str = "online"  # online, degraded, offline
    wake_word: bool = False
    stt: bool = False
    tts: bool = False
    llm: bool = False
    context_engine: bool = False
    ha: bool = False
    speaker_verify: bool = False
    uptime_s: float = 0.0
    interactions: int = 0
    last_interaction: str = ""


class HealthReporter:
    """Reports pipeline health to HA as a sensor entity via REST API."""

    def __init__(self, ha_url: str, ha_token: str, interval: float = 60.0) -> None:
        self._ha_url = ha_url.rstrip("/")
        self._ha_token = ha_token
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._status = HealthStatus()
        self._start_time = time.monotonic()

    async def start(self, get_status_fn) -> None:
        """Start periodic health reporting."""
        self._get_status = get_status_fn
        self._task = asyncio.create_task(self._report_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Report offline on shutdown
        await self._post_state("offline", {})

    async def _report_loop(self) -> None:
        """Periodically report health to HA."""
        while True:
            try:
                status = self._get_status()
                status.uptime_s = time.monotonic() - self._start_time

                # Determine overall state
                critical = [status.wake_word, status.stt, status.tts]
                if all(critical):
                    state = "online"
                elif any(critical):
                    state = "degraded"
                else:
                    state = "offline"

                attrs = {
                    "wake_word": status.wake_word,
                    "stt": status.stt,
                    "tts": status.tts,
                    "llm": status.llm,
                    "context_engine": status.context_engine,
                    "ha": status.ha,
                    "speaker_verify": status.speaker_verify,
                    "uptime_hours": round(status.uptime_s / 3600, 1),
                    "interactions": status.interactions,
                    "last_interaction": status.last_interaction,
                    "friendly_name": "ChaosVector Audio",
                    "icon": "mdi:microphone-message",
                }

                await self._post_state(state, attrs)

            except Exception as e:
                log.debug("health report failed: %s", e)

            await asyncio.sleep(self._interval)

    async def _post_state(self, state: str, attributes: dict) -> None:
        """Update sensor entity in HA via REST API."""
        if not self._ha_token:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._ha_url}/api/states/sensor.chaosvector_audio",
                    headers={
                        "Authorization": f"Bearer {self._ha_token}",
                        "Content-Type": "application/json",
                    },
                    json={"state": state, "attributes": attributes},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status in (200, 201):
                        log.debug("health reported: %s", state)
                    else:
                        log.debug("health report HTTP %d", resp.status)
        except Exception as e:
            log.debug("health post failed: %s", e)
