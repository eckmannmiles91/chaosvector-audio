"""Context engine client — HTTP to Pi-Fi context engine for local answers.

Handles time, weather, calendar, presence without needing the LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    url: str = "http://10.1.1.176:8400"
    cache_path: str = "/tmp/pifi_context_cache.json"
    answer_timeout: float = 2.0
    cache_refresh_interval: float = 300.0  # 5 minutes


class ContextClient:
    """HTTP client for the Pi-Fi context engine."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()
        self._session: aiohttp.ClientSession | None = None
        self._available = False
        self._refresh_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(
                f"{self.config.url}/health",
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                self._available = resp.status == 200
                if self._available:
                    log.info("context engine connected: %s", self.config.url)
                    await self._refresh_cache()
                    self._refresh_task = asyncio.create_task(self._cache_loop())
                return self._available
        except Exception as e:
            log.warning("context engine unavailable: %s", e)
            self._available = False
            return False

    async def disconnect(self) -> None:
        self._available = False
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def is_available(self) -> bool:
        return self._available and self._session is not None

    async def get_answer(self, query_type: str, speaker: str | None = None) -> str | None:
        """Get a pre-computed answer from the context engine.

        query_type: "weather", "forecast", "calendar", "presence", "time",
                    "indoor_temp", "thermostat", "lights_status", etc.
        """
        if not self.is_available:
            return None
        params: dict = {"q": query_type}
        if speaker:
            params["speaker"] = speaker
        try:
            async with self._session.get(
                f"{self.config.url}/answer",
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.config.answer_timeout),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("answer")
        except Exception as e:
            log.warning("context engine query failed (%s): %s", query_type, e)
        return None

    async def get_relevant_context(self, query: str, speaker: str | None = None) -> dict | None:
        """Get context relevant to a query (for LLM prompt building)."""
        if not self.is_available:
            return None
        params: dict = {"q": query}
        if speaker:
            params["speaker"] = speaker
        try:
            async with self._session.get(
                f"{self.config.url}/context/relevant",
                params=params,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    async def _refresh_cache(self) -> None:
        """Fetch full context snapshot and write to disk cache."""
        try:
            async with self._session.get(
                f"{self.config.url}/context/slim",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    Path(self.config.cache_path).write_text(json.dumps(data))
                    log.debug("context cache refreshed")
        except Exception as e:
            log.debug("context cache refresh failed: %s", e)

    async def _cache_loop(self) -> None:
        """Background loop to refresh the context cache periodically."""
        while True:
            await asyncio.sleep(self.config.cache_refresh_interval)
            await self._refresh_cache()


def get_local_time() -> str:
    """Return current time formatted for TTS."""
    now = datetime.now()
    hour = now.hour % 12 or 12
    minute = now.minute
    ampm = "AM" if now.hour < 12 else "PM"
    if minute == 0:
        return f"It's {hour} {ampm}."
    elif minute < 10:
        return f"It's {hour} oh {minute} {ampm}."
    else:
        return f"It's {hour} {minute} {ampm}."
