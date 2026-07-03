"""HTTP API — exposes /speak endpoint for HA automations and external callers.

Enables proactive announcements: door open, timer done, weather alerts, etc.
HA automations can call this to push speech through the speaker.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

from aiohttp import web

log = logging.getLogger(__name__)


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8300


class HTTPApi:
    """Lightweight HTTP API for the voice pipeline."""

    def __init__(self, config: APIConfig | None = None) -> None:
        self.config = config or APIConfig()
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._speak_fn: Callable[[str], Awaitable[None]] | None = None
        self._status_fn: Callable[[], dict] | None = None

        self._app.router.add_post("/speak", self._handle_speak)
        self._app.router.add_get("/speak", self._handle_speak_get)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/status", self._handle_status)

    async def start(
        self,
        speak_fn: Callable[[str], Awaitable[None]],
        status_fn: Callable[[], dict] | None = None,
    ) -> None:
        self._speak_fn = speak_fn
        self._status_fn = status_fn
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()
        log.info("HTTP API listening on %s:%d", self.config.host, self.config.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            log.info("HTTP API stopped")

    async def _handle_speak(self, request: web.Request) -> web.Response:
        """POST /speak — synthesize and play a message.

        Body: {"message": "Front door opened"} or {"message": "...", "priority": "high"}
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        message = data.get("message", "").strip()
        if not message:
            return web.json_response({"error": "message required"}, status=400)

        if self._speak_fn is None:
            return web.json_response({"error": "speak not available"}, status=503)

        log.info("HTTP /speak: %s", message[:80])
        try:
            await self._speak_fn(message)
            return web.json_response({"status": "ok", "message": message[:80]})
        except Exception as e:
            log.error("HTTP /speak failed: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_speak_get(self, request: web.Request) -> web.Response:
        """GET /speak?message=... — convenience for simple curl/webhook use."""
        message = request.query.get("message", "").strip()
        if not message:
            return web.json_response({"error": "message query param required"}, status=400)

        if self._speak_fn is None:
            return web.json_response({"error": "speak not available"}, status=503)

        log.info("HTTP /speak (GET): %s", message[:80])
        try:
            await self._speak_fn(message)
            return web.json_response({"status": "ok", "message": message[:80]})
        except Exception as e:
            log.error("HTTP /speak failed: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — simple health check."""
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /status — pipeline status details."""
        if self._status_fn:
            return web.json_response(self._status_fn())
        return web.json_response({"status": "ok"})
