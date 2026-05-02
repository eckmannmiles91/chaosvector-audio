"""Home Assistant client — WebSocket intent execution for device control.

Sends user commands ("turn off the lights") to HA's Assist pipeline via
WebSocket and returns the response text.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)


@dataclass
class HAConfig:
    ws_url: str = "ws://10.1.1.53:8123/api/websocket"
    http_url: str = "http://10.1.1.53:8123"
    token: str = ""
    pipeline: str | None = None  # Assist pipeline ID
    intent_timeout: float = 10.0


class HAClient:
    """Home Assistant WebSocket client for intent execution."""

    def __init__(self, config: HAConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._available = False
        self._msg_id = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to HA WebSocket and authenticate."""
        if not self.config.token:
            log.warning("HA token not configured")
            return False

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self.config.ws_url,
                timeout=aiohttp.ClientTimeout(total=10.0),
            )

            # Receive auth_required
            msg = await self._ws.receive_json(timeout=5.0)
            if msg.get("type") != "auth_required":
                log.warning("HA unexpected message: %s", msg)
                await self._close_ws()
                return False

            # Send auth
            await self._ws.send_json({
                "type": "auth",
                "access_token": self.config.token,
            })

            # Receive auth_ok
            msg = await self._ws.receive_json(timeout=5.0)
            if msg.get("type") != "auth_ok":
                log.warning("HA auth failed: %s", msg)
                await self._close_ws()
                return False

            self._available = True
            log.info("HA connected and authenticated")
            return True

        except Exception as e:
            log.warning("HA connect failed: %s", e)
            await self._close_ws()
            return False

    @property
    def is_available(self) -> bool:
        return self._available and self._ws is not None and not self._ws.closed

    async def run_intent(self, text: str, conversation_id: str | None = None) -> str | None:
        """Send text to HA Assist pipeline and return response speech.

        Returns response text or None on failure.
        """
        if not self.is_available:
            # Try reconnect
            if not await self.connect():
                return None

        async with self._lock:
            self._msg_id += 1
            msg_id = self._msg_id

            cmd: dict = {
                "id": msg_id,
                "type": "assist_pipeline/run",
                "start_stage": "intent",
                "end_stage": "intent",
                "input": {"text": text},
                "timeout": int(self.config.intent_timeout),
            }
            if conversation_id:
                cmd["conversation_id"] = conversation_id
            if self.config.pipeline:
                cmd["pipeline"] = self.config.pipeline

            try:
                await self._ws.send_json(cmd)

                # Read events until intent-end or error
                async with asyncio.timeout(self.config.intent_timeout):
                    async for ws_msg in self._ws:
                        if ws_msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(ws_msg.data)
                            if data.get("id") != msg_id:
                                continue

                            if data.get("type") == "result":
                                if not data.get("success"):
                                    log.warning("HA intent failed: %s", data)
                                    return None
                                continue

                            event = data.get("event", {})
                            event_type = event.get("type")

                            if event_type == "intent-end":
                                intent_output = event.get("data", {}).get("intent_output", {})
                                response = intent_output.get("response", {})
                                speech = response.get("speech", {}).get("plain", {})
                                response_text = speech.get("speech", "")
                                log.info("HA response: \"%s\"", response_text[:80])
                                return response_text or "Done."

                            if event_type == "error":
                                log.warning("HA error: %s", event.get("data"))
                                return None

                        elif ws_msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            self._available = False
                            return None

            except asyncio.TimeoutError:
                log.warning("HA intent timed out")
                return None
            except Exception as e:
                log.warning("HA intent error: %s", e)
                self._available = False
                return None

        return None

    async def disconnect(self) -> None:
        self._available = False
        await self._close_ws()
        if self._session:
            await self._session.close()
            self._session = None

    async def _close_ws(self) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None
