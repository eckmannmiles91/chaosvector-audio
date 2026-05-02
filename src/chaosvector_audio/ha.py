"""Home Assistant client — WebSocket intent execution for device control.

Uses a fresh connection per intent request to avoid stale WebSocket issues.
This matches the philosophy of the STT/TTS clients (connect, do work, disconnect).
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
    """Home Assistant WebSocket client for intent execution.

    Uses a fresh WebSocket connection per intent to avoid stale connections.
    """

    def __init__(self, config: HAConfig) -> None:
        self.config = config
        self._available = False

    async def connect(self) -> bool:
        """Verify HA is reachable (quick health check via HTTP)."""
        if not self.config.token:
            log.warning("HA token not configured")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.http_url}/api/",
                    headers={"Authorization": f"Bearer {self.config.token}"},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    self._available = resp.status == 200
                    if self._available:
                        log.info("HA connected and authenticated")
                    else:
                        log.warning("HA health check returned %d", resp.status)
                    return self._available
        except Exception as e:
            log.warning("HA connect failed: %s", e)
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    async def run_intent(self, text: str, conversation_id: str | None = None) -> str | None:
        """Send text to HA Assist pipeline and return response speech.

        Opens a fresh WebSocket connection per request — no stale connections.
        Returns response text or None on failure.
        """
        if not self.config.token:
            return None

        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(
                self.config.ws_url,
                timeout=aiohttp.ClientTimeout(total=10.0),
            )
        except Exception as e:
            log.warning("HA WebSocket connect failed: %s", e)
            await session.close()
            return None

        try:
            # Auth handshake
            msg = await ws.receive_json(timeout=5.0)
            if msg.get("type") != "auth_required":
                log.warning("HA unexpected: %s", msg)
                return None

            await ws.send_json({
                "type": "auth",
                "access_token": self.config.token,
            })

            msg = await ws.receive_json(timeout=5.0)
            if msg.get("type") != "auth_ok":
                log.warning("HA auth failed: %s", msg)
                return None

            # Send intent
            msg_id = 1
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

            await ws.send_json(cmd)

            # Read events until intent-end or error
            async with asyncio.timeout(self.config.intent_timeout):
                async for ws_msg in ws:
                    if ws_msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(ws_msg.data)
                        if data.get("id") != msg_id:
                            continue

                        if data.get("type") == "result":
                            if not data.get("success"):
                                error = data.get("error", {})
                                log.warning("HA intent failed: %s", error.get("message", data))
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
                            error_data = event.get("data", {})
                            log.warning("HA error event: %s", error_data)
                            return None

                    elif ws_msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        log.warning("HA WebSocket closed unexpectedly")
                        return None

        except asyncio.TimeoutError:
            log.warning("HA intent timed out (%.1fs)", self.config.intent_timeout)
            return None
        except Exception as e:
            log.warning("HA intent error: %s", e)
            return None
        finally:
            await ws.close()
            await session.close()

        return None

    async def disconnect(self) -> None:
        """No-op — connections are per-request."""
        self._available = False
