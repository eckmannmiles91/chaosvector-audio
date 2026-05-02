"""LLM client — OpenAI-compatible streaming to llama-server (Gemma 4).

Simple, self-contained. No dependency on pi-fi OllamaClient.
Streams sentences for progressive TTS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    url: str = "http://10.1.1.228:8080"
    timeout: float = 15.0
    max_tokens: int = 120
    temperature: float = 0.3
    system_prompt: str = ""


# Abbreviations that should NOT trigger a sentence split
_ABBREVS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|approx)\.\s*$",
    re.IGNORECASE,
)


class LLMClient:
    """OpenAI-compatible streaming client for llama-server."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._session: aiohttp.ClientSession | None = None
        self._available = False
        self._history: deque[dict] = deque(maxlen=8)  # 4 turns

    async def connect(self) -> bool:
        """Check if llama-server is reachable."""
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(
                f"{self.config.url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                self._available = resp.status == 200
                if self._available:
                    log.info("LLM connected: %s", self.config.url)
                return self._available
        except Exception as e:
            log.warning("LLM health check failed: %s", e)
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available and self._session is not None

    async def disconnect(self) -> None:
        self._available = False
        self._history.clear()
        if self._session:
            await self._session.close()
            self._session = None

    def clear_history(self) -> None:
        self._history.clear()

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream sentences from the LLM as they're generated.

        Yields complete sentences for progressive TTS synthesis.
        """
        if not self.is_available:
            # Try reconnect
            if not await self.connect():
                return

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self._history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }

        full_response: list[str] = []
        token_buffer = ""
        first_yielded = False

        try:
            async with self._session.post(
                f"{self.config.url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                if resp.status != 200:
                    log.warning("LLM HTTP %d", resp.status)
                    return

                while True:
                    raw_line = await resp.content.readline()
                    if not raw_line:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    # SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith(b":"):
                        continue
                    if line.startswith(b"data: "):
                        line = line[6:]
                    elif line.startswith(b"data:"):
                        line = line[5:]
                    else:
                        continue
                    if line == b"[DONE]":
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract token
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if choices[0].get("finish_reason") is not None:
                        break

                    if token:
                        token_buffer += token
                        full_response.append(token)

                        # First chunk: break on clause for faster first audio
                        if not first_yielded:
                            pos = _find_clause_break(token_buffer)
                        else:
                            pos = _find_sentence_break(token_buffer)

                        if pos is not None:
                            sentence = token_buffer[:pos].strip()
                            token_buffer = token_buffer[pos:]
                            if sentence:
                                first_yielded = True
                                yield sentence

        except asyncio.TimeoutError:
            log.warning("LLM stream timed out")
        except (aiohttp.ClientError, OSError) as e:
            log.warning("LLM stream error: %s", e)

        # Yield remaining buffer
        remaining = token_buffer.strip()
        if remaining:
            yield remaining

        # Update history
        complete = "".join(full_response).strip()
        if complete:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": complete})


def _find_sentence_break(text: str) -> int | None:
    """Find index after the first sentence boundary."""
    for i, ch in enumerate(text):
        if ch in ".!?" and i + 1 < len(text) and text[i + 1] in " \n":
            if ch == "." and _ABBREVS.search(text[: i + 1]):
                continue
            return i + 2
        if ch == "\n" and i + 1 < len(text) and text[i + 1] == "\n":
            return i + 2
    return None


def _find_clause_break(text: str) -> int | None:
    """Find clause boundary for faster first audio chunk."""
    min_words = 10
    min_words_sentence = 4
    for i, ch in enumerate(text):
        if ch in ".!?" and i + 1 < len(text) and text[i + 1] in " \n":
            if ch == "." and _ABBREVS.search(text[: i + 1]):
                continue
            if len(text[:i].split()) >= min_words_sentence:
                return i + 2
        if ch in ",;:" and i + 1 < len(text) and text[i + 1] == " ":
            if len(text[:i].split()) >= min_words:
                return i + 2
        if ch == "\n" and i + 1 < len(text) and text[i + 1] == "\n":
            if len(text[:i].split()) >= min_words_sentence:
                return i + 2
    return None
