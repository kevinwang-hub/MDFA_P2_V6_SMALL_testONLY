"""Qwen3.5 9B text-only client via Ollama native /api/chat.

Uses requests.post directly instead of the OpenAI-compatible endpoint
because the OpenAI client silently drops Ollama-specific params like
num_ctx and think, causing truncation and runaway thinking."""

import json
import logging
import time

import requests

from config import QWEN_TEXT_MODEL
from models.base import LLMClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
TIMEOUT = 1800  # 30 min — 9B model on M1, large aggregation prompts
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
NUM_CTX = 16384  # enough for ~12K-token prompts + 4K output


class QwenTextClient(LLMClient):
    """Client for Qwen3.5 9B served via Ollama (native API)."""

    def __init__(
        self,
        model: str = QWEN_TEXT_MODEL,
        endpoint: str | None = None,  # kept for compatibility, ignored
    ):
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_ctx": NUM_CTX,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "think": False,
            "stream": False,
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    OLLAMA_CHAT_URL,
                    json=payload,
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                logger.info(
                    "Qwen Text response: %d chars, eval %.1fs",
                    len(content),
                    data.get("eval_duration", 0) / 1e9,
                )
                return content
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Qwen Text attempt %d/%d failed: %s — retrying in %ds",
                    attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Qwen Text failed after {MAX_RETRIES} retries: {last_error}"
        ) from last_error
