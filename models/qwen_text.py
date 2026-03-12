"""Qwen2.5-32B-Instruct text-only client via OpenAI-compatible API (vLLM)."""

import logging
import time

import openai

from config import QWEN_TEXT_ENDPOINT, QWEN_TEXT_MODEL
from models.base import LLMClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
TIMEOUT = 600  # 10 min — 32B text model on local hardware


class QwenTextClient(LLMClient):
    """Client for Qwen2.5-32B-Instruct served via vLLM."""

    def __init__(
        self,
        endpoint: str = QWEN_TEXT_ENDPOINT,
        model: str = QWEN_TEXT_MODEL,
    ):
        self.model = model
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key="EMPTY",
            timeout=TIMEOUT,
        )

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
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
