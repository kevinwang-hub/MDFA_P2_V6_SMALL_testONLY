"""Qwen3-VL-4B Vision-Language client via OpenAI-compatible API (Ollama)."""

import logging
import time

import openai

from config import QWEN_VL_ENDPOINT, QWEN_VL_MODEL
from models.base import VLMClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
TIMEOUT = 300  # 5 min — 4B vision model on local hardware


class QwenVLClient(VLMClient):
    """Client for Qwen3-VL-4B served via Ollama."""

    def __init__(
        self,
        endpoint: str = QWEN_VL_ENDPOINT,
        model: str = QWEN_VL_MODEL,
    ):
        self.model = model
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key="EMPTY",
            timeout=TIMEOUT,
        )

    def generate(
        self,
        image: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    {"type": "text", "text": user},
                ],
            },
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
                msg = response.choices[0].message
                content = msg.content or ""
                # Qwen3 models use thinking/reasoning mode by default.
                # If content is empty, try the reasoning field.
                if not content.strip():
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
                    if reasoning.strip():
                        logger.info("Using reasoning field as content (thinking model)")
                        content = reasoning
                return content
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                last_error = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Qwen VL attempt %d/%d failed: %s — retrying in %ds",
                    attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Qwen VL failed after {MAX_RETRIES} retries: {last_error}"
        ) from last_error
