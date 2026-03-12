"""JSON I/O, response parsing, and logging setup."""

import json
import logging
import re
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def save_json(data: dict, path: str) -> None:
    """Save dict as formatted JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON to %s", path)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_response(response: str) -> dict:
    """
    Parse JSON from model response.

    Handles common issues:
      - Strip markdown code fences (```json ... ```)
      - Strip leading/trailing whitespace
      - Try json.loads first
      - If that fails, try to find JSON object/array in the string
      - If still fails, return {"_raw_response": response, "_parse_error": True}
    """
    text = response.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the string
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from model response (length=%d)", len(response))
    return {"_raw_response": response, "_parse_error": True}


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with timestamps."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy HTTP debug logs (they dump full base64 image payloads)
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
