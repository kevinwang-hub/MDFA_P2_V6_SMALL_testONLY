"""Fuzzy-match image filename to figure caption in the paper."""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def match_caption(
    image_filename: str, caption_map: dict[str, str]
) -> dict[str, str | None]:
    """
    Given an image filename, find its figure identifier and caption.

    caption_map may contain both:
      - Normalized figure IDs ("Figure 1") -> caption text  (from chunker)
      - Hash-based filenames ("abc123...jpg") -> caption text  (from loader image_map)

    Returns:
        {
            "figure_id": str or None,
            "caption": str or None,
            "panel": str or None,
        }
    """
    # 1. Direct filename lookup (handles hash-based filenames from image_map)
    if image_filename in caption_map:
        caption = caption_map[image_filename]
        # Try to extract a figure ID from the caption text itself
        fig_id = None
        if caption:
            fid_match = FIGURE_ID_PATTERN.search(caption)
            if fid_match:
                fig_id = _normalize_figure_id(fid_match.group(1))
        return {"figure_id": fig_id, "caption": caption or None, "panel": None}

    stem = re.sub(r"\.\w+$", "", image_filename)  # strip extension

    # Extract number and optional panel letter
    num_match = re.search(r"(\d+)", stem)
    panel_match = re.search(r"\d+([a-z])", stem, re.IGNORECASE)

    if not num_match:
        logger.debug("No number found in filename: %s", image_filename)
        return {"figure_id": None, "caption": None, "panel": None}

    number = num_match.group(1)
    panel = panel_match.group(1).lower() if panel_match else None

    # Try exact matches against caption_map keys
    candidates = [
        f"Figure {number}",
        f"Figure S{number}",
        f"Scheme {number}",
        f"Table {number}",
    ]

    for candidate in candidates:
        if candidate in caption_map:
            return {
                "figure_id": candidate,
                "caption": caption_map[candidate],
                "panel": panel,
            }

    # Fuzzy match: find closest key in caption_map
    best_key = None
    best_score = 0.0
    stem_lower = stem.lower()

    for key in caption_map:
        score = SequenceMatcher(None, stem_lower, key.lower()).ratio()
        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score > 0.4:
        logger.debug(
            "Fuzzy matched '%s' → '%s' (score=%.2f)",
            image_filename, best_key, best_score,
        )
        return {
            "figure_id": best_key,
            "caption": caption_map[best_key],
            "panel": panel,
        }

    logger.debug("No caption match found for %s", image_filename)
    return {"figure_id": None, "caption": None, "panel": panel}
