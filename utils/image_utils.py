"""Image loading, resizing, and panel splitting utilities."""

import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from config import MAX_IMAGE_DIMENSION

logger = logging.getLogger(__name__)


def load_image_as_base64(path: str, max_dim: int = MAX_IMAGE_DIMENSION) -> str:
    """Load image, resize if needed, return base64 string."""
    img = Image.open(path)

    # Resize if largest dimension exceeds max_dim
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Resized %s from %dx%d to %dx%d", path, w, h, new_w, new_h)

    # Convert to RGB if necessary (handles RGBA, palette, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def detect_panels(image_path: str) -> list[dict]:
    """
    Attempt to detect multi-panel figures via gutter detection.

    Strategy:
      1. Convert to grayscale.
      2. Look for continuous white/near-white rows or columns (gutters).
      3. Split along those gutters to identify bounding boxes.
      4. Return list of bounding boxes with optional label guesses.

    This is a best-effort heuristic — complex panels should fall back to
    processing the full image.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    h, w = arr.shape

    WHITE_THRESH = 240
    MIN_GUTTER = 5  # minimum gutter width in pixels

    def _find_splits(axis_profile: np.ndarray, min_gap: int) -> list[int]:
        """Find split positions along an axis based on white runs."""
        is_white = axis_profile > WHITE_THRESH
        splits = []
        run_start = None
        for i, val in enumerate(is_white):
            if val and run_start is None:
                run_start = i
            elif not val and run_start is not None:
                run_len = i - run_start
                if run_len >= min_gap:
                    splits.append(run_start + run_len // 2)
                run_start = None
        return splits

    # Mean intensity per row / column
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)

    h_splits = _find_splits(row_means, MIN_GUTTER)
    v_splits = _find_splits(col_means, MIN_GUTTER)

    # Build row boundaries
    row_bounds = []
    prev = 0
    for s in h_splits:
        if s - prev > h * 0.05:  # panel must be at least 5% of image height
            row_bounds.append((prev, s))
        prev = s
    if h - prev > h * 0.05:
        row_bounds.append((prev, h))

    # Build column boundaries
    col_bounds = []
    prev = 0
    for s in v_splits:
        if s - prev > w * 0.05:
            col_bounds.append((prev, s))
        prev = s
    if w - prev > w * 0.05:
        col_bounds.append((prev, w))

    if not row_bounds:
        row_bounds = [(0, h)]
    if not col_bounds:
        col_bounds = [(0, w)]

    # Generate panel bounding boxes
    panels = []
    labels = "abcdefghijklmnopqrstuvwxyz"
    idx = 0
    for r_start, r_end in row_bounds:
        for c_start, c_end in col_bounds:
            label = labels[idx] if idx < len(labels) else str(idx)
            panels.append({
                "x": int(c_start),
                "y": int(r_start),
                "w": int(c_end - c_start),
                "h": int(r_end - r_start),
                "label_guess": label,
            })
            idx += 1

    # If only one panel detected, return empty — no splitting needed
    if len(panels) <= 1:
        return []

    logger.info("Detected %d panels in %s", len(panels), image_path)
    return panels


def crop_panel(image_path: str, bbox: dict) -> str:
    """Crop a panel from the image, return as base64."""
    img = Image.open(image_path)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cropped = img.crop((x, y, x + w, y + h))

    if cropped.mode not in ("RGB", "L"):
        cropped = cropped.convert("RGB")

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
