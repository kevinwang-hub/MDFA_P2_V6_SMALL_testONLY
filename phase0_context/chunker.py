"""Chunk text by section with metadata, and build a caption map."""

import logging
import re

logger = logging.getLogger(__name__)

# Max words per chunk before splitting
MAX_CHUNK_WORDS = 500
# Target words when splitting oversized groups
TARGET_CHUNK_WORDS = 300

# Pattern to detect captions
CAPTION_START_PATTERN = re.compile(
    r"^(Figure|Fig\.|Scheme|Table)\s+S?\d+", re.IGNORECASE
)
# Pattern to extract figure identifier from caption
FIGURE_ID_PATTERN = re.compile(
    r"((?:Figure|Fig\.?|Scheme|Table)\s+S?\d+)", re.IGNORECASE
)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_at_sentence_boundary(text: str, target_words: int) -> list[str]:
    """Split text at sentence boundaries, targeting ~target_words per chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sw = _word_count(sentence)
        if current_words + sw > target_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = sw
        else:
            current.append(sentence)
            current_words += sw

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_content(items: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """
    Produce BM25-ready chunks and a caption map from normalized content items.

    Returns:
        (chunks, caption_map)
        chunks: list of dicts with chunk_id, text, section_type, pages, figure_refs
        caption_map: dict mapping figure identifiers (e.g., "Figure 1") to full caption text
    """
    # Step 1: Group consecutive items by section_type
    groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_type: str | None = None

    for item in items:
        if item["section_type"] != current_type:
            if current_group:
                groups.append(current_group)
            current_group = [item]
            current_type = item["section_type"]
        else:
            current_group.append(item)

    if current_group:
        groups.append(current_group)

    # Step 2: Build chunks
    chunks: list[dict] = []
    chunk_id = 0

    for group in groups:
        section_type = group[0]["section_type"]
        combined_text = " ".join(item["text"] for item in group)
        pages = sorted(set(item["page"] for item in group))
        figure_refs = sorted(
            set(ref for item in group for ref in item["figure_refs"])
        )

        if _word_count(combined_text) > MAX_CHUNK_WORDS:
            parts = _split_at_sentence_boundary(combined_text, TARGET_CHUNK_WORDS)
            for part in parts:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": part,
                    "section_type": section_type,
                    "pages": pages,
                    "figure_refs": figure_refs,
                })
                chunk_id += 1
        else:
            chunks.append({
                "chunk_id": chunk_id,
                "text": combined_text,
                "section_type": section_type,
                "pages": pages,
                "figure_refs": figure_refs,
            })
            chunk_id += 1

    # Step 3: Build caption map
    caption_map = _build_caption_map(items)

    logger.info(
        "Built %d chunks and %d caption entries", len(chunks), len(caption_map)
    )
    return chunks, caption_map


def _build_caption_map(items: list[dict]) -> dict[str, str]:
    """
    Build a mapping from figure identifiers to their full caption text.

    Captions are items where section_type == "caption" or text matches
    the caption start pattern.
    """
    caption_map: dict[str, str] = {}

    for item in items:
        text = item["text"].strip()
        is_caption = (
            item["section_type"] == "caption"
            or CAPTION_START_PATTERN.match(text)
        )
        if not is_caption:
            continue

        match = FIGURE_ID_PATTERN.search(text)
        if match:
            fig_id = _normalize_figure_id(match.group(1))
            caption_map[fig_id] = text

    return caption_map


def _normalize_figure_id(raw: str) -> str:
    """Normalize 'Fig. 1', 'Figure 1', 'Fig 1' to a canonical form."""
    raw = raw.strip()
    raw = re.sub(r"\bFig\.?\s*", "Figure ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw)
    return raw
