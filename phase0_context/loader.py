"""Parse content_list_v2.json and normalize into a list of content items.

Handles the page-based nested format where:
  - Top level: list of pages
  - Each page: list of items with {type, content, bbox}
  - content is a dict with type-specific keys (paragraph_content, title_content,
    image_source/image_caption/image_footnote, list_items, etc.)
  - Inline elements in content lists have {type: "text"|"equation_inline", content: str}
"""

import logging
import re

from utils.io_utils import load_json

logger = logging.getLogger(__name__)

# Regex to find figure/table/scheme references in text
FIGURE_REF_PATTERN = re.compile(
    r"(?:Figure|Fig\.?|Scheme|Table)\s*(?:S?\d+[a-z]?(?:\s*[-–,]\s*[a-z\d])*)",
    re.IGNORECASE,
)


def _extract_figure_refs(text: str) -> list[str]:
    """Extract all figure/table/scheme references from text."""
    return FIGURE_REF_PATTERN.findall(text)


def _flatten_inline_elements(elements) -> str:
    """Flatten a list of inline elements ({type, content}) into plain text."""
    if isinstance(elements, str):
        return elements
    if not isinstance(elements, list):
        return str(elements) if elements else ""
    parts = []
    for el in elements:
        if isinstance(el, dict):
            parts.append(str(el.get("content", "")))
        elif isinstance(el, str):
            parts.append(el)
    return " ".join(parts)


def _extract_item_text(item: dict) -> str:
    """Extract plain text from an item based on its type."""
    item_type = item.get("type", "")
    content = item.get("content", {})

    if not isinstance(content, dict):
        return str(content) if content else ""

    if item_type == "paragraph":
        return _flatten_inline_elements(content.get("paragraph_content", ""))

    if item_type == "title":
        return _flatten_inline_elements(content.get("title_content", ""))

    if item_type == "image":
        caption = _flatten_inline_elements(content.get("image_caption", []))
        footnote = _flatten_inline_elements(content.get("image_footnote", []))
        return (caption + " " + footnote).strip()

    if item_type == "list":
        list_items = content.get("list_items", [])
        parts = []
        for li in list_items:
            if isinstance(li, dict):
                parts.append(_flatten_inline_elements(li.get("item_content", "")))
            else:
                parts.append(str(li))
        return " ".join(parts)

    if item_type == "table":
        return str(content)[:2000]

    if item_type in ("page_header", "page_footer", "page_footnote", "page_aside_text"):
        key = f"{item_type}_content"
        return _flatten_inline_elements(content.get(key, ""))

    if item_type == "page_number":
        return _flatten_inline_elements(content.get("page_number_content", ""))

    return ""


def load_paper_content(content_list_path: str) -> tuple[list[dict], dict[str, str]]:
    """
    Load content_list_v2.json and normalize into a list of dicts.

    Handles two formats:
      - Page-based: list[list[{type, content, bbox}]]  (actual format)
      - Flat: list[{text, page, section_type, ...}]    (legacy)

    Returns:
        (items, image_map)
        items: list of {text, page, section_type, figure_refs}
        image_map: dict mapping image filename to caption text
    """
    raw = load_json(content_list_path)

    # Detect format
    if isinstance(raw, dict):
        for key in ("content", "items", "content_list", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    # Page-based nested format: list of lists
    if raw and isinstance(raw[0], list):
        return _parse_page_format(raw)

    # Flat format fallback
    return _parse_flat_format(raw), {}


def _parse_page_format(pages: list[list]) -> tuple[list[dict], dict[str, str]]:
    """Parse the page-based nested format."""
    items = []
    image_map = {}  # image filename -> caption text
    current_section = "other"

    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1  # 1-indexed

        # Try to get actual page number from page_number items
        for item in page:
            if item.get("type") == "page_number":
                pn_text = _extract_item_text(item).strip()
                try:
                    page_num = int(pn_text)
                except ValueError:
                    pass

        for item in page:
            item_type = item.get("type", "")
            text = _extract_item_text(item)

            if not text.strip() and item_type != "image":
                continue

            # Track current section from titles
            if item_type == "title":
                current_section = _infer_section_from_title(text)

            # Determine section_type for this item
            section_type = _map_item_type(item_type, text, current_section)

            # Handle images: build image_map
            if item_type == "image":
                content = item.get("content", {})
                if isinstance(content, dict):
                    src = content.get("image_source", {})
                    if isinstance(src, dict):
                        img_path = src.get("path", "")
                        img_filename = img_path.rsplit("/", 1)[-1] if img_path else ""
                        caption_text = _flatten_inline_elements(
                            content.get("image_caption", [])
                        )
                        if img_filename:
                            image_map[img_filename] = caption_text

                        # If there's caption text, add as content item
                        if caption_text.strip():
                            items.append({
                                "text": caption_text,
                                "page": page_num,
                                "section_type": "caption",
                                "figure_refs": _extract_figure_refs(caption_text),
                            })
                continue

            # Skip page furniture
            if item_type in ("page_number", "page_header", "page_footer", "page_aside_text"):
                continue

            figure_refs = _extract_figure_refs(text)
            items.append({
                "text": text,
                "page": page_num,
                "section_type": section_type,
                "figure_refs": figure_refs,
            })

    logger.info(
        "Loaded %d content items and %d image entries from page-based format",
        len(items), len(image_map),
    )
    return items, image_map


def _parse_flat_format(entries: list) -> list[dict]:
    """Parse legacy flat format."""
    items = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", entry.get("content", entry.get("body", "")))
        page = entry.get("page_number", entry.get("page", entry.get("page_num", 0)))
        section_type = _map_item_type(
            entry.get("type", ""), str(text), "other"
        )
        figure_refs = _extract_figure_refs(str(text))
        items.append({
            "text": str(text),
            "page": int(page) if page else 0,
            "section_type": section_type,
            "figure_refs": figure_refs,
        })
    logger.info("Loaded %d content items from flat format", len(items))
    return items


def _infer_section_from_title(title_text: str) -> str:
    """Infer section type from a title/heading."""
    t = title_text.lower().strip()
    section_keywords = {
        "abstract": "abstract",
        "introduction": "introduction",
        "experiment": "experimental",
        "materials and methods": "experimental",
        "synthesis": "experimental",
        "results and discussion": "results",
        "results": "results",
        "discussion": "discussion",
        "supporting": "SI",
        "supplementary": "SI",
        "references": "references",
        "conclusion": "discussion",
        "characterization": "results",
        "high-throughput": "results",
    }
    for keyword, stype in section_keywords.items():
        if keyword in t:
            return stype
    return "other"


def _map_item_type(item_type: str, text: str, current_section: str) -> str:
    """Map a content_list_v2 item type to our canonical section_type."""
    direct = {
        "title": "title",
        "table": "table",
        "image": "caption",
        "page_footnote": "other",
    }
    if item_type in direct:
        return direct[item_type]

    text_lower = text.lower().strip()

    # Check if text looks like a caption
    if re.match(r"^(figure|fig\.|scheme|table)\s+s?\d+", text_lower):
        return "caption"

    # Check for references section items
    if current_section == "references":
        return "references"

    # Paragraphs and lists inherit the current section
    if item_type in ("paragraph", "list"):
        return current_section

    return "other"
