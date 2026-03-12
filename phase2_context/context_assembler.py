"""Assemble context for a single image from captions + BM25 + targeted sections."""

import logging

from config import CONTEXT_TOKEN_BUDGET
from phase0_context.retriever import ContextRetriever
from phase2_context.caption_matcher import match_caption

logger = logging.getLogger(__name__)

# Map image type to relevant paper sections
TYPE_TO_SECTIONS = {
    "synthesis_scheme": ["experimental", "SI"],
    "diffraction": ["results", "discussion", "SI"],
    "spectroscopy": ["results", "discussion", "SI"],
    "thermal_analysis": ["results", "discussion"],
    "adsorption": ["results", "discussion"],
    "crystal_structure": ["results", "SI", "experimental"],
    "microscopy": ["results", "discussion"],
    "table_figure": ["results", "experimental", "SI"],
    "computational": ["results", "discussion"],
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


class ContextAssembler:
    """Assemble paper context for a single image."""

    def __init__(
        self,
        retriever: ContextRetriever,
        caption_map: dict[str, str],
    ):
        self.retriever = retriever
        self.caption_map = caption_map

    def assemble(self, image_filename: str, classification: dict) -> dict:
        """
        Build context for a single image.

        Returns:
            {
                "caption": str or None,
                "figure_id": str or None,
                "bm25_chunks": [str, ...],
                "targeted_chunks": [str, ...],
                "assembled_prompt_context": str,
            }
        """
        # 1. Caption lookup
        caption_info = match_caption(image_filename, self.caption_map)
        caption = caption_info["caption"]
        figure_id = caption_info["figure_id"]

        # 2. BM25 retrieval
        query_parts = [
            caption or "",
            classification.get("sub_type", ""),
            classification.get("detailed_description", ""),
        ]
        query = " ".join(part for part in query_parts if part)
        bm25_results = self.retriever.query(query, top_k=5) if query.strip() else []

        # 3. Targeted section pull
        primary_type = classification.get("primary_type", "other")
        target_sections = TYPE_TO_SECTIONS.get(primary_type, ["results"])
        targeted_results = self.retriever.query_by_section(target_sections)

        # 4. Deduplicate: remove targeted chunks already in BM25 results
        bm25_ids = {c["chunk_id"] for c in bm25_results}
        targeted_results = [
            c for c in targeted_results if c["chunk_id"] not in bm25_ids
        ]

        # 5. Format assembled context
        bm25_texts = [
            f"[Section: {c['section_type']}, Page {c['pages']}] {c['text']}"
            for c in bm25_results
        ]
        targeted_texts = [
            f"[{c['section_type']}] {c['text']}"
            for c in targeted_results
        ]

        assembled = self._format_context(caption, bm25_texts, targeted_texts)

        # 6. Token budget enforcement
        assembled = self._enforce_token_budget(
            caption, bm25_texts, targeted_texts, assembled
        )

        return {
            "caption": caption,
            "figure_id": figure_id,
            "bm25_chunks": bm25_texts,
            "targeted_chunks": targeted_texts,
            "assembled_prompt_context": assembled,
        }

    def _format_context(
        self,
        caption: str | None,
        bm25_texts: list[str],
        targeted_texts: list[str],
    ) -> str:
        parts = [
            "---",
            "PAPER CONTEXT FOR THIS FIGURE",
            "---",
            "",
            "FIGURE CAPTION:",
            caption or "No caption found for this figure.",
            "",
            "RELEVANT PAPER SECTIONS:",
        ]
        parts.extend(bm25_texts)
        if targeted_texts:
            parts.append("")
            parts.append("ADDITIONAL CONTEXT:")
            parts.extend(targeted_texts)
        parts.append("---")
        return "\n".join(parts)

    def _enforce_token_budget(
        self,
        caption: str | None,
        bm25_texts: list[str],
        targeted_texts: list[str],
        assembled: str,
    ) -> str:
        """Truncate if assembled context exceeds token budget."""
        if _estimate_tokens(assembled) <= CONTEXT_TOKEN_BUDGET:
            return assembled

        # Truncate targeted chunks first (from the end)
        while targeted_texts and _estimate_tokens(assembled) > CONTEXT_TOKEN_BUDGET:
            targeted_texts.pop()
            assembled = self._format_context(caption, bm25_texts, targeted_texts)

        # Then truncate BM25 chunks from the bottom
        while bm25_texts and _estimate_tokens(assembled) > CONTEXT_TOKEN_BUDGET:
            bm25_texts.pop()
            assembled = self._format_context(caption, bm25_texts, targeted_texts)

        return assembled
