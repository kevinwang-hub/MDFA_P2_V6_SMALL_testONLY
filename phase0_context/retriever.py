"""BM25 index builder and query interface."""

import logging
import re
import string

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class ContextRetriever:
    """BM25-based retrieval over paper chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info("Built BM25 index over %d chunks", len(chunks))

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks ranked by BM25 relevance to query_text."""
        tokens = _tokenize(query_text)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(score)
            results.append(chunk)

        return results

    def query_by_section(self, section_types: list[str]) -> list[dict]:
        """Return all chunks matching given section types."""
        return [
            chunk
            for chunk in self.chunks
            if chunk["section_type"] in section_types
        ]
