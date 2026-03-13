"""Orchestrator: process a single paper through all 5 phases."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from config import (
    SAVE_INTERMEDIATE,
    SKIP_LOW_RELEVANCE_THRESHOLD,
    SUPPORTED_IMAGE_FORMATS,
)
from models.gemma_vl import GemmaVLClient
from models.qwen_text import QwenTextClient
from models.qwen_vl import QwenVLClient
from phase0_context.chunker import chunk_content
from phase0_context.loader import load_paper_content
from phase0_context.retriever import ContextRetriever
from phase1_scout.classifier import ImageClassifier
from phase1_scout.router import Router
from phase2_context.context_assembler import ContextAssembler
from phase3_extractor.extractor import Extractor
from phase4_critic.verifier import Verifier
from phase5_synthesizer.aggregator import Aggregator, fallback_synthesis
from utils.io_utils import load_json, save_json

logger = logging.getLogger(__name__)


def _find_content_list(paper_dir: str) -> str | None:
    """Find the content_list_v2.json file in a paper directory (searches recursively).
    Prefers files containing 'content_list_v2' over plain 'content_list'."""
    p = Path(paper_dir)
    candidates = []
    # Direct children first
    for f in p.iterdir():
        if f.is_file() and "content_list" in f.name and f.suffix == ".json":
            candidates.append(str(f))
    # Recurse one level into subdirectories
    if not candidates:
        for d in p.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file() and "content_list" in f.name and f.suffix == ".json":
                        candidates.append(str(f))
    if not candidates:
        return None
    # Prefer v2 over non-v2
    v2 = [c for c in candidates if "content_list_v2" in c]
    return v2[0] if v2 else candidates[0]


def _find_images(paper_dir: str) -> list[str]:
    """Find all supported image files in images/ subfolder (searches recursively)."""
    p = Path(paper_dir)
    candidates = [p / "images"]
    # Also check one level deeper for nested structure
    for d in p.iterdir():
        if d.is_dir():
            sub_images = d / "images"
            if sub_images.is_dir():
                candidates.append(sub_images)

    images = []
    for images_dir in candidates:
        if not images_dir.is_dir():
            continue
        for f in sorted(images_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                images.append(str(f))
    return sorted(images)


def _extract_paper_metadata(items: list[dict]) -> dict:
    """Extract basic paper metadata from content items."""
    title = ""
    for item in items:
        if item["section_type"] == "title" and item["text"].strip():
            title = item["text"].strip()
            break

    return {
        "title": title,
        "authors": "",
        "doi": "",
        "journal": "",
    }


class PaperPipeline:
    """Process a single paper through all five extraction phases."""

    def __init__(self):
        self.qwen_vl = QwenVLClient()
        self.gemma_vl = GemmaVLClient()
        self.qwen_text = QwenTextClient()

        self.classifier = ImageClassifier(client=self.qwen_vl)
        self.router = Router()
        self.extractor = Extractor(gemma_client=self.gemma_vl, qwen_vl_client=self.qwen_vl)
        self.verifier = Verifier(client=self.gemma_vl)
        self.aggregator = Aggregator(client=self.qwen_text)

    def process_paper(
        self,
        paper_dir: str,
        skip_phases: set[int] | None = None,
        only_phase: int | None = None,
        only_image: str | None = None,
        dry_run: bool = False,
        max_images: int | None = None,
    ) -> dict:
        """
        Process one paper folder through Phases 0–5.

        Args:
            paper_dir: Path to the paper directory.
            skip_phases: Set of phase numbers to skip (use cached results).
            only_phase: If set, run only this phase.
            only_image: If set with only_phase, process only this image.
            dry_run: If True, show what would be processed without calling models.

        Returns:
            Paper-level aggregation result (Phase 5 output).
        """
        paper_dir = str(Path(paper_dir).resolve())
        skip_phases = skip_phases or set()
        extractions_dir = os.path.join(paper_dir, "extractions")

        logger.info("Processing paper: %s", paper_dir)

        # ── Phase 0: Context Preparation ──
        chunks = []
        caption_map = {}
        image_map = {}
        retriever = None
        items = []

        if only_phase is None or only_phase == 0:
            if 0 not in skip_phases:
                logger.info("Phase 0: Loading context...")
                content_list_path = _find_content_list(paper_dir)
                if content_list_path:
                    items, image_map = load_paper_content(content_list_path)
                    chunks, caption_map = chunk_content(items)
                    # Merge image_map from loader into caption_map
                    for fname, cap in image_map.items():
                        if cap.strip():
                            caption_map[fname] = cap
                    retriever = ContextRetriever(chunks)
                    if SAVE_INTERMEDIATE:
                        save_json(
                            {"chunks": chunks, "caption_map": caption_map},
                            os.path.join(extractions_dir, "phase0_context.json"),
                        )
                else:
                    logger.warning("No content_list JSON found in %s", paper_dir)
            else:
                cached = os.path.join(extractions_dir, "phase0_context.json")
                if os.path.exists(cached):
                    data = load_json(cached)
                    chunks = data["chunks"]
                    caption_map = data["caption_map"]
                    retriever = ContextRetriever(chunks)

        if only_phase == 0:
            return {"phase0": {"chunks": len(chunks), "captions": len(caption_map)}}

        # ── Phase 1: Classification ──
        images = _find_images(paper_dir)
        if max_images is not None:
            images = images[:max_images]
        classifications = {}

        if only_phase is None or only_phase == 1:
            if 1 not in skip_phases:
                logger.info("Phase 1: Classifying %d images...", len(images))
                if dry_run:
                    logger.info("[DRY RUN] Would classify: %s", images)
                else:
                    for img_path in images:
                        try:
                            cls_result = self.classifier.classify(img_path)
                            classifications[img_path] = cls_result
                        except Exception:
                            logger.exception("Classification failed for %s", img_path)
                            classifications[img_path] = {
                                "_error": True,
                                "primary_type": "other",
                                "relevance_to_synthesis": 0,
                            }

                    if SAVE_INTERMEDIATE:
                        save_json(
                            classifications,
                            os.path.join(extractions_dir, "phase1_classification.json"),
                        )
            else:
                cached = os.path.join(extractions_dir, "phase1_classification.json")
                if os.path.exists(cached):
                    classifications = load_json(cached)

        if only_phase == 1:
            return {"phase1": classifications}

        # ── Route and sort by priority ──
        routed_images = []
        for img_path, cls_result in classifications.items():
            routing = self.router.route(cls_result)
            cls_result["extraction_prompt_key"] = routing["extraction_prompt_key"]
            routed_images.append({
                "image_path": img_path,
                "classification": cls_result,
                "routing": routing,
            })

        routed_images.sort(key=lambda x: x["routing"]["priority"], reverse=True)

        if only_image:
            routed_images = [
                r for r in routed_images
                if Path(r["image_path"]).name == only_image
            ]

        # ── Phases 2-4: Per-image processing ──
        context_assembler = ContextAssembler(retriever, caption_map) if retriever else None
        verified_extractions = []

        for entry in routed_images:
            img_path = entry["image_path"]
            cls_result = entry["classification"]
            routing = entry["routing"]
            img_name = Path(img_path).stem

            if routing["skip_extraction"]:
                logger.info("Skipping %s (low relevance)", img_path)
                continue

            if dry_run:
                logger.info(
                    "[DRY RUN] Would process: %s (type=%s, priority=%d)",
                    img_path, cls_result.get("primary_type"), routing["priority"],
                )
                continue

            try:
                # Phase 2: Context assembly
                context = {"assembled_prompt_context": "", "caption": None, "figure_id": None}
                if (only_phase is None or only_phase == 2) and 2 not in skip_phases:
                    if context_assembler:
                        context = context_assembler.assemble(
                            Path(img_path).name, cls_result
                        )
                        if SAVE_INTERMEDIATE:
                            save_json(
                                context,
                                os.path.join(
                                    extractions_dir, "phase2_context",
                                    f"{img_name}_context.json",
                                ),
                            )
                else:
                    cached = os.path.join(
                        extractions_dir, "phase2_context",
                        f"{img_name}_context.json",
                    )
                    if os.path.exists(cached):
                        context = load_json(cached)

                if only_phase == 2:
                    continue

                # Phase 3: Extraction
                extraction = {}
                if (only_phase is None or only_phase == 3) and 3 not in skip_phases:
                    extraction = self.extractor.extract(
                        img_path, cls_result, context, routing["extraction_model"],
                    )
                    if SAVE_INTERMEDIATE:
                        save_json(
                            extraction,
                            os.path.join(
                                extractions_dir, "phase3_extraction",
                                f"{img_name}_extraction.json",
                            ),
                        )
                else:
                    cached = os.path.join(
                        extractions_dir, "phase3_extraction",
                        f"{img_name}_extraction.json",
                    )
                    if os.path.exists(cached):
                        extraction = load_json(cached)

                if only_phase == 3:
                    continue

                # Phase 4: Verification
                verified = {}
                if (only_phase is None or only_phase == 4) and 4 not in skip_phases:
                    free_text = extraction.get("_free_text", "")
                    verified = self.verifier.verify(
                        img_path, extraction, context,
                        free_extraction_text=free_text,
                    )
                    if SAVE_INTERMEDIATE:
                        save_json(
                            verified,
                            os.path.join(
                                extractions_dir, "phase4_verification",
                                f"{img_name}_verified.json",
                            ),
                        )
                else:
                    cached = os.path.join(
                        extractions_dir, "phase4_verification",
                        f"{img_name}_verified.json",
                    )
                    if os.path.exists(cached):
                        verified = load_json(cached)

                # Use corrected extraction if available, otherwise original
                corrected = verified.get("corrected_extraction")
                final_extraction = corrected if corrected else extraction
                final_extraction["_figure_id"] = context.get("figure_id")
                final_extraction["_image_path"] = img_path
                # Preserve Stage 1 free-form text from two-stage extraction
                if "_free_text" not in final_extraction and "_free_text" in extraction:
                    final_extraction["_free_text"] = extraction["_free_text"]
                verified_extractions.append(final_extraction)

            except Exception:
                logger.exception("Processing failed for %s", img_path)
                continue

        if only_phase in (2, 3, 4):
            return {"processed_images": len(verified_extractions)}

        # ── Phase 5: Aggregation ──
        paper_summary = {}
        if (only_phase is None or only_phase == 5) and 5 not in skip_phases:
            if verified_extractions:
                logger.info(
                    "Phase 5: Aggregating %d extractions...",
                    len(verified_extractions),
                )
                if dry_run:
                    logger.info("[DRY RUN] Would aggregate %d extractions", len(verified_extractions))
                else:
                    try:
                        paper_metadata = _extract_paper_metadata(items)

                        # Build context summary from key sections (capped at ~1K tokens)
                        key_sections = ["abstract", "experimental", "results", "discussion"]
                        context_parts = []
                        total_len = 0
                        for chunk in chunks:
                            if chunk["section_type"] in key_sections:
                                text = chunk["text"]
                                if total_len + len(text) > 4000:
                                    text = text[:4000 - total_len]
                                    context_parts.append(text)
                                    break
                                context_parts.append(text)
                                total_len += len(text)
                        paper_context_summary = "\n\n".join(context_parts)

                        paper_summary = self.aggregator.aggregate(
                            verified_extractions, paper_metadata, paper_context_summary,
                        )
                        if SAVE_INTERMEDIATE:
                            save_json(
                                paper_summary,
                                os.path.join(extractions_dir, "phase5_paper_summary.json"),
                            )
                    except Exception:
                        logger.exception("Phase 5 aggregation failed — using fallback")
                        paper_metadata = _extract_paper_metadata(items)
                        paper_summary = fallback_synthesis(verified_extractions, paper_metadata)
                        if SAVE_INTERMEDIATE:
                            save_json(
                                paper_summary,
                                os.path.join(extractions_dir, "phase5_paper_summary.json"),
                            )
            else:
                logger.warning("No verified extractions to aggregate.")

        return {
            "paper_summary": paper_summary,
            "image_extractions": verified_extractions,
            "classifications": classifications,
        }

    def process_batch(
        self,
        papers_root: str,
        max_workers: int = 1,
    ) -> list[dict]:
        """
        Process all paper folders under papers_root.

        Each paper is processed sequentially internally.
        max_workers > 1 enables parallel processing across papers.
        """
        papers_root = Path(papers_root)
        paper_dirs = sorted(
            d for d in papers_root.iterdir()
            if d.is_dir() and (d / "images").is_dir()
        )

        logger.info("Found %d papers to process in %s", len(paper_dirs), papers_root)

        if max_workers <= 1:
            results = []
            for paper_dir in paper_dirs:
                try:
                    result = self.process_paper(str(paper_dir))
                    results.append(result)
                except Exception:
                    logger.exception("Failed to process %s", paper_dir)
                    results.append({"_error": True, "_paper": str(paper_dir)})
            return results

        # Parallel processing across papers
        # Note: each worker creates its own model clients
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_paper_standalone, str(d)): str(d)
                for d in paper_dirs
            }
            for future in futures:
                paper = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    logger.exception("Failed to process %s", paper)
                    results.append({"_error": True, "_paper": paper})

        return results


def _process_paper_standalone(paper_dir: str) -> dict:
    """Standalone function for parallel processing (creates its own pipeline)."""
    pipeline = PaperPipeline()
    return pipeline.process_paper(paper_dir)
