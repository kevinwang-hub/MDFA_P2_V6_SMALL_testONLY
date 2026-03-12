"""Run the pipeline for a single specified image."""
import json
import os
import sys

from utils.io_utils import setup_logging
setup_logging("INFO")

from pathlib import Path
from config import SAVE_INTERMEDIATE
from phase0_context.loader import load_paper_content
from phase0_context.chunker import chunk_content
from phase0_context.retriever import ContextRetriever
from phase1_scout.classifier import ImageClassifier
from phase1_scout.router import Router
from phase2_context.context_assembler import ContextAssembler
from phase3_extractor.extractor import Extractor
from phase4_critic.verifier import Verifier
from phase5_synthesizer.aggregator import Aggregator
from models.qwen_vl import QwenVLClient
from models.gemma_vl import GemmaVLClient
from models.qwen_text import QwenTextClient
from utils.io_utils import save_json, load_json

PAPER_DIR = os.path.expanduser(
    "~/Documents/MOF/test_parsed_output/source/test_6_science.1152516/science.1152516/hybrid_auto/"
)
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "db701208d417c1d16f203c8e35efb275eff8ef6877f11cf393c53bf7dc7cf1a4.jpg"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Downloads/mof_extraction_db7012.json")

img_path = os.path.join(PAPER_DIR, "images", IMAGE_NAME)
img_stem = Path(IMAGE_NAME).stem
extractions_dir = os.path.join(PAPER_DIR, "extractions")

# Phase 0: Load context (use cache if available)
cached_p0 = os.path.join(extractions_dir, "phase0_context.json")
if os.path.exists(cached_p0):
    print("Phase 0: Loading cached context...")
    data = load_json(cached_p0)
    chunks = data["chunks"]
    caption_map = data["caption_map"]
else:
    print("Phase 0: Building context...")
    from pipeline import _find_content_list
    content_list_path = _find_content_list(PAPER_DIR)
    items, image_map = load_paper_content(content_list_path)
    chunks, caption_map = chunk_content(items)
    for fname, cap in image_map.items():
        if cap.strip():
            caption_map[fname] = cap

retriever = ContextRetriever(chunks)

# Init models
qwen_vl = QwenVLClient()
gemma_vl = GemmaVLClient()
qwen_text = QwenTextClient()

classifier = ImageClassifier(client=qwen_vl)
router = Router()
context_assembler = ContextAssembler(retriever=retriever, caption_map=caption_map)
extractor = Extractor(gemma_client=gemma_vl, qwen_vl_client=qwen_vl)
verifier = Verifier(client=gemma_vl)

# Phase 1: Classify
print(f"Phase 1: Classifying {IMAGE_NAME}...")
cls_result = classifier.classify(img_path)
print(f"  Type: {cls_result['primary_type']}, Relevance: {cls_result['relevance_to_synthesis']}")
print(f"  Description: {cls_result.get('detailed_description', '')}")

# Route
routing = router.route(cls_result)
cls_result["extraction_prompt_key"] = routing["extraction_prompt_key"]

# Phase 2: Context
print("Phase 2: Assembling context...")
context = context_assembler.assemble(IMAGE_NAME, cls_result)
print(f"  Caption: {context.get('caption', 'None')}")
print(f"  Figure ID: {context.get('figure_id', 'None')}")

# Phase 3: Extract
print(f"Phase 3: Extracting with model={routing['extraction_model']}...")
extraction = extractor.extract(img_path, cls_result, context, routing["extraction_model"])
save_json(extraction, os.path.join(extractions_dir, "phase3_extraction", f"{img_stem}_extraction.json"))
print(f"  Extraction keys: {list(extraction.keys())[:6]}")

# Phase 4: Verify
print("Phase 4: Verifying extraction...")
verified = verifier.verify(img_path, extraction, context)
save_json(verified, os.path.join(extractions_dir, "phase4_verification", f"{img_stem}_verified.json"))
print(f"  Assessment: {verified.get('overall_assessment')}, Confidence: {verified.get('overall_confidence')}")

corrected = verified.get("corrected_extraction")
final = corrected if corrected else extraction
final["_figure_id"] = context.get("figure_id")
final["_image_path"] = img_path

# Save result
result = {
    "classification": cls_result,
    "extraction": final,
    "verification_summary": {
        "assessment": verified.get("overall_assessment"),
        "confidence": verified.get("overall_confidence"),
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nResult saved to: {OUTPUT_PATH}")
print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
