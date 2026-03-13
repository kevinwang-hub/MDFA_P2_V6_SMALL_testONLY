"""
Run pipeline on test_2 paper with only the 7 specified images.
Usage: python run_test2.py
"""
import json
import os
import sys

# Setup logging first
from utils.io_utils import setup_logging
setup_logging("INFO")

import pipeline as _pipeline
from pipeline import PaperPipeline, _find_images as _original_find_images

PAPER_DIR = os.path.expanduser(
    "~/Documents/test_parsed_output/Source/"
    "test_2_41586_2015_BFnature15732_MOESM68_ESM/"
    "41586_2015_BFnature15732_MOESM68_ESM/hybrid_auto"
)

# Only process this 1 image
TARGET_HASHES = [
    "64bb925a7860d7aa025204c8375bee4aa4ff3f7276f1368dc8f207a499b29e71",
]

def _find_images_filtered(paper_dir: str) -> list[str]:
    """Return only the 7 target images from the paper directory."""
    all_images = _original_find_images(paper_dir)
    from pathlib import Path
    filtered = [
        img for img in all_images
        if Path(img).stem in TARGET_HASHES
    ]
    print(f"\n=== Filtered to {len(filtered)}/{len(all_images)} images ===")
    for f in filtered:
        print(f"  {os.path.basename(f)}")
    print()
    return filtered

# Monkey-patch _find_images in the pipeline module
_pipeline._find_images = _find_images_filtered

OUTPUT_PATH = os.path.expanduser("~/Downloads/test2_v6_1img_64bb_results.json")

print(f"Paper dir: {PAPER_DIR}")
print(f"Output: {OUTPUT_PATH}")
print(f"Processing {len(TARGET_HASHES)} image...\n")

pipe = PaperPipeline()
result = pipe.process_paper(paper_dir=PAPER_DIR)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"DONE. Result saved to: {OUTPUT_PATH}")
print(f"Classifications: {len(result.get('classifications', {}))}")
print(f"Extractions: {len(result.get('image_extractions', []))}")
print(f"{'='*60}")
