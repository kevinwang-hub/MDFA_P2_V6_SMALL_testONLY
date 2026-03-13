"""
Run pipeline on test_1 (02439suppappendix) with specified images only.
Usage: python run_test1.py
"""
import json
import os
import sys
from pathlib import Path

from utils.io_utils import setup_logging
setup_logging("INFO")

import pipeline as _pipeline
from pipeline import PaperPipeline, _find_images as _original_find_images

PAPER_DIR = os.path.expanduser(
    "~/Documents/test_parsed_output/Source/"
    "test_1_02439suppappendix/"
    "02439suppappendix/hybrid_auto"
)

TARGET_HASHES = [
    "0a9a31014065a1f5eab6d0363a21ac51b58d1b53f1b79f8c20cb623faf7932a9",
]

def _find_images_filtered(paper_dir: str) -> list[str]:
    all_images = _original_find_images(paper_dir)
    filtered = [
        img for img in all_images
        if Path(img).stem in TARGET_HASHES
    ]
    print(f"\n=== Filtered to {len(filtered)}/{len(all_images)} images ===")
    for f in filtered:
        print(f"  {os.path.basename(f)}")
    print()
    return filtered

_pipeline._find_images = _find_images_filtered

OUTPUT_PATH = os.path.expanduser("~/Downloads/test1_1img_0a9a_results.json")

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
