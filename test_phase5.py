#!/usr/bin/env python3
"""Test Phase 5 only — uses cached Phase 0-4 results, runs only aggregation.

Usage:
    python3 test_phase5.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.io_utils import setup_logging

setup_logging("INFO")

# test_2 is the smallest (~6K tokens).  Change to test_1 or test_3 for harder tests.
PAPER_DIR = (
    "/Users/mac/Documents/test_parsed_output/Source/"
    "test_2_41586_2015_BFnature15732_MOESM68_ESM/"
    "41586_2015_BFnature15732_MOESM68_ESM/hybrid_auto"
)

from pipeline import PaperPipeline

print("=" * 60)
print("Phase 5 Test — skipping Phases 0-4 (using cached results)")
print(f"Paper: {PAPER_DIR}")
print("=" * 60)

t0 = time.time()

pipeline = PaperPipeline()
result = pipeline.process_paper(
    PAPER_DIR,
    skip_phases={0, 1, 2, 3, 4},  # skip everything except Phase 5
)

elapsed = time.time() - t0

print("\n" + "=" * 60)
print(f"Phase 5 completed in {elapsed:.1f}s")

summary = result.get("paper_summary", {})
if summary.get("_fallback"):
    print("STATUS: FALLBACK (LLM failed, programmatic output used)")
elif summary.get("_error") or summary.get("_parse_error"):
    print("STATUS: ERROR")
else:
    print("STATUS: SUCCESS (LLM produced valid JSON)")

# Save output
out_path = os.path.join(PAPER_DIR, "extractions", "phase5_paper_summary.json")
print(f"\nOutput file: {out_path}")
if os.path.exists(out_path):
    size = os.path.getsize(out_path)
    print(f"Output size: {size:,} bytes")
    with open(out_path) as f:
        content = f.read()
    print(f"\nFirst 500 chars:\n{content[:500]}")
else:
    print("(no output file)")
    content = json.dumps(summary, indent=2, ensure_ascii=False)
    print(f"\nResult dict ({len(content)} chars):\n{content[:500]}")

print("\n" + "=" * 60)
print(f"Total time: {elapsed:.1f}s")
