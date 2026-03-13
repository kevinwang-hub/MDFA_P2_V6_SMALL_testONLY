"""
Run the MOF extraction pipeline on 3 test papers (up to 10 images each).
Saves individual + combined results to ~/Downloads/.

Papers:
  test_1: 02439suppappendix  (76 images total, first 10)
  test_2: 41586_2015_BFnature15732_MOESM68_ESM  (82 images, first 10)
  test_3: banerjee.som  (111 images, first 10)

Usage: python run_3papers.py
"""

import json
import os
import sys
import time
import traceback

from utils.io_utils import setup_logging
setup_logging("INFO")

from pipeline import PaperPipeline

# ── Paper directories ──
BASE = os.path.expanduser("~/Documents/test_parsed_output/Source")

PAPERS = {
    "test_1_02439suppappendix": os.path.join(
        BASE,
        "test_1_02439suppappendix",
        "02439suppappendix",
        "hybrid_auto",
    ),
    "test_2_41586_2015_BFnature15732_MOESM68_ESM": os.path.join(
        BASE,
        "test_2_41586_2015_BFnature15732_MOESM68_ESM",
        "41586_2015_BFnature15732_MOESM68_ESM",
        "hybrid_auto",
    ),
    "test_3_banerjee.som": os.path.join(
        BASE,
        "test_3_ banerjee.som",
        "banerjee.som",
        "hybrid_auto",
    ),
}

MAX_IMAGES = 10
OUTPUT_DIR = os.path.expanduser("~/Downloads/mof_3papers_results")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = PaperPipeline()

    all_results = {}
    total_start = time.time()

    for paper_name, paper_dir in PAPERS.items():
        print(f"\n{'='*70}")
        print(f"  PROCESSING: {paper_name}")
        print(f"  Directory:  {paper_dir}")
        print(f"  Max images: {MAX_IMAGES}")
        print(f"{'='*70}\n")

        if not os.path.isdir(paper_dir):
            print(f"  ERROR: Directory not found — skipping")
            all_results[paper_name] = {"_error": "directory not found"}
            continue

        paper_start = time.time()
        try:
            result = pipe.process_paper(
                paper_dir=paper_dir,
                max_images=MAX_IMAGES,
            )
            elapsed = time.time() - paper_start

            # Save individual result
            out_path = os.path.join(OUTPUT_DIR, f"{paper_name}_results.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            n_cls = len(result.get("classifications", {}))
            n_ext = len(result.get("image_extractions", []))
            summary = result.get("paper_summary", {})
            has_summary = bool(summary)

            print(f"\n  ✓ {paper_name} DONE in {elapsed:.0f}s")
            print(f"    Classifications: {n_cls}")
            print(f"    Extractions:     {n_ext}")
            print(f"    Phase 5 summary: {'Yes' if has_summary else 'No'}")
            print(f"    Saved to: {out_path}")

            all_results[paper_name] = result

        except Exception as e:
            elapsed = time.time() - paper_start
            print(f"\n  ✗ {paper_name} FAILED after {elapsed:.0f}s: {e}")
            traceback.print_exc()
            all_results[paper_name] = {"_error": str(e)}

    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_3papers_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {total_elapsed:.0f}s total")
    print(f"  Combined results: {combined_path}")
    print(f"  Individual results in: {OUTPUT_DIR}/")
    print(f"{'='*70}")

    # Quick summary table
    print(f"\n{'Paper':<50} {'Cls':>5} {'Ext':>5} {'Ph5':>5}")
    print("-" * 70)
    for name, res in all_results.items():
        if "_error" in res:
            print(f"{name:<50} {'ERR':>5} {'ERR':>5} {'ERR':>5}")
        else:
            n_c = len(res.get("classifications", {}))
            n_e = len(res.get("image_extractions", []))
            p5 = "Yes" if res.get("paper_summary") else "No"
            print(f"{name:<50} {n_c:>5} {n_e:>5} {p5:>5}")


if __name__ == "__main__":
    main()
