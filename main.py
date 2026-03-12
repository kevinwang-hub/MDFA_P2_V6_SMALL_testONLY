"""
CLI entry point for the MOF Image Extraction Pipeline.

Usage:
  python main.py --paper papers/paper_001/
  python main.py --batch papers/ --workers 4
  python main.py --paper papers/paper_001/ --skip-phases 0,1
  python main.py --paper papers/paper_001/ --only-phase 3 --image fig2.png
  python main.py --paper papers/paper_001/ --dry-run
"""

import argparse
import json
import sys

from config import LOG_LEVEL
from utils.io_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="MOF Image Extraction Pipeline — extract structured data from research paper images."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--paper", type=str,
        help="Process a single paper directory.",
    )
    mode.add_argument(
        "--batch", type=str,
        help="Process all paper directories under this root.",
    )

    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for batch mode (default: 1).",
    )
    parser.add_argument(
        "--skip-phases", type=str, default="",
        help="Comma-separated phase numbers to skip, using cached results (e.g., '0,1').",
    )
    parser.add_argument(
        "--only-phase", type=int, default=None,
        help="Run only this phase number (0-5).",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Process only this image filename (use with --only-phase).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Process at most this many images (sorted alphabetically).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write final JSON result to this file path.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without calling models.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Setup logging
    level = "DEBUG" if args.verbose else LOG_LEVEL
    setup_logging(level)

    # Parse skip-phases
    skip_phases = set()
    if args.skip_phases:
        skip_phases = {int(x.strip()) for x in args.skip_phases.split(",")}

    # Import pipeline here to avoid loading models before logging is set up
    from pipeline import PaperPipeline

    pipeline = PaperPipeline()

    if args.paper:
        result = pipeline.process_paper(
            paper_dir=args.paper,
            skip_phases=skip_phases,
            only_phase=args.only_phase,
            only_image=args.image,
            dry_run=args.dry_run,
            max_images=args.max_images,
        )
        output_text = json.dumps(result, indent=2, ensure_ascii=False)
        print(output_text)
        if args.output:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"\nOutput saved to: {args.output}")

    elif args.batch:
        results = pipeline.process_batch(
            papers_root=args.batch,
            max_workers=args.workers,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
