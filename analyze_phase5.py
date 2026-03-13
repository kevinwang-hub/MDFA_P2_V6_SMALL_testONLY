"""Analyze Phase 5 failures across all 3 papers."""
import json, os, glob

papers = {
    "test_1": "/Users/mac/Documents/test_parsed_output/Source/test_1_02439suppappendix/02439suppappendix/hybrid_auto/extractions",
    "test_2": "/Users/mac/Documents/test_parsed_output/Source/test_2_41586_2015_BFnature15732_MOESM68_ESM/41586_2015_BFnature15732_MOESM68_ESM/hybrid_auto/extractions",
    "test_3": "/Users/mac/Documents/test_parsed_output/Source/test_3_ banerjee.som/banerjee.som/hybrid_auto/extractions",
}

for name, edir in papers.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Phase 1
    p1 = os.path.join(edir, "phase1_classification.json")
    if os.path.exists(p1):
        with open(p1) as f:
            d1 = json.load(f)
        types = {}
        for k, v in d1.items():
            t = v.get("primary_type", "unknown")
            types[t] = types.get(t, 0) + 1
        rel_scores = [v.get("relevance_to_synthesis", 0) for v in d1.values()]
        skipped = sum(1 for r in rel_scores if r <= 1)
        print(f"  Phase 1: {len(d1)} classified")
        print(f"    Types: {types}")
        print(f"    Relevance scores: {sorted(rel_scores, reverse=True)}")
        print(f"    Would skip (relevance<=1): {skipped}")
    else:
        print(f"  Phase 1: NOT FOUND")

    # Phase 3
    p3dir = os.path.join(edir, "phase3_extraction")
    if os.path.isdir(p3dir):
        p3files = glob.glob(os.path.join(p3dir, "*.json"))
        print(f"  Phase 3: {len(p3files)} extractions")
        # Calculate total size of extractions
        total_bytes = sum(os.path.getsize(f) for f in p3files)
        print(f"    Total extraction data: {total_bytes:,} bytes")
    else:
        print(f"  Phase 3: NOT FOUND")

    # Phase 4
    p4dir = os.path.join(edir, "phase4_verification")
    if os.path.isdir(p4dir):
        p4files = glob.glob(os.path.join(p4dir, "*.json"))
        print(f"  Phase 4: {len(p4files)} verified")
        total_bytes = sum(os.path.getsize(f) for f in p4files)
        print(f"    Total verification data: {total_bytes:,} bytes")
    else:
        print(f"  Phase 4: NOT FOUND")

    # Phase 5
    p5 = os.path.join(edir, "phase5_paper_summary.json")
    if os.path.exists(p5):
        with open(p5) as f:
            d5 = json.load(f)
        size = len(json.dumps(d5))
        has_parse_err = "_parse_error" in d5
        has_err = "_error" in d5
        print(f"  Phase 5: EXISTS ({size} bytes)")
        print(f"    Keys: {list(d5.keys())}")
        if has_parse_err:
            print(f"    PARSE ERROR: {d5['_parse_error']}")
        if has_err:
            print(f"    ERROR: {d5['_error']}")
        # Check timestamp
        mtime = os.path.getmtime(p5)
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime)
        print(f"    Last modified: {ts}")
    else:
        print(f"  Phase 5: NOT FOUND")

    # Estimate Phase 5 prompt size
    p4files_list = glob.glob(os.path.join(edir, "phase4_verification", "*.json")) if os.path.isdir(os.path.join(edir, "phase4_verification")) else []
    if p4files_list:
        total_extraction_chars = 0
        for pf in p4files_list:
            with open(pf) as f:
                d = json.load(f)
            corrected = d.get("corrected_extraction", d)
            total_extraction_chars += len(json.dumps(corrected, indent=2))
        print(f"\n  >> Phase 5 prompt size estimate:")
        print(f"     Extraction JSON payload: ~{total_extraction_chars:,} chars (~{total_extraction_chars//4:,} tokens)")
        print(f"     + System prompt template: ~3,000 chars (~750 tokens)")
        print(f"     Estimated total input: ~{(total_extraction_chars + 3000)//4:,} tokens")

# Summary
print(f"\n{'='*60}")
print(f"  PHASE 5 FAILURE ANALYSIS")
print(f"{'='*60}")
print("""
ROOT CAUSE: qwen3.5:9b timeout on large aggregation prompts

The Phase 5 aggregation sends ALL verified extractions (as JSON) to 
qwen3.5:9b in a single prompt. With 9-10 images worth of extraction 
data, the input is very large and the model must generate an equally 
large structured JSON output.

TIMELINE from run logs:
  - test_1: 10 extractions -> Phase 5 started at 02:58
    * Attempt 1/3: timed out at 03:28 (600s)
    * Attempt 2/3: timed out at 03:58 (600s)  
    * Attempt 3/3: timed out at 04:28 (600s)
    * RESULT: RuntimeError -> test_1 result NOT saved to Downloads
    
  - test_2: 5 extractions -> Phase 5 completed at 06:38
    * Succeeded (fewer extractions = smaller prompt)
    * BUT output quality is poor (645 bytes, mostly nulls)
    
  - test_3: 9 extractions -> Phase 5 started at 07:33
    * Pipeline was still running when cancelled
    * No phase5_paper_summary.json saved

CONTRIBUTING FACTORS:
  1. THINKING MODE: qwen3.5:9b uses "thinking" by default, which 
     generates long internal reasoning BEFORE the actual JSON output.
     This eats into the timeout without producing visible output.
  
  2. TIMEOUT TOO SHORT: 600s (10 min) timeout is insufficient for 
     9B model doing thinking + generating ~8K token JSON output on M1.
  
  3. PROMPT TOO LARGE: The aggregation template includes a massive 
     JSON schema (~3KB) + all extraction data. Combined input can 
     exceed the model's effective context window for quality output.
  
  4. NO ERROR HANDLING: The original pipeline.py did not have 
     try/except around Phase 5, so a timeout crash prevented the 
     Phase 0-4 results from being saved.

FIXES ALREADY APPLIED:
  - /no_think added to System prompt (disables thinking mode)
  - Timeout increased from 600s -> 1200s
  - try/except added around Phase 5 in pipeline.py
""")
