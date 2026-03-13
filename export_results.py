"""Export all available pipeline results to ~/Downloads/test2_v6_all_phases_export.json"""
import json, os, glob

EXTR = "/Users/mac/Documents/test_parsed_output/Source/test_2_41586_2015_BFnature15732_MOESM68_ESM/41586_2015_BFnature15732_MOESM68_ESM/hybrid_auto/extractions"
OUT = os.path.expanduser("~/Downloads/test2_v6_all_phases_export.json")

result = {}

# Phase 0
p0 = os.path.join(EXTR, "phase0_context.json")
if os.path.exists(p0):
    with open(p0) as f:
        d = json.load(f)
    result["phase0_summary"] = {"chunks": len(d.get("chunks", [])), "captions": len(d.get("caption_map", {}))}

# Phase 1
p1 = os.path.join(EXTR, "phase1_classification.json")
if os.path.exists(p1):
    with open(p1) as f:
        result["phase1_classifications"] = json.load(f)

# Phase 2
result["phase2_contexts"] = {}
for fp in sorted(glob.glob(os.path.join(EXTR, "phase2_context", "*_context.json"))):
    name = os.path.basename(fp).replace("_context.json", "")[:16]
    with open(fp) as fh:
        result["phase2_contexts"][name] = json.load(fh)

# Phase 3
result["phase3_extractions"] = {}
for fp in sorted(glob.glob(os.path.join(EXTR, "phase3_extraction", "*_extraction.json"))):
    name = os.path.basename(fp).replace("_extraction.json", "")[:16]
    with open(fp) as fh:
        result["phase3_extractions"][name] = json.load(fh)

# Phase 4
result["phase4_verifications"] = {}
for fp in sorted(glob.glob(os.path.join(EXTR, "phase4_verification", "*_verified.json"))):
    name = os.path.basename(fp).replace("_verified.json", "")[:16]
    with open(fp) as fh:
        result["phase4_verifications"][name] = json.load(fh)

# Phase 5
p5 = os.path.join(EXTR, "phase5_paper_summary.json")
if os.path.exists(p5):
    with open(p5) as f:
        result["phase5_paper_summary"] = json.load(f)
else:
    result["phase5_paper_summary"] = "(not completed - stuck during aggregation)"

with open(OUT, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("EXPORTED TO:", OUT)
print()
print("Phase 0: context loaded (%d chunks, %d captions)" % (result["phase0_summary"]["chunks"], result["phase0_summary"]["captions"]))
print("Phase 1: %d classifications" % len(result["phase1_classifications"]))
for p, c in result["phase1_classifications"].items():
    print("  %s -> %s (rel=%s)" % (os.path.basename(p)[:16], c.get("primary_type", "?"), c.get("relevance_to_synthesis", "?")))
print("Phase 2: %d contexts" % len(result["phase2_contexts"]))
print("Phase 3: %d extractions" % len(result["phase3_extractions"]))
for name, ext in result["phase3_extractions"].items():
    keys = [k for k in ext if not k.startswith("_")][:5]
    print("  %s -> %s" % (name, keys))
print("Phase 4: %d verifications" % len(result["phase4_verifications"]))
for name, ver in result["phase4_verifications"].items():
    print("  %s -> assessment=%s conf=%s" % (name, ver.get("overall_assessment", "?"), ver.get("overall_confidence", "?")))
print("Phase 5: %s" % ("completed" if isinstance(result["phase5_paper_summary"], dict) else "NOT completed"))
