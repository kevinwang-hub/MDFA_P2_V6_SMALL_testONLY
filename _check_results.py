import json

with open("/Users/mac/Downloads/test2_v6_1img_64bb_results.json") as f:
    r = json.load(f)

# Phase 1
cls = list(r["classifications"].values())[0]
print("=== PHASE 1 ===")
print(f"  type: {cls.get('primary_type')}, relevance: {cls.get('relevance_to_synthesis')}")

# Phase 3
ext = r["image_extractions"][0]
print("\n=== PHASE 3 ===")
print(f"  _free_text: {len(ext.get('_free_text',''))} chars")
print(f"  stage1_chars: {ext.get('_metadata',{}).get('stage1_chars')}")
data_keys = [k for k in ext if not k.startswith("_")]
print(f"  data keys: {data_keys}")

# Phase 5
ps = r["paper_summary"]
print("\n=== PHASE 5 ===")
print(f"  fallback: {ps.get('_fallback', False)}")
mats = ps.get("materials_reported", [])
print(f"  materials: {len(mats)}")
for m in mats:
    print(f"    - {m.get('name','?')} ({m.get('material_class','?')})")
print(f"  gaps: {len(ps.get('data_gaps',[]))}")
print(f"  conflicts: {len(ps.get('cross_figure_conflicts',[]))}")
print(f"\nFull Phase 5 JSON ({len(json.dumps(ps))} bytes):")
print(json.dumps(ps, indent=2))
