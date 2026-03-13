"""Quick summary of the single-image pipeline result."""
import json, os

path = os.path.expanduser("~/Downloads/test2_v6_1img_64bb_results.json")
with open(path) as f:
    d = json.load(f)

print("=== CLASSIFICATION ===")
for p, c in d.get("classifications", {}).items():
    print(f"  Type: {c.get('primary_type')}")
    print(f"  Relevance: {c.get('relevance_to_synthesis')}")
    desc = c.get("detailed_description", "?")
    print(f"  Description: {desc[:250]}")

print("\n=== EXTRACTION ===")
for ext in d.get("image_extractions", []):
    keys = [k for k in ext if not k.startswith("_")]
    for k in keys[:15]:
        v = ext[k]
        if isinstance(v, (dict, list)):
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
        else:
            print(f"  {k}: {v}")

print("\n=== PAPER SUMMARY ===")
ps = d.get("paper_summary", {})
has_data = bool(ps and not ps.get("_parse_error"))
print(f"  Completed: {has_data}")
if has_data:
    mats = ps.get("materials_reported", [])
    print(f"  Materials: {len(mats)}")
    for m in mats[:5]:
        print(f"    - {m.get('name','?')} ({m.get('material_class','?')})")
else:
    raw = ps.get("_raw_response", "")[:400]
    print(f"  Raw: {raw}")

print(f"\nFile size: {os.path.getsize(path)} bytes")
print(f"Location: {path}")
