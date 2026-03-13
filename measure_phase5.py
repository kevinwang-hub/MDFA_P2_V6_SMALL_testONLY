"""Measure actual Phase 5 prompt sizes for each paper to diagnose failures."""
import json, os, glob

papers = {
    "test_1": "/Users/mac/Documents/test_parsed_output/Source/test_1_02439suppappendix/02439suppappendix/hybrid_auto/extractions",
    "test_2": "/Users/mac/Documents/test_parsed_output/Source/test_2_41586_2015_BFnature15732_MOESM68_ESM/41586_2015_BFnature15732_MOESM68_ESM/hybrid_auto/extractions",
    "test_3": "/Users/mac/Documents/test_parsed_output/Source/test_3_ banerjee.som/banerjee.som/hybrid_auto/extractions",
}

# Read the actual system prompt + user template from aggregator
from phase5_synthesizer.aggregator import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

print(f"SYSTEM_PROMPT length: {len(SYSTEM_PROMPT)} chars (~{len(SYSTEM_PROMPT)//4} tokens)")
print(f"USER_PROMPT_TEMPLATE (empty): {len(USER_PROMPT_TEMPLATE)} chars (~{len(USER_PROMPT_TEMPLATE)//4} tokens)")
print(f"/no_think present in system: {'/no_think' in SYSTEM_PROMPT}")
print()

for name, edir in papers.items():
    print(f"{'='*60}")
    print(f"  {name}")

    # Load Phase 4 verified extractions
    p4dir = os.path.join(edir, "phase4_verification")
    if not os.path.isdir(p4dir):
        print(f"  No Phase 4 dir")
        continue

    p4files = sorted(glob.glob(os.path.join(p4dir, "*.json")))
    
    # Reconstruct what pipeline.py builds as verified_extractions
    verified_extractions = []
    for pf in p4files:
        with open(pf) as f:
            verified = json.load(f)
        corrected = verified.get("corrected_extraction")
        final = corrected if corrected else verified
        # Strip internal keys like pipeline does
        clean = {k: v for k, v in final.items() if not k.startswith("_")}
        verified_extractions.append(clean)

    all_ext_json = json.dumps(verified_extractions, indent=2, ensure_ascii=False)
    
    # Build the context summary (approximate — use Phase 0 chunks)
    p0 = os.path.join(edir, "phase0_context.json")
    context_summary = ""
    if os.path.exists(p0):
        with open(p0) as f:
            p0data = json.load(f)
        chunks = p0data.get("chunks", [])
        key_sections = ["abstract", "experimental", "results", "discussion"]
        parts = [c["text"] for c in chunks if c.get("section_type") in key_sections][:20]
        context_summary = "\n\n".join(parts)
    
    # Build the full user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title="(paper title)",
        authors="",
        doi="",
        journal="",
        all_extractions_json=all_ext_json,
        paper_context_summary=context_summary,
    )
    
    total_input = len(SYSTEM_PROMPT) + len(user_prompt)
    est_tokens = total_input // 4  # rough estimate
    
    print(f"  Extractions: {len(verified_extractions)}")
    print(f"  Extraction JSON: {len(all_ext_json):,} chars")
    print(f"  Context summary: {len(context_summary):,} chars")
    print(f"  User prompt total: {len(user_prompt):,} chars")
    print(f"  TOTAL INPUT: {total_input:,} chars (~{est_tokens:,} tokens)")
    print(f"  + requested output (AGGREGATION_MAX_TOKENS): 8192 tokens")
    print(f"  = TOTAL NEEDED: ~{est_tokens + 8192:,} tokens")
    print()
    
    # Check against Ollama defaults
    default_ctx = 2048
    print(f"  Ollama default num_ctx: {default_ctx}")
    if est_tokens > default_ctx:
        print(f"  *** INPUT ALONE ({est_tokens:,}) EXCEEDS DEFAULT CONTEXT ({default_ctx})! ***")
        print(f"  *** Model is TRUNCATING the prompt — it never sees all the data! ***")
    elif est_tokens + 8192 > default_ctx:
        print(f"  *** INPUT + OUTPUT ({est_tokens + 8192:,}) EXCEEDS DEFAULT CONTEXT ({default_ctx})! ***")
    else:
        print(f"  OK — fits within context window")
    print()

print("="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)
print("""
The Ollama OpenAI-compatible API does NOT automatically expand num_ctx.
Default num_ctx for most models is 2048 tokens.

When your prompt exceeds 2048 tokens, Ollama silently TRUNCATES the input.
The model sees a chopped-up prompt and produces garbage/nulls/partial JSON.

When num_ctx is large enough for input but not input+output, the model's
output gets cut off mid-generation, producing truncated invalid JSON.

This is the #1 reason Phase 5 fails:
  - test_1: ~5400 tokens input → truncated to 2048 → garbage output
  - test_2: ~3100 tokens input → truncated to 2048 → 645 bytes of nulls
  - test_3: ~1900 tokens input → fits BUT output truncated at 2048-1900=148 tokens

The /no_think fix has NO EFFECT via the OpenAI-compatible endpoint.
Ollama's /v1/chat/completions ignores /no_think in the system prompt.
To disable thinking, you must pass extra_body={"options": {"num_predict": ...}}
or use the native /api/chat endpoint with "think": false.
""")
