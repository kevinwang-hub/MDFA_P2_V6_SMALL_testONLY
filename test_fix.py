"""Focused test: does think=false + num_ctx fix Phase 5?"""
import openai, time, json

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="EMPTY", timeout=120)

print("TEST: think=false + num_ctx=8192 (the fix)")
print("=" * 50)
t0 = time.time()
r = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[
        {"role": "system", "content": "You output ONLY valid JSON. No explanation."},
        {"role": "user", "content": 'Return: {"status": "ok", "model": "qwen3.5:9b", "thinking_disabled": true}'},
    ],
    temperature=0.1,
    max_tokens=256,
    extra_body={
        "options": {"num_ctx": 8192},
        "think": False,
    },
)
elapsed = time.time() - t0
content = r.choices[0].message.content or ""
reasoning = getattr(r.choices[0].message, "reasoning", None) or ""
print(f"Time: {elapsed:.1f}s")
print(f"Content: {content[:300]}")
print(f"Reasoning present: {bool(reasoning)}")
print(f"Finish: {r.choices[0].finish_reason}")
print(f"Usage: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")

# Verify JSON parseable
try:
    d = json.loads(content)
    print(f"Parsed OK: {d}")
except:
    print(f"JSON parse failed!")
