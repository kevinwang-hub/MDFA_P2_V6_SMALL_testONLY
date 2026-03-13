"""Quick empirical test: does Ollama truncate prompts when num_ctx is not set?"""
import openai
import time

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="EMPTY", timeout=120)

# Test 1: Small prompt (should work fine)
print("=" * 60)
print("TEST 1: Small prompt (~100 tokens)")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="qwen3.5:9b",
        messages=[
            {"role": "system", "content": "/no_think\nYou output ONLY valid JSON."},
            {"role": "user", "content": 'Return this JSON: {"status": "ok", "test": 1}'},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    elapsed = time.time() - t0
    content = r.choices[0].message.content or ""
    reasoning = getattr(r.choices[0].message, "reasoning", None) or ""
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Content length: {len(content)}")
    print(f"  Content: {content[:200]}")
    print(f"  Reasoning length: {len(reasoning)}")
    if reasoning:
        print(f"  Reasoning (first 200): {reasoning[:200]}")
    print(f"  Finish reason: {r.choices[0].finish_reason}")
    print(f"  Usage: {r.usage}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Same but with num_ctx and think=false
print()
print("=" * 60)
print("TEST 2: Small prompt + num_ctx=4096 + think=false")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="qwen3.5:9b",
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON."},
            {"role": "user", "content": 'Return this JSON: {"status": "ok", "test": 2}'},
        ],
        temperature=0.1,
        max_tokens=256,
        extra_body={
            "options": {"num_ctx": 4096},
            "think": False,
        },
    )
    elapsed = time.time() - t0
    content = r.choices[0].message.content or ""
    reasoning = getattr(r.choices[0].message, "reasoning", None) or ""
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Content length: {len(content)}")
    print(f"  Content: {content[:200]}")
    print(f"  Reasoning length: {len(reasoning)}")
    print(f"  Finish reason: {r.choices[0].finish_reason}")
    print(f"  Usage: {r.usage}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Large prompt WITHOUT num_ctx — does it get truncated?
big_data = "x " * 3000  # ~3000 tokens
print()
print("=" * 60)
print(f"TEST 3: Large prompt (~3000 tokens) WITHOUT num_ctx")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="qwen3.5:9b",
        messages=[
            {"role": "system", "content": "/no_think\nYou output ONLY valid JSON."},
            {"role": "user", "content": f'Ignore this padding: {big_data}\n\nNow return: {{"status": "ok", "test": 3, "saw_full_prompt": true}}'},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    elapsed = time.time() - t0
    content = r.choices[0].message.content or ""
    reasoning = getattr(r.choices[0].message, "reasoning", None) or ""
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Content length: {len(content)}")
    print(f"  Content (first 300): {content[:300]}")
    print(f"  Reasoning length: {len(reasoning)}")
    print(f"  Finish reason: {r.choices[0].finish_reason}")
    print(f"  Usage: {r.usage}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 4: Large prompt WITH num_ctx
print()
print("=" * 60)
print(f"TEST 4: Large prompt (~3000 tokens) WITH num_ctx=8192 + think=false")
t0 = time.time()
try:
    r = client.chat.completions.create(
        model="qwen3.5:9b",
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON."},
            {"role": "user", "content": f'Ignore this padding: {big_data}\n\nNow return: {{"status": "ok", "test": 4, "saw_full_prompt": true}}'},
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
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Content length: {len(content)}")
    print(f"  Content (first 300): {content[:300]}")
    print(f"  Reasoning length: {len(reasoning)}")
    print(f"  Finish reason: {r.choices[0].finish_reason}")
    print(f"  Usage: {r.usage}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("Compare Test 3 vs Test 4 — if Test 3 fails or returns garbage")
print("but Test 4 works, then num_ctx is the root cause.")
