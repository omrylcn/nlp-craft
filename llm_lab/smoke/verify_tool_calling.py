"""Tool calling verification for Qwen3.5 and Gemma 4.

Strategy: test BOTH Unsloth's registered template AND the upstream HF tokenizer's
native template. The HF native template (from tokenizer_config.json) might support
tools even when Unsloth's hard-coded version doesn't.

Critical questions:
1. Qwen3.5 — Unsloth has no 'qwen-3.5' template. Does upstream HF have tools?
2. Gemma 4 — Unsloth's 'gemma-4' template ignores tools. Does upstream HF differ?
"""
import unsloth
from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES
from transformers import AutoTokenizer

print("=" * 70)
print("Available Unsloth templates (filtered to qwen/gemma/llama):")
print("=" * 70)
for k in sorted(CHAT_TEMPLATES.keys()):
    if any(x in k for x in ["qwen", "gemma", "llama"]):
        print(f"  {k}")
print()

MODELS = {
    "Qwen3.5-2B (Vision)":      "unsloth/Qwen3.5-2B",
    "Qwen3-4B-Instruct-2507":   "unsloth/Qwen3-4B-Instruct-2507",
    "Gemma-4-E2B-it":           "unsloth/gemma-4-E2B-it",
    "Llama-3.1-8B-Instruct":    "unsloth/Llama-3.1-8B-Instruct",
}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

test_msgs = [{"role": "user", "content": "Istanbul hava nasil?"}]

# ============================================================
# Part 1 — Native HF chat templates (no get_chat_template)
# ============================================================
print("=" * 70)
print("PART 1 — Native HF chat templates (tokenizer's own template)")
print("=" * 70)

for label, path in MODELS.items():
    print(f"\n--- {label} ({path}) ---")
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        ct = tok.chat_template
        has_tools_branch = ct is not None and ("tools" in ct or "tool_calls" in ct)
        print(f"  Has chat_template: {ct is not None}")
        print(f"  Native template mentions 'tools'/'tool_calls': {has_tools_branch}")
        if not ct:
            print("  → No native template, skipping render test")
            continue

        without = tok.apply_chat_template(test_msgs, tokenize=False)
        try:
            with_t = tok.apply_chat_template(test_msgs, tokenize=False, tools=tools)
            supported = with_t != without
            diff = len(with_t) - len(without)
            print(f"  tools= renders: {supported} (diff: +{diff} chars)")
            if supported:
                # show first 250 chars of tool-augmented output
                print(f"  Sample: {with_t[:250]!r}")
        except Exception as e:
            print(f"  tools= raised: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  Tokenizer load FAIL: {type(e).__name__}: {e}")

# ============================================================
# Part 2 — Try assistant tool_call rendering (does it round-trip?)
# ============================================================
print("\n" + "=" * 70)
print("PART 2 — Multi-turn tool call rendering (assistant→tool→assistant)")
print("=" * 70)

multi_turn = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "Weather in Istanbul?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{
         "id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": {"city": "Istanbul"}},
     }]},
    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
     "content": '{"city": "Istanbul", "temp_c": 18}'},
    {"role": "assistant", "content": "It is 18°C in Istanbul."},
]

for label, path in MODELS.items():
    print(f"\n--- {label} ---")
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        if tok.chat_template is None:
            print("  No native template, skip")
            continue
        try:
            out = tok.apply_chat_template(multi_turn, tokenize=False, tools=tools)
            print(f"  Length: {len(out)} chars")
            print(f"  Output:\n{out}")
        except Exception as e:
            print(f"  RENDER FAIL: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  Tokenizer load FAIL: {e}")

# ============================================================
# Part 3 — Compare Unsloth's get_chat_template vs native
# ============================================================
print("\n" + "=" * 70)
print("PART 3 — Unsloth get_chat_template vs native (Qwen3.5 / Gemma 4)")
print("=" * 70)

# Qwen3.5 — no Unsloth template registered
print("\n--- Qwen3.5-2B native vs unsloth ---")
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-2B")
print(f"  Native template length: {len(tok.chat_template) if tok.chat_template else 0}")

for try_name in ["qwen-3", "qwen3", "qwen3-instruct", "qwen-2.5", "qwen2.5"]:
    try:
        tok2 = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-2B")
        tok2 = get_chat_template(tok2, chat_template=try_name)
        out = tok2.apply_chat_template(test_msgs, tokenize=False, tools=tools)
        print(f"  get_chat_template('{try_name}') + tools: works ({len(out)} chars)")
    except Exception as e:
        print(f"  get_chat_template('{try_name}'): FAIL {type(e).__name__}: {str(e)[:80]}")

# Gemma 4 — Unsloth template registered but no tools branch
print("\n--- Gemma-4-E2B native vs unsloth ---")
tok = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B-it")
print(f"  Native template length: {len(tok.chat_template) if tok.chat_template else 0}")

for try_name in ["gemma-4", "gemma-3", "gemma2"]:
    try:
        tok2 = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B-it")
        tok2 = get_chat_template(tok2, chat_template=try_name)
        out = tok2.apply_chat_template(test_msgs, tokenize=False, tools=tools)
        diff = len(out) - len(tok2.apply_chat_template(test_msgs, tokenize=False))
        print(f"  get_chat_template('{try_name}') + tools: diff +{diff} chars")
    except Exception as e:
        print(f"  get_chat_template('{try_name}'): FAIL {type(e).__name__}: {str(e)[:80]}")

print("\n=== VERIFY DONE ===")
