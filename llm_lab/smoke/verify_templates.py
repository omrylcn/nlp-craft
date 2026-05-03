"""Chat template + data prep verification for the planned 00_1 notebook.
Captures real outputs from the installed unsloth/transformers stack so the notebook
text reflects ground truth rather than guesses."""
import unsloth
from unsloth.chat_templates import get_chat_template, standardize_data_formats, CHAT_TEMPLATES
from transformers import AutoTokenizer
from datasets import Dataset
import json

print("=" * 70)
print("PART 0 — Available templates in CHAT_TEMPLATES dict")
print("=" * 70)
print(sorted(CHAT_TEMPLATES.keys()))
print()

# The 5 modern templates in scope
SCOPE = ["qwen3-instruct", "qwen3-thinking", "gemma-4", "gemma-4-thinking", "llama-3.1"]

# Modeller — tokenizer'a ait template'le ve get_chat_template(name) ile aynı sonuç var mı?
TOK_FOR = {
    "qwen3-instruct": "unsloth/Qwen3-4B-Instruct-2507",
    "qwen3-thinking": "unsloth/Qwen3-4B-Thinking-2507",
    "gemma-4":        "unsloth/gemma-4-E2B-it",
    "gemma-4-thinking": "unsloth/gemma-4-E2B-it",   # aynı tokenizer, farklı template
    "llama-3.1":      "unsloth/Llama-3.1-8B-Instruct",
}

msgs = [
    {"role": "user", "content": "Faiz nedir?"},
    {"role": "assistant", "content": "Faiz, paranin zaman degeridir."},
]

print("=" * 70)
print("PART 1 — Render comparison: same conversation, 5 templates")
print("=" * 70)
rendered = {}
for name in SCOPE:
    print(f"\n----- {name} -----")
    try:
        tok = AutoTokenizer.from_pretrained(TOK_FOR[name])
        tok = get_chat_template(tok, chat_template=name)
        out = tok.apply_chat_template(msgs, tokenize=False)
        rendered[name] = out
        print(out)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("PART 2 — add_generation_prompt=True: what's appended at the end?")
print("=" * 70)
for name in SCOPE:
    try:
        tok = AutoTokenizer.from_pretrained(TOK_FOR[name])
        tok = get_chat_template(tok, chat_template=name)
        no_gen = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=False)
        with_gen = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        suffix = with_gen[len(no_gen):]
        print(f"{name}: appended = {suffix!r}")
    except Exception as e:
        print(f"{name}: FAIL {e}")

print("\n" + "=" * 70)
print("PART 3 — tools= parameter support per template")
print("=" * 70)
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
test_msgs = [{"role": "user", "content": "Istanbul'da hava nasil?"}]
for name in SCOPE:
    try:
        tok = AutoTokenizer.from_pretrained(TOK_FOR[name])
        tok = get_chat_template(tok, chat_template=name)
        without = tok.apply_chat_template(test_msgs, tokenize=False)
        with_t  = tok.apply_chat_template(test_msgs, tokenize=False, tools=tools)
        supported = with_t != without
        print(f"\n----- {name}: tools supported = {supported} -----")
        if supported:
            print(f"  Diff length: {len(with_t) - len(without)} chars added")
            print(f"  Tools rendered as: {with_t[:300]}...")
        else:
            print(f"  tools= ignored (no diff). Output length stays {len(without)}")
    except Exception as e:
        print(f"{name}: FAIL {e}")

print("\n" + "=" * 70)
print("PART 4 — enable_thinking= parameter support")
print("=" * 70)
for name in ["qwen3-instruct", "qwen3-thinking", "gemma-4", "gemma-4-thinking"]:
    try:
        tok = AutoTokenizer.from_pretrained(TOK_FOR[name])
        tok = get_chat_template(tok, chat_template=name)
        for et in [False, True]:
            try:
                out = tok.apply_chat_template([msgs[0]], tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=et)
                tail = out[-60:]
                print(f"  {name} enable_thinking={et}: ...{tail!r}")
            except Exception as e:
                print(f"  {name} enable_thinking={et}: FAIL {type(e).__name__}: {e}")
    except Exception as e:
        print(f"{name}: load FAIL {e}")

print("\n" + "=" * 70)
print("PART 5 — standardize_data_formats: input → output")
print("=" * 70)

# 5a — Alpaca format
print("\n--- 5a. Alpaca input ---")
alpaca = Dataset.from_list([
    {"instruction": "Define faiz", "input": "", "output": "Faiz parayi zamanin degeridir."},
    {"instruction": "Translate", "input": "hello", "output": "merhaba"},
])
print(f"Cols before: {alpaca.column_names}")
print(f"Sample 0:    {alpaca[0]}")
try:
    out = standardize_data_formats(alpaca)
    print(f"Cols after:  {out.column_names}")
    print(f"Sample 0:    {out[0]}")
except Exception as e:
    print(f"FAIL: {e}")

# 5b — ShareGPT format
print("\n--- 5b. ShareGPT input ---")
sharegpt = Dataset.from_list([
    {"conversations": [
        {"from": "human", "value": "Define faiz"},
        {"from": "gpt", "value": "Faiz parayi zamanin degeridir."},
    ]},
])
print(f"Cols before: {sharegpt.column_names}")
print(f"Sample 0:    {sharegpt[0]}")
try:
    out = standardize_data_formats(sharegpt)
    print(f"Cols after:  {out.column_names}")
    print(f"Sample 0:    {out[0]}")
except Exception as e:
    print(f"FAIL: {e}")

# 5c — OpenAI messages
print("\n--- 5c. OpenAI messages input ---")
openai = Dataset.from_list([
    {"messages": [
        {"role": "user", "content": "Define faiz"},
        {"role": "assistant", "content": "Faiz parayi zamanin degeridir."},
    ]},
])
print(f"Cols before: {openai.column_names}")
print(f"Sample 0:    {openai[0]}")
try:
    out = standardize_data_formats(openai)
    print(f"Cols after:  {out.column_names}")
    print(f"Sample 0:    {out[0]}")
except Exception as e:
    print(f"FAIL: {e}")

print("\n" + "=" * 70)
print("PART 6 — Multi-turn + system prompt rendering")
print("=" * 70)
multi = [
    {"role": "system", "content": "You are a helpful Turkish assistant."},
    {"role": "user", "content": "Faiz nedir?"},
    {"role": "assistant", "content": "Faiz, paranin zaman degeridir."},
    {"role": "user", "content": "Bir ornek ver."},
]
for name in ["qwen3-instruct", "gemma-4"]:
    print(f"\n----- {name} -----")
    try:
        tok = AutoTokenizer.from_pretrained(TOK_FOR[name])
        tok = get_chat_template(tok, chat_template=name)
        print(tok.apply_chat_template(multi, tokenize=False, add_generation_prompt=True))
    except Exception as e:
        print(f"FAIL: {e}")

print("\n=== VERIFY DONE ===")
