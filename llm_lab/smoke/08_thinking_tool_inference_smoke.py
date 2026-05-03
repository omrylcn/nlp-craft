"""Thinking + tool calling inference patterns smoke test.

Coverage:
1. Thinking model (Qwen3-4B-Thinking-2507) inference
   - Auto <think> generation
   - Parse <think>...</think> + final answer separately
2. Tool calling inference (Qwen3-4B-Instruct-2507 with tools)
   - Render tools via chat template
   - Parse <tool_call>{...}</tool_call> from output
   - Multi-turn: assistant → tool → assistant final answer
"""
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import re
import json

print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# ============================================================
# PART 1 — THINKING MODEL INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("PART 1 — Thinking Model (Qwen3-4B-Thinking-2507)")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Thinking-2507",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")
FastLanguageModel.for_inference(model)

# 1.1 — Generate, capture full output (think + answer)
msgs = [{"role": "user", "content": "Hesapla 137 * 49"}]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(
    **inputs, max_new_tokens=400,
    temperature=0.6, top_p=0.95, top_k=20, do_sample=True,
)
full = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n--- Full output (first 500 chars) ---")
print(full[:500])

# 1.2 — Parse <think>...</think> separately
think_match = re.search(r"<think>\s*(.*?)\s*</think>", full, re.DOTALL)
think_part = think_match.group(1).strip() if think_match else ""
final_part = re.sub(r"<think>.*?</think>\s*", "", full, count=1, flags=re.DOTALL).strip()

print(f"\n--- Thinking ({len(think_part)} chars) ---")
print(think_part[:300] + ("..." if len(think_part) > 300 else ""))
print(f"\n--- Final answer ({len(final_part)} chars) ---")
print(final_part[:300] + ("..." if len(final_part) > 300 else ""))

# 1.3 — Hide thinking from user (production pattern)
print(f"\n--- For user (only final): ---")
print(final_part[:200])

# Cleanup before next model
import gc
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# PART 2 — TOOL CALLING INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("PART 2 — Tool Calling (Qwen3-4B-Instruct-2507)")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
FastLanguageModel.for_inference(model)

# 2.1 — Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

# 2.2 — First turn: model decides to call a tool
print("\n--- Turn 1: User asks, model decides ---")
conv = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user",   "content": "Istanbul hava nasil?"},
]
text = tokenizer.apply_chat_template(conv, tokenize=False, tools=tools, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(
    **inputs, max_new_tokens=120,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
)
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Model output:\n{response}")

# 2.3 — Parse <tool_call>{...}</tool_call> (Qwen3 format)
tool_call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
if tool_call_match:
    call_json = json.loads(tool_call_match.group(1))
    tool_name = call_json["name"]
    tool_args = call_json["arguments"]
    print(f"\nParsed tool call: {tool_name}({tool_args})")
else:
    print("\n(No tool call detected — model answered directly)")
    tool_name = None

# 2.4 — Execute tool (mock) + feed result back
if tool_name == "get_weather":
    fake_result = {"city": tool_args["city"], "temp_c": 18, "condition": "cloudy"}
    print(f"Executed → {fake_result}")

    # Multi-turn: append assistant tool_calls + tool result + ask again
    print("\n--- Turn 2: Feed tool result, get final answer ---")
    conv_full = conv + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": tool_name, "arguments": tool_args}}]},
        {"role": "tool", "tool_call_id": "c1", "name": tool_name,
         "content": json.dumps(fake_result)},
    ]
    text2 = tokenizer.apply_chat_template(
        conv_full, tokenize=False, tools=tools, add_generation_prompt=True,
    )
    inputs2 = tokenizer(text2, return_tensors="pt").to("cuda")
    out2 = model.generate(
        **inputs2, max_new_tokens=80,
        temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    )
    final = tokenizer.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nFinal answer:\n{final}")

print("\n=== SMOKE DONE ===")
