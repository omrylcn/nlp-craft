"""Quick check: which Gemma 4 variants have thinking support?"""
import unsloth
from unsloth.chat_templates import CHAT_TEMPLATES
from transformers import AutoTokenizer

print("Unsloth registered templates with 'thinking' in name:")
for k in sorted(CHAT_TEMPLATES.keys()):
    if "thinking" in k:
        print(f"  - {k}")

print("\nGemma 4 model variants (check Hub):")
candidates = [
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E2B-thinking-it",
    "unsloth/gemma-4-E2B-thinking",
    "unsloth/gemma-4-E4B-thinking-it",
    "unsloth/gemma-4-E4B-thinking",
    "unsloth/gemma-4-31B-thinking-it",
]
for c in candidates:
    try:
        tok = AutoTokenizer.from_pretrained(c)
        ct = tok.chat_template or ""
        has_think = "<think>" in ct or "thought" in ct.lower() or "channel" in ct
        print(f"  EXISTS: {c}  template_len={len(ct)} thinking_markers={has_think}")
    except Exception as e:
        msg = str(e).split("\n")[0][:60]
        print(f"  NOT found: {c} ({msg})")

# Native gemma-4-E2B-it analizi
print("\nNative gemma-4-E2B-it template details:")
tok = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B-it")
ct = tok.chat_template
print(f"  total length: {len(ct)} chars")
for marker in ["<think>", "</think>", "thought", "channel", "enable_thinking", "<|think|>"]:
    print(f"  '{marker}' present: {marker in ct}")
