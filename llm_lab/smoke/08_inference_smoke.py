"""Inference patterns smoke test — Qwen3-4B-Instruct.

Coverage:
1. Base model inference (no training)
2. Generation parameters (T, top_p, top_k, min_p, rep_penalty)
3. Quick 3-step SFT → save LoRA
4. Reload LoRA from disk + inference
5. Batched inference (multiple prompts)
6. TextIteratorStreamer (production pattern)
7. FastLanguageModel.for_inference() optimization
8. fast_inference=True (vLLM) attempt
"""
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch
from transformers import TextStreamer, TextIteratorStreamer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from threading import Thread

print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

MODEL = "unsloth/Qwen3-4B-Instruct-2507"

# ============================================================
# 1. Load base model + chat template
# ============================================================
print("\n=== 1. LOAD BASE MODEL ===")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
FastLanguageModel.for_inference(model)              # 2x faster inference

# ============================================================
# 2. Basic inference with chat template
# ============================================================
print("\n=== 2. BASIC INFERENCE ===")
def gen_text(messages, **gen_kwargs):
    """Helper — apply chat template, tokenize, generate."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, **gen_kwargs)
    # Decode only the new tokens (skip prompt)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

msgs = [{"role": "user", "content": "Faiz nedir? Kisaca aciklayin."}]

print("\n-- Default (greedy) --")
print(gen_text(msgs, max_new_tokens=80, do_sample=False))

# ============================================================
# 3. Generation parameter exploration
# ============================================================
print("\n=== 3. GENERATION PARAMETERS ===")

print("\n-- Qwen3 instruct önerisi (T=0.7, top_p=0.8, top_k=20) --")
print(gen_text(msgs, max_new_tokens=80,
               temperature=0.7, top_p=0.8, top_k=20, do_sample=True))

print("\n-- Yüksek T (T=1.5, daha yaratıcı) --")
print(gen_text(msgs, max_new_tokens=80,
               temperature=1.5, top_p=0.95, do_sample=True))

print("\n-- repetition_penalty=1.3 (tekrarı azalt) --")
print(gen_text(msgs, max_new_tokens=80,
               temperature=0.7, top_p=0.8, top_k=20,
               repetition_penalty=1.3, do_sample=True))

# ============================================================
# 4. TextStreamer (basic) vs TextIteratorStreamer (production)
# ============================================================
print("\n=== 4. STREAMING ===")
print("\n-- TextStreamer (basit, stdout'a yazar) --")
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
_ = model.generate(
    **inputs, max_new_tokens=60,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

print("\n\n-- TextIteratorStreamer (async-friendly, production) --")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(**inputs, max_new_tokens=60,
                  temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
                  streamer=streamer)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
collected = ""
for chunk in streamer:
    collected += chunk
    print(chunk, end="", flush=True)
thread.join()
print(f"\n[Total chunks collected: {len(collected)} chars]")

# ============================================================
# 5. Batched inference
# ============================================================
print("\n\n=== 5. BATCHED INFERENCE ===")
prompts = [
    "Istanbul nerededir?",
    "Kar nasil olusur?",
    "Quantum nedir?",
]
formatted = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True,
    ) for p in prompts
]
# Pad on left for batched generation
tokenizer.padding_side = "left"
batch_inputs = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda")
out = model.generate(
    **batch_inputs, max_new_tokens=50,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)
print(f"Batch shape: {out.shape}")
for i, prompt in enumerate(prompts):
    new_tokens = out[i][batch_inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\nQ: {prompt}")
    print(f"A: {response[:120]}...")

tokenizer.padding_side = "right"                    # restore

# ============================================================
# 6. Quick training to produce a LoRA
# ============================================================
print("\n\n=== 6. QUICK SFT (3 steps) ===")
FastLanguageModel.for_training(model)               # back to training mode
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
mini_data = Dataset.from_list([
    {"conversations": [
        {"role": "user", "content": f"Söyle: {x}"},
        {"role": "assistant", "content": f"BIRINCI YANIT: {x}!"},
    ]}
    for x in ["evet", "hayır", "tamam"] * 10
])
mini_data = mini_data.map(lambda e: {"text": [
    tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
    for c in e["conversations"]
]}, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=mini_data,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        warmup_steps=2, max_steps=3, learning_rate=2e-4,
        logging_steps=1, optim="adamw_8bit",
        weight_decay=0.001, lr_scheduler_type="linear",
        seed=3407, report_to="none",
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
trainer.train()

# Save LoRA
print("\n=== 7. SAVE LoRA ===")
model.save_pretrained("test_lora")
tokenizer.save_pretrained("test_lora")
print("Saved to test_lora/")

# ============================================================
# 7. Post-training inference (model has LoRA active)
# ============================================================
print("\n=== 8. POST-TRAIN INFERENCE (LoRA active in current model) ===")
FastLanguageModel.for_inference(model)
test_msg = [{"role": "user", "content": "Söyle: evet"}]
print(gen_text(test_msg, max_new_tokens=30,
               temperature=0.7, top_p=0.8, top_k=20, do_sample=True))

# ============================================================
# 8. Reload LoRA from disk (production scenario)
# ============================================================
print("\n=== 9. RELOAD LoRA FROM DISK ===")
print("(Yeni session simülasyonu — diskten LoRA + base yükle)")

# In production you'd start a fresh process. Here we just load the saved adapter on top.
# Standard pattern: from_pretrained loads the saved adapter automatically if you point to the LoRA dir
# But here we're already in same session, so adapter is active. To DEMO reload:
import gc
del model, trainer
gc.collect()
torch.cuda.empty_cache()

model_reloaded, tok_reloaded = FastLanguageModel.from_pretrained(
    "test_lora",                                    # YEAH — point to saved LoRA dir
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_reloaded)
text = tok_reloaded.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True)
inputs = tok_reloaded(text, return_tensors="pt").to("cuda")
out = model_reloaded.generate(
    **inputs, max_new_tokens=30,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
)
new_tokens = out[0][inputs["input_ids"].shape[1]:]
print(f"Reloaded LoRA response: {tok_reloaded.decode(new_tokens, skip_special_tokens=True)}")

print("\n=== SMOKE DONE ===")
