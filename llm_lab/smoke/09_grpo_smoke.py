"""GRPO smoke test — Qwen3-1.7B-Base + math reasoning + vLLM.

Goal: prove GRPO pipeline works on RTX 4070 Ti SUPER 16GB:
- Pre-finetune SFT not needed for smoke (just verify GRPO loop)
- Custom markers <start_working_out>...<SOLUTION>
- 4 reward functions
- vLLM fast generation with shared memory (UNSLOTH_VLLM_STANDBY=1)
- 3 step training to verify pipeline
"""
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"           # vLLM memory release during optimizer step

import unsloth
from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# ============================================================
# 1. Markers + system prompt (Qwen3-4B-GRPO official pattern)
# ============================================================
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# ============================================================
# 2. Model — Qwen3-1.7B-Base for 16GB safety
# ============================================================
max_seq_length = 1024
max_prompt_length = 256
max_completion_length = max_seq_length - max_prompt_length

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-1.7B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,                          # GRPO bf16 LoRA (4bit problematic)
    fast_inference=True,                         # vLLM enabled
    gpu_memory_utilization=0.6,                  # leave room for training
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16,                          # smaller r for safety
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Custom chat template that auto-prepends <start_working_out> at gen prompt
# (so model never has to emit opening tag itself)
chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for m in loop_messages %}"
    "{% if m['role'] == 'user' %}{{ m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] + eos_token }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
)
tokenizer.chat_template = chat_template

# ============================================================
# 3. Dataset — OpenMathReasoning-mini (small for smoke)
# ============================================================
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
print(f"Total rows: {len(dataset)}")

def extract_answer(text):
    # Solution field contains R1 trace; extract numeric/short answer at end
    return text.strip().split("\n")[-1] if text else ""

def make_grpo_row(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": example["problem"]},
        ],
        "answer": str(example["expected_answer"]),
    }

# Take small subset, filter by length, transform
import pandas as pd
df = dataset.to_pandas()
df = df[pd.to_numeric(df["expected_answer"], errors="coerce").notnull()]
df = df.head(50)                                  # small for smoke
dataset = dataset.from_pandas(df)
dataset = dataset.map(
    make_grpo_row,
    remove_columns=[c for c in dataset.column_names if c not in ["prompt", "answer"]],
)
print(f"After filter+transform: {len(dataset)} rows")
print(f"Sample[0] prompt: {dataset[0]['prompt']}")
print(f"Sample[0] answer: {dataset[0]['answer']}")

# ============================================================
# 4. Reward functions — 4 of them
# ============================================================
match_format = re.compile(
    rf"{reasoning_end}.*?{solution_start}(.+?){solution_end}\s*$",
    re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(rf"{solution_start}.*?(-?[\d\.\,]+)", re.DOTALL)

def match_format_exactly(completions, **kwargs):
    scores = []
    for c in completions:
        text = c[0]["content"]
        score = 3.0 if match_format.search(text) is not None else 0.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for c in completions:
        text = c[0]["content"]
        s = 0.0
        s += 0.5 if text.count(reasoning_end)  == 1 else -1.0
        s += 0.5 if text.count(solution_start) == 1 else -1.0
        s += 0.5 if text.count(solution_end)   == 1 else -1.0
        scores.append(s)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [g.group(1) if (g := match_format.search(r)) else None for r in responses]
    scores = []
    for guess, true in zip(extracted, answer):
        s = 0.0
        if guess is None: scores.append(-2.0); continue
        if guess.strip() == true.strip(): s += 5.0
        else:
            try:
                ratio = float(guess) / float(true)
                if 0.9 <= ratio <= 1.1: s += 2.0
                elif 0.8 <= ratio <= 1.2: s += 1.5
                else: s -= 2.5
            except: s -= 4.5
        scores.append(s)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [g.group(1) if (g := match_numbers.search(r)) else None for r in responses]
    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None: scores.append(-2.5); continue
        try:
            true_v = float(str(true).strip())
            guess_v = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess_v == true_v else -1.5)
        except: scores.append(0.0)
    return scores

# ============================================================
# 5. GRPOConfig + Trainer
# ============================================================
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p=0.1, top_p=1.0, top_k=-1, seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,                            # 4 group size
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=3,                                  # SMOKE
    save_steps=999,
    report_to="none",
    output_dir="outputs_grpo",
)

print("\n=== TRAINING START ===")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)

start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"Start mem: {start_mem} GB")

trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n=== RESULTS ===")
print(f"Peak VRAM: {used} GB")
print("=== SMOKE DONE ===")
