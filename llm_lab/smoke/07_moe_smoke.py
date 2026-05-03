"""MoE SFT smoke test — TinyQwen3 MoE 2.8B/0.7B (T4-compatible).

Goal: prove MoE SFT pipeline works on RTX 4070 Ti SUPER 16GB with current
stack (trl 0.24.0 — note: official MoE notebooks pin trl==0.22.2!).

Key MoE-specific differences:
- `load_in_4bit=False` — bnb-4bit not supported for MoE expert params
- `fast_inference=False` — vLLM not supported for MoE yet
- target_modules includes `gate_up_proj` (routed-expert MLPs)
- `lora_alpha = r * 2` (2x speeds up training, official recommendation)
- UNSLOTH_MOE_DISABLE_AUTOTUNE=1 — saves memory on consumer GPUs
"""
import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'   # before unsloth import

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — TinyQwen3 MoE (Unsloth's T4-compatible reference)
MODEL = "imdatta0/tiny_qwen3_moe_2.8B_0.7B"
max_seq_length = 2048
lora_rank = 32

print(f"\nLoading {MODEL}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length = max_seq_length,
    load_in_4bit = False,                    # MoE: bnb-4bit DESTEKLENMIYOR
    fast_inference = False,                  # vLLM MoE'de YOK
)
print(f"Model loaded — {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# 2. LoRA — MoE için target_modules'a gate_up_proj eklendi
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "gate_up_proj",                        # MoE expert layers
    ],
    lora_alpha = lora_rank * 2,                # 2x — official notebook recipe
    use_gradient_checkpointing = True,
    random_state = 3407,
    bias = "none",
)

# 3. Chat template — TinyQwen3 might use Qwen3 chat format
try:
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    print("Chat template: qwen-3")
except Exception as e:
    print(f"get_chat_template failed: {e}")
    print("Using native template")
    print(f"Native template length: {len(tokenizer.chat_template) if tokenizer.chat_template else 0}")

# 4. Dataset — same as all official MoE notebooks
print("\nLoading dataset...")
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
print(f"Total rows: {len(dataset)}")

def to_messages(example):
    return {"conversations": [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["generated_solution"]},
    ]}

dataset = dataset.map(to_messages, remove_columns=dataset.column_names)
dataset = dataset.select(range(100))                # smoke için subset

def fmt(examples):
    return {"text": [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples["conversations"]
    ]}
dataset = dataset.map(fmt, batched=True)

# 5. Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 2,
        max_steps = 3,                         # SMOKE
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

# 6. Memory + Train
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start mem = {start_mem} GB")

print("\n=== TRAINING START ===")
trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n=== RESULTS ===")
print(f"Train runtime: {trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss:    {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM:     {used} GB")
print("\n=== SMOKE TEST DONE ===")
