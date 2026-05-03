"""DPO smoke test — Qwen3-4B-Instruct-2507 (text-only, FastLanguageModel).

Replaces Qwen3.5 since Qwen3.5 is multimodal and Unsloth DPO doesn't support
text-only DPO with vision-capable models (vision processor branch fails).
"""
import unsloth
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — Qwen3-4B-Instruct (text-only)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=4096,
    load_in_4bit=True,                          # 4-bit QLoRA
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=64, lora_alpha=64, lora_dropout=0,        # DPO için r=64 önerilir
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Padding side — DPO requires LEFT
tokenizer.padding_side = "left"
print(f"Padding side: {tokenizer.padding_side}")

# 2. Dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:200]")
print(f"\nDataset rows: {len(dataset)}")
print(f"Cols: {dataset.column_names}")
print(f"\nSample[0] (first 200 chars per col):")
sample = dataset[0]
for k, v in sample.items():
    s = str(v)[:200]
    print(f"  {k}: {s}{'...' if len(str(v))>200 else ''}")

# 3. DPOConfig
training_args = DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    max_steps=3,                                 # SMOKE
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    bf16=True,
    seed=3407,
    output_dir="outputs_dpo_qwen3",
    report_to="none",
)

# 4. Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,                              # PEFT auto-handles
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 5. Memory + Train
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start mem = {start_mem} GB")

print("\n=== TRAINING START ===")
trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM: {used} GB")

for k in ['train_rewards/chosen', 'train_rewards/rejected', 'train_rewards/accuracies']:
    if k in trainer_stats.metrics:
        print(f"{k}: {trainer_stats.metrics[k]:.4f}")

print("\n=== SMOKE TEST DONE ===")
