"""DPO smoke test — Gemma 4 E2B with FastModel + UltraFeedback dataset.

CRITICAL ORDER:
1. PatchDPOTrainer() BEFORE `from trl import DPOTrainer`
2. ref_model=None → PEFT auto-toggles LoRA for reference logprobs
3. processing_class= (NOT tokenizer=)
4. beta/max_length in DPOConfig (NOT DPOTrainer kwargs)
"""
import unsloth
from unsloth import PatchDPOTrainer
PatchDPOTrainer()                              # MUST run before TRL import

import torch
from datasets import load_dataset
from unsloth import FastModel
from trl import DPOTrainer, DPOConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — Gemma 4 E2B (FastModel — multimodal-native)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=False,                          # 16-bit (Gemma 4 4-bit destek farklı)
    full_finetuning=False,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,                # Text-only DPO
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, lora_alpha=8, lora_dropout=0,           # Gemma 4 küçük, r=8 yeter
    bias="none",
    random_state=3407,
)

# Padding side
if hasattr(tokenizer, "tokenizer"):
    tokenizer.tokenizer.padding_side = "left"
else:
    tokenizer.padding_side = "left"

# 2. Dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:200]")
print(f"\nDataset rows: {len(dataset)}")
print(f"Cols: {dataset.column_names}")

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
    max_steps=3,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    bf16=True,
    seed=3407,
    output_dir="outputs_dpo_gemma4",
    report_to="none",
)

# 4. Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
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
