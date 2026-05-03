"""DPO smoke test — Qwen3.5-2B with FastVisionModel + UltraFeedback dataset.

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
from unsloth import FastVisionModel
from trl import DPOTrainer, DPOConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — Qwen3.5-2B (multimodal, FastVisionModel)
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-2B",
    load_in_4bit=False,                         # 16-bit LoRA
    use_gradient_checkpointing="unsloth",
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,               # Text-only DPO
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32, lora_alpha=32, lora_dropout=0,        # DPO için r=64 önerilir; smoke için 32 yeter
    bias="none",
    random_state=3407,
)

# Padding side check — DPO requires LEFT
if hasattr(tokenizer, "tokenizer"):
    tokenizer.tokenizer.padding_side = "left"
else:
    tokenizer.padding_side = "left"
print(f"Padding side: {getattr(tokenizer, 'padding_side', tokenizer.tokenizer.padding_side if hasattr(tokenizer, 'tokenizer') else 'unknown')}")

# 2. Dataset — ultrafeedback_binarized (already conversational format)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:200]")
print(f"\nDataset rows: {len(dataset)}")
print(f"Cols: {dataset.column_names}")
# Qwen3.5 multimodal — vision branch needs empty 'images' column
dataset = dataset.map(lambda x: {**x, "images": []})
print(f"After adding empty images col: {dataset.column_names}")
print(f"\nSample[0]:")
sample = dataset[0]
for k, v in sample.items():
    s = str(v)[:200]
    print(f"  {k}: {s}{'...' if len(str(v))>200 else ''}")

# 3. DPOConfig
training_args = DPOConfig(
    beta=0.1,                                    # KL penalty
    loss_type="sigmoid",                         # default
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,                          # DPO için düşük (SFT 2e-4'ten 40x az)
    warmup_ratio=0.1,
    max_steps=3,                                 # SMOKE
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    bf16=True,
    seed=3407,
    output_dir="outputs_dpo_qwen35",
    report_to="none",
)

# 4. Trainer — ref_model=None for PEFT auto-handling
trainer = DPOTrainer(
    model=model,
    ref_model=None,                              # PEFT base = reference
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,                  # NOT tokenizer= (TRL 0.24 rename)
)

# 5. Memory snapshot
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start mem = {start_mem} GB")

# 6. Train
print("\n=== TRAINING START ===")
trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM: {used} GB")

# Look for DPO-specific metrics
if 'train_rewards/chosen' in trainer_stats.metrics:
    print(f"Rewards chosen:   {trainer_stats.metrics['train_rewards/chosen']:.4f}")
if 'train_rewards/rejected' in trainer_stats.metrics:
    print(f"Rewards rejected: {trainer_stats.metrics['train_rewards/rejected']:.4f}")
if 'train_rewards/accuracies' in trainer_stats.metrics:
    print(f"Accuracy:         {trainer_stats.metrics['train_rewards/accuracies']:.4f}")

print("\n=== SMOKE TEST DONE ===")
