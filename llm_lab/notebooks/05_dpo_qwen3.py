import unsloth                                # MUTLAKA EN BASTA
from unsloth import PatchDPOTrainer
PatchDPOTrainer()                            # KRITIK: trl import'undan ONCE

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'unsloth/Qwen3-4B-Instruct-2507',
    max_seq_length = 4096,
    load_in_4bit = True,                     # 4-bit QLoRA (4GB peak)
    full_finetuning = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                                   # DPO icin yuksek (SFT 16-32 vs DPO 64)
    lora_alpha = 64,
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout = 0,
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = 3407,
)

# Padding side — DPOTrainer requires LEFT
tokenizer.padding_side = 'left'
print(f'Padding: {tokenizer.padding_side}')

dataset = load_dataset('trl-lib/ultrafeedback_binarized', split='train[:5000]')
print(f'Dataset rows: {len(dataset)}')
print(f'Cols: {dataset.column_names}\n')

# Sample
sample = dataset[0]
print('--- Sample[0] ---')
for k, v in sample.items():
    s = str(v)[:300]
    print(f'{k}: {s}{"..." if len(str(v))>300 else ""}')

training_args = DPOConfig(
    # ----- DPO-specific -----
    beta = 0.1,
    loss_type = 'sigmoid',
    max_length = 1024,
    max_prompt_length = 512,
    # ----- HF Trainer -----
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,         # effective batch = 8
    learning_rate = 5e-6,                    # DPO icin DUSUK
    warmup_ratio = 0.1,
    num_train_epochs = 1,                    # DPO 1-3 epoch arasi yeterli
    logging_steps = 10,
    optim = 'adamw_8bit',
    weight_decay = 0.0,
    lr_scheduler_type = 'linear',
    bf16 = True,
    seed = 3407,
    output_dir = 'outputs_dpo',
    report_to = 'none',
)

trainer = DPOTrainer(
    model = model,
    ref_model = None,                        # PEFT auto-handles (KEY!)
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer,            # NOT 'tokenizer=' (TRL 0.24 rename)
)

gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_mem = round(gpu_stats.total_memory / 1024**3, 3)
print(f'GPU = {gpu_stats.name} | start mem = {start_mem} / {max_mem} GB')

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f'Peak VRAM: {used} GB')

for k in ['train_rewards/chosen', 'train_rewards/rejected', 'train_rewards/margins', 'train_rewards/accuracies']:
    if k in trainer_stats.metrics:
        print(f'{k}: {trainer_stats.metrics[k]:.4f}')

# A. LoRA adapter (en kucuk)
model.save_pretrained('qwen3_dpo_lora')
tokenizer.save_pretrained('qwen3_dpo_lora')
print('LoRA saved: qwen3_dpo_lora/')

# B. Merged 16-bit (vLLM/HF)
# model.save_pretrained_merged(
#     'qwen3_dpo_merged', tokenizer, save_method='merged_16bit',
# )

# C. GGUF
# model.save_pretrained_gguf('qwen3_dpo_gguf', tokenizer, quantization_method='q4_k_m')
