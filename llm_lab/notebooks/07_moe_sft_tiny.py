import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'    # MUTLAKA unsloth import'undan ONCE

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

MODEL = 'imdatta0/tiny_qwen3_moe_2.8B_0.7B'         # T4-uyumlu
max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length = max_seq_length,
    load_in_4bit = False,                            # MoE: 4-bit YOK
    fast_inference = False,                          # MoE: vLLM YOK
)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
        'gate_up_proj',                               # MoE expert layers — KRITIK
    ],
    lora_alpha = lora_rank * 2,                       # 2x — resmi recipe
    use_gradient_checkpointing = True,
    random_state = 3407,
    bias = 'none',
)

tokenizer = get_chat_template(tokenizer, chat_template='qwen-3')

dataset = load_dataset('unsloth/OpenMathReasoning-mini', split='cot')
print(f'Total rows: {len(dataset)}')

def to_messages(example):
    return {'conversations': [
        {'role': 'user', 'content': example['problem']},
        {'role': 'assistant', 'content': example['generated_solution']},
    ]}
dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

def fmt(examples):
    return {'text': [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples['conversations']
    ]}
dataset = dataset.map(fmt, batched=True)
print(f'\nFirst 200 chars:')
print(dataset[0]['text'][:200])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = 'text',
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 50,                              # demo; production: num_train_epochs=1
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = 'adamw_8bit',
        weight_decay = 0.001,
        lr_scheduler_type = 'linear',
        seed = 3407,
        report_to = 'none',
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_mem = round(gpu_stats.total_memory / 1024**3, 3)
print(f'GPU = {gpu_stats.name} | start mem = {start_mem} / {max_mem} GB')

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n=== RESULTS ===")
print(f"Train runtime: {trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss:    {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM:     {used} GB")
