import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import requests

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

MODEL_NAME = 'unsloth/Qwen3-4B-Instruct-2507'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map = 'auto',
    dtype = torch.bfloat16,
    attn_implementation = 'sdpa',                # 'flash_attention_2' if installed
)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params loaded')

# QLoRA için ZORUNLU
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False                   # gradient checkpointing compat

lora_config = LoraConfig(
    r = 32,
    lora_alpha = 32,
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout = 0,
    bias = 'none',
    task_type = 'CAUSAL_LM',                     # PEFT'te ZORUNLU (Unsloth otomatik)
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json'
raw = requests.get(url, timeout=30).json()

def alpaca_to_conversations(entry):
    user_text = entry['instruction']
    if entry.get('input'):
        user_text += f"\n\n{entry['input']}"
    return {'messages': [
        {'role': 'user',      'content': user_text},
        {'role': 'assistant', 'content': entry['output']},
    ]}

n = len(raw)
train_raw = raw[:int(n*0.85)]
dataset = Dataset.from_list([alpaca_to_conversations(e) for e in train_raw])
print(f'Train: {len(dataset)}')

# Format — vanilla'da Unsloth'un standardize_data_formats yok
# Direkt apply_chat_template
def fmt(examples):
    return {'text': [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in examples['messages']
    ]}
dataset = dataset.map(fmt, batched=True)
print('\nFirst 200 chars:')
print(dataset[0]['text'][:200])

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = 'text',
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,                          # demo; production: num_train_epochs=1
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = 'adamw_8bit',
        weight_decay = 0.001,
        lr_scheduler_type = 'linear',
        seed = 3407,
        report_to = 'none',
        bf16 = True,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant': False},
        max_length = 2048,
        # assistant_only_loss = True,            # Qwen3 chat template `{% generation %}` destekliyor
    ),
    processing_class = tokenizer,
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
print(f"Sec / step:    {trainer_stats.metrics['train_runtime']/60:.2f}")

# A. LoRA adapter (PEFT standart)
model.save_pretrained('vanilla_sft_lora')
tokenizer.save_pretrained('vanilla_sft_lora')
print('LoRA saved: vanilla_sft_lora/')

# B. Merged 16-bit (PEFT'in merge_and_unload metodu)
# merged = model.merge_and_unload()
# merged.save_pretrained('vanilla_sft_merged', safe_serialization=True)
# tokenizer.save_pretrained('vanilla_sft_merged')

# Dikkat: Unsloth'un save_pretrained_merged ve save_pretrained_gguf metodları yok vanilla'da.
# GGUF için ayrıca llama.cpp convert-hf-to-gguf.py kullan.
