import unsloth                                              # MUTLAKA EN BAŞTA
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    train_on_responses_only,
)
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'unsloth/Qwen3-4B-Instruct-2507',
    max_seq_length = 2048,
    load_in_4bit = True,                          # QLoRA, 4 GB peak
    full_finetuning = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, lora_alpha = 32, lora_dropout = 0,
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = 3407,
)

# qwen3-instruct (NOT qwen3-thinking — biz sifirdan ogretiyoruz)
tokenizer = get_chat_template(tokenizer, chat_template = 'qwen3-instruct')

test_msgs = [{'role': 'user', 'content': 'Hesapla 137 * 49 ve detayli cozumu goster.'}]
text = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)

print('=== PRE-TRAIN ===')
inputs = tokenizer(text, return_tensors='pt').to('cuda')
_ = model.generate(
    **inputs, max_new_tokens=200,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

dataset = load_dataset('unsloth/OpenMathReasoning-mini', split='cot')
print(f'Total rows: {len(dataset)}')
print(f'Columns: {dataset.column_names}\n')

# Sample 0
print('--- Sample[0] ---')
print(f"Problem: {dataset[0]['problem'][:200]}")
print(f"\nGenerated solution (first 500 chars):")
print(dataset[0]['generated_solution'][:500])

def to_messages(example):
    return {'conversations': [
        {'role': 'user',      'content': example['problem']},
        {'role': 'assistant', 'content': example['generated_solution']},
    ]}

dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

def fmt(examples):
    return {'text': [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples['conversations']
    ]}
dataset = dataset.map(fmt, batched=True)

print('--- Formatted text[0] (first 800 chars) ---')
print(dataset[0]['text'][:800])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = 'text',
        per_device_train_batch_size = 1,
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
    ),
)

# Qwen3 markerlari
trainer = train_on_responses_only(
    trainer,
    instruction_part = '<|im_start|>user\n',
    response_part    = '<|im_start|>assistant\n',
)

# Masking dogrulama — unmasked label <think> ile baslamali
sample = trainer.train_dataset[0]
print('=== UNMASKED LABELS (sadece response gorunmeli, <think> ile baslamali) ===')
decoded = tokenizer.decode([
    tokenizer.pad_token_id if x == -100 else x
    for x in sample['labels']
]).replace(tokenizer.pad_token, ' ')
print(decoded[:800])

gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_mem   = round(gpu_stats.total_memory / 1024**3, 3)
print(f'GPU = {gpu_stats.name} | start mem = {start_mem} GB / {max_mem} GB')

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f'Peak VRAM: {used} GB ({used/max_mem*100:.1f}%)')

print('=== POST-TRAIN INFERENCE ===')
inputs = tokenizer(text, return_tensors='pt').to('cuda')
_ = model.generate(
    **inputs, max_new_tokens=400,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# A. LoRA adapter (en kucuk)
model.save_pretrained('qwen3_thinking_lora')
tokenizer.save_pretrained('qwen3_thinking_lora')
print('LoRA saved: qwen3_thinking_lora/')

# B. Merged 16-bit (vLLM/HF)
# model.save_pretrained_merged(
#     'qwen3_thinking_merged',
#     tokenizer,
#     save_method='merged_16bit',
# )

# C. GGUF (Ollama/llama.cpp)
# model.save_pretrained_gguf(
#     'qwen3_thinking_gguf',
#     tokenizer,
#     quantization_method='q4_k_m',
# )
