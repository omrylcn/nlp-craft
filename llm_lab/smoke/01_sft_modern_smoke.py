import unsloth                                # MUTLAKA EN BAŞTA
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import requests, json

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 2048,
    load_in_4bit = True,                  # QLoRA
    load_in_8bit = False,                 # [NEW!] daha doğru, 2x bellek
    full_finetuning = False,              # [NEW!] full FT artık var
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                                # Choose any > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,
    lora_dropout = 0,                      # 0 — optimized
    bias = "none",                          # "none" — optimized
    use_gradient_checkpointing = "unsloth", # %30 az VRAM
    random_state = 3407,
    use_rslora = False,                    # RSLoRA destekli
    loftq_config = None,                   # LoftQ destekli
)
model.print_trainable_parameters()

tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-instruct")

# Doğrula
sample_msgs = [
    {"role": "user", "content": "Faiz nedir?"},
    {"role": "assistant", "content": "Faiz, paranin zaman degeridir."},
]
formatted = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
print(formatted)

# 1. ch07 instruction-data.json
url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json'
raw = requests.get(url, timeout=30).json()
print(f'Total: {len(raw)}')

# 2. Alpaca → conversations format
def alpaca_to_conversations(entry):
    user_text = entry['instruction']
    if entry.get('input'):
        user_text += f"\n\n{entry['input']}"
    return {
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": entry['output']},
        ]
    }

conversations_data = [alpaca_to_conversations(e) for e in raw]

# 3. Train/test split
n = len(conversations_data)
train_raw = conversations_data[:int(n*0.85)]
test_raw  = conversations_data[int(n*0.85):]

dataset = Dataset.from_list(train_raw)
print(f'Train: {len(dataset)}')

# 4. standardize_data_formats — Unsloth'un format normalizer'i
dataset = standardize_data_formats(dataset)
print(dataset[0])

# 5. formatting_prompts_func — chat template uygulayip 'text' alani olustur
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0]["text"][:300])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,                    # tokenizer= (resmi notebook), processing_class= değil
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,       # effective batch = 8
        warmup_steps = 5,                      # sabit step (warmup_ratio değil)
        max_steps = 3,                        # demo; production: num_train_epochs=1
        learning_rate = 2e-4,                  # LoRA için. Long training: 2e-5
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,                  # 0.001 — resmi default (0.01 değil)
        lr_scheduler_type = "linear",          # linear — resmi default (cosine değil)
        seed = 3407,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# Masking dogru mu? 100. ornegi kontrol et
sample_idx = min(100, len(trainer.train_dataset) - 1)

print("=== FULL INPUT (instruction + response) ===")
print(tokenizer.decode(trainer.train_dataset[sample_idx]["input_ids"]))

print("\n=== ONLY UNMASKED LABELS (sadece response gorunmeli) ===")
print(tokenizer.decode([
    tokenizer.pad_token_id if x == -100 else x
    for x in trainer.train_dataset[sample_idx]["labels"]
]))

# Training oncesi memory snapshot (resmi notebook'lardaki gibi)
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Training sonrasi memory + time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Continue the sequence: 1, 1, 2, 3, 5, 8,"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,             # ZORUNLU
)

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens = 500,
    temperature = 0.7, top_p = 0.8, top_k = 20,    # Qwen3 non-thinking önerisi
    streamer = TextStreamer(tokenizer, skip_prompt=True),
)

# A. LoRA adapter (en kucuk)
# model.save_pretrained("qwen_sft_lora")
# tokenizer.save_pretrained("qwen_sft_lora")

# B. Merged 16-bit (vLLM/HF inference)
# model.save_pretrained_merged(
#     "qwen_sft_merged",
#     tokenizer,
#     save_method = "merged_16bit",
# )

# C. GGUF (Ollama/llama.cpp)
# model.save_pretrained_gguf(
#     "qwen_sft_gguf",
#     tokenizer,
#     quantization_method = "q4_k_m",
# )

# D. Hub'a push
# model.push_to_hub("USER/qwen_sft", token="hf_xxx")

print("LoRA saved: qwen_sft_lora/")
