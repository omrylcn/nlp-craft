import unsloth                                # MUTLAKA EN BAŞTA
from unsloth import FastModel                  # FastLanguageModel DEĞİL — Gemma 4 multimodal-native
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    dtype = None,                           # auto-detect (bf16 if Ampere+)
    max_seq_length = 1024,                   # uzun context için artır
    load_in_4bit = False,                   # 16-bit LoRA (16GB'a sığar). True = 4-bit QLoRA
    full_finetuning = False,                # [NEW!] full FT artık var
    # token = "YOUR_HF_TOKEN",              # gated modeller için
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,    # Text-only SFT — vision tower'i dondur
    finetune_language_layers   = True,     # LM ana gövde — açık
    finetune_attention_modules = True,     # Attention — GRPO için de iyi
    finetune_mlp_modules       = True,     # MLP — her zaman açık

    r = 8,                                  # Larger = higher accuracy ama overfit riski
    lora_alpha = 8,                         # alpha == r önerilir
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")

# Doğrula
sample_msgs = [
    {"role": "user", "content": "Faiz nedir?"},
    {"role": "assistant", "content": "Faiz, paranin zaman degeridir."},
]
formatted = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
print(formatted)

dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")
print(f'Raw rows: {len(dataset)}')

# standardize_data_formats: ShareGPT/Alpaca → 'conversations' kolonu
dataset = standardize_data_formats(dataset)
print(dataset[0])

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
        for c in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0]["text"][:400])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,       # effective batch = 4
        warmup_steps = 5,
        max_steps = 3,                         # demo; production: num_train_epochs=1
        learning_rate = 2e-4,                   # LoRA için. Long training: 2e-5
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,                   # 0.001 — resmi default
        lr_scheduler_type = "linear",           # linear — resmi default
        seed = 3407,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",     # Gemma 4 markerı
    response_part    = "<|turn>model\n",    # Gemma 4 markerı (assistant değil — 'model'!)
)

# Masking dogrulama — 100. ornek
sample_idx = min(100, len(trainer.train_dataset) - 1)
print("=== FULL INPUT (instruction + response) ===")
print(tokenizer.decode(trainer.train_dataset[sample_idx]["input_ids"]))

print("\n=== ONLY UNMASKED LABELS (sadece response gorunmeli) ===")
print(tokenizer.decode([
    tokenizer.pad_token_id if x == -100 else x
    for x in trainer.train_dataset[sample_idx]["labels"]
]).replace(tokenizer.pad_token, " "))

# Memory snapshot — resmi notebook'lardaki gibi
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")

from transformers import TextStreamer

messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Continue the sequence: 1, 1, 2, 3, 5, 8,"}],
}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,           # ZORUNLU — generation icin
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

_ = model.generate(
    **inputs,
    max_new_tokens = 128,
    temperature = 1.0, top_p = 0.95, top_k = 64,    # Gemma 4 önerisi
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# A. LoRA adapter (en kucuk)
# model.save_pretrained("gemma4_e2b_lora")
# tokenizer.save_pretrained("gemma4_e2b_lora")

# B. Merged 16-bit (vLLM/HF inference)
# model.save_pretrained_merged(
#     "gemma4_e2b_merged",
#     tokenizer,
#     save_method = "merged_16bit",
# )

# C. GGUF (Ollama/llama.cpp)
# model.save_pretrained_gguf(
#     "gemma4_e2b_gguf",
#     tokenizer,
#     quantization_method = "q4_k_m",
# )

# D. Hub'a push
# model.push_to_hub("USER/gemma4_e2b_lora", token="hf_xxx")

print("LoRA saved: gemma4_e2b_lora/")
