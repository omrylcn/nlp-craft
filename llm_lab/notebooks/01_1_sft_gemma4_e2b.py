import unsloth                                # MUST be the very first import
from unsloth import FastModel                  # NOT FastLanguageModel - Gemma 4 is multimodal-native
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
    max_seq_length = 1024,                   # raise for longer context
    load_in_4bit = False,                   # 16-bit LoRA (fits in 16GB). True = 4-bit QLoRA
    full_finetuning = False,                # [NEW!] full FT now available
    # token = "YOUR_HF_TOKEN",              # for gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,    # Text-only SFT - freeze the vision tower
    finetune_language_layers   = True,     # LM body - on
    finetune_attention_modules = True,     # Attention - also good for GRPO
    finetune_mlp_modules       = True,     # MLP - always on

    r = 8,                                  # Larger = higher accuracy but overfit risk
    lora_alpha = 8,                         # alpha == r recommended
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")

# Verify
sample_msgs = [
    {"role": "user", "content": "What is interest?"},
    {"role": "assistant", "content": "Interest is the time value of money."},
]
formatted = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
print(formatted)

dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")
print(f'Raw rows: {len(dataset)}')

# standardize_data_formats: ShareGPT/Alpaca -> 'conversations' column
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
        max_steps = 60,                         # demo; production: num_train_epochs=1
        learning_rate = 2e-4,                   # for LoRA. Long training: 2e-5
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,                   # 0.001 - official default
        lr_scheduler_type = "linear",           # linear - official default
        seed = 3407,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",     # Gemma 4 marker
    response_part    = "<|turn>model\n",    # Gemma 4 marker (not assistant - 'model'!)
)

# Masking verification - sample 100
sample_idx = min(100, len(trainer.train_dataset) - 1)
print("=== FULL INPUT (instruction + response) ===")
print(tokenizer.decode(trainer.train_dataset[sample_idx]["input_ids"]))

print("\n=== ONLY UNMASKED LABELS (should only show the response) ===")
print(tokenizer.decode([
    tokenizer.pad_token_id if x == -100 else x
    for x in trainer.train_dataset[sample_idx]["labels"]
]).replace(tokenizer.pad_token, " "))

# Memory snapshot - as in the official notebooks
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
    add_generation_prompt = True,           # MANDATORY for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

_ = model.generate(
    **inputs,
    max_new_tokens = 128,
    temperature = 1.0, top_p = 0.95, top_k = 64,    # Gemma 4 recommendation
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# A. LoRA adapter (smallest)
model.save_pretrained("gemma4_e2b_lora")
tokenizer.save_pretrained("gemma4_e2b_lora")

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

# D. Push to Hub
# model.push_to_hub("USER/gemma4_e2b_lora", token="hf_xxx")

print("LoRA saved: gemma4_e2b_lora/")
