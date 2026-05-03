"""Vanilla PEFT + TRL SFT smoke test (NO UNSLOTH).

Goal: prove the standard HuggingFace path works on the same hardware.
This is the "before Unsloth" recipe — useful to compare:
- Code complexity (more boilerplate)
- Training speed (slower)
- VRAM (higher peak)
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import requests

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# ============================================================
# 1. Quantization config (4-bit NF4 with double quant)
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # nf4 standard
    bnb_4bit_compute_dtype=torch.bfloat16,        # bf16 compute
    bnb_4bit_use_double_quant=True,               # extra ~0.4 bit savings
)

# ============================================================
# 2. Model + tokenizer (vanilla HF, NO Unsloth)
# ============================================================
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"     # using unsloth-pretrained checkpoint, but loading vanilla

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",                    # fallback if flash-attn missing
)
print(f"\nModel loaded — params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# Required for QLoRA training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False                    # gradient checkpointing compat

# ============================================================
# 3. PEFT LoRA config (same as Unsloth's get_peft_model)
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 4. Dataset — same as 01_sft_modern (Sebastian Raschka ch07)
# ============================================================
url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json'
raw = requests.get(url, timeout=30).json()

def alpaca_to_conversations(entry):
    user_text = entry['instruction']
    if entry.get('input'):
        user_text += f"\n\n{entry['input']}"
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": entry['output']},
    ]}

dataset = Dataset.from_list([alpaca_to_conversations(e) for e in raw[:935]])
print(f"\nDataset rows: {len(dataset)}")

# Apply chat template manually (vanilla path — Unsloth's get_chat_template is not available)
def fmt(examples):
    return {"text": [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in examples["messages"]
    ]}

dataset = dataset.map(fmt, batched=True)
print(f"\nFirst formatted text (200 chars):")
print(dataset[0]["text"][:200])

# ============================================================
# 5. SFTTrainer — same TRL config as Unsloth path
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=3,                              # SMOKE
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=2048,
    ),
    processing_class=tokenizer,
)

# Note: NO Unsloth's train_on_responses_only — that's an Unsloth-specific helper.
# For vanilla path, use TRL's `assistant_only_loss=True` in SFTConfig (Qwen3 supports it):
# But it requires `{% generation %}` keyword in chat template — Qwen3 default has it.
# We skip masking helper here for simplicity (smoke test), full sequence loss.

# ============================================================
# 6. Train
# ============================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start mem = {start_mem} GB")

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n=== RESULTS ===")
print(f"Train runtime: {trainer_stats.metrics['train_runtime']:.2f} sec")
print(f"Train loss:    {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM:     {used} GB")
print(f"Sec / step:    {trainer_stats.metrics['train_runtime']/3:.2f}")
print("\n=== SMOKE TEST DONE ===")
