"""Thinking SFT smoke test — Qwen3-4B-Instruct-2507 (NON-thinking base).

Goal: prove the pipeline that converts a non-thinking instruction-tuned model
into a thinking model via SFT on reasoning traces.

Strategy:
- Base: unsloth/Qwen3-4B-Instruct-2507 (knows chat format, doesn't <think>)
- Dataset: unsloth/OpenMathReasoning-mini (has <think>...</think> traces)
- Pre-train inference: no thinking
- 3-step SFT
- Post-train inference: should attempt thinking format

If 3 steps not enough to learn thinking, that's expected — smoke test
verifies pipeline runs, not convergence.
"""
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — Qwen3-4B-Instruct (NON-thinking)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,
    load_in_4bit=True,                       # 4-bit QLoRA, 4 GB peak
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 2. Chat template — qwen3-instruct (NOT qwen3-thinking; we WANT to teach it thinking)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

# 3. Pre-train inference — verify model does NOT think
from transformers import TextStreamer
print("\n=== PRE-TRAIN INFERENCE (model should NOT use <think>) ===")
test_msgs = [{"role": "user", "content": "Hesapla 137 * 49 ve bana detaylı çözümü göster."}]
text = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
_ = model.generate(
    **inputs, max_new_tokens=200,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# 4. Dataset — OpenMathReasoning-mini cot split
print("\n=== LOADING DATASET ===")
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
print(f"Total rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"\nSample[0]:")
sample = dataset[0]
for k, v in sample.items():
    s = str(v)[:200]
    print(f"  {k}: {s}{'...' if len(str(v))>200 else ''}")

# 5. Convert to messages format
# OpenMathReasoning-mini cot has: 'expected_answer', 'problem_type', 'problem_source',
# 'generation_model', 'pass_rate_72b_tir', 'problem', 'generated_solution', 'inference_mode'
# 'generated_solution' contains <think>...</think> + final answer
def to_messages(example):
    return {"conversations": [
        {"role": "user",      "content": example["problem"]},
        {"role": "assistant", "content": example["generated_solution"]},
    ]}

dataset = dataset.map(to_messages, remove_columns=dataset.column_names)
print(f"\nAfter conversion — sample assistant content:")
print(dataset[0]["conversations"][1]["content"][:500])

# 6. Apply chat template
def fmt(examples):
    return {"text": [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples["conversations"]
    ]}
dataset = dataset.map(fmt, batched=True)
print(f"\nFormatted text[0] (first 800 chars):")
print(dataset[0]["text"][:800])

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=3,                              # SMOKE
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Verify masking — should show <think> in unmasked portion
print("\n=== MASKING CHECK (sample 0, response should include <think>) ===")
sample = trainer.train_dataset[0]
unmasked = [tokenizer.pad_token_id if x == -100 else x for x in sample["labels"]]
decoded = tokenizer.decode(unmasked).replace(tokenizer.pad_token, " ")
print(decoded[:600])

# 8. Train
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start mem = {start_mem} GB")

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']:.2f} sec | loss = {trainer_stats.metrics['train_loss']:.4f}")
print(f"Peak VRAM = {used} GB")

# 9. Post-train inference
print("\n=== POST-TRAIN INFERENCE (3 steps not enough — but pipeline check) ===")
inputs = tokenizer(text, return_tensors="pt").to("cuda")
_ = model.generate(
    **inputs, max_new_tokens=300,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print("\n=== SMOKE TEST DONE ===")
