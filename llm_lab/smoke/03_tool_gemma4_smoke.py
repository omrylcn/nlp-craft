"""Tool calling SFT smoke test — Gemma 4 E2B-it with NATIVE chat template.

Strategy:
- Use AutoTokenizer's native chat_template (NOT get_chat_template)
- Native Gemma 4 template uses FunctionGemma-style markers:
  <|tool>declaration:NAME{...}<tool|>
  <|tool_call>call:NAME{...}<tool_call|>
  <|tool_response>response:NAME{...}<tool_response|>
- Mini synthetic dataset (50 examples, repeated)
- 3 training steps to verify pipeline works
"""
import unsloth
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — FastModel for Gemma 4 (multimodal-native)
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=False,
    full_finetuning=False,
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,         # Text-only tool calling
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, lora_alpha=8, lora_dropout=0,
    bias="none",
    random_state=3407,
)

# 2. NATIVE template — DO NOT call get_chat_template
print(f"\nNative chat_template length: {len(tokenizer.chat_template)} chars")

# 3. Tools + synthetic dataset (same shape as Qwen3.5 test, for comparison)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

raw = [
    {"q": "Istanbul hava nasil?",   "tool": "get_weather", "args": {"city": "Istanbul"}, "result": '{"city": "Istanbul", "temp_c": 18}', "answer": "Istanbul'da 18°C."},
    {"q": "Ankara hava nasil?",     "tool": "get_weather", "args": {"city": "Ankara"},   "result": '{"city": "Ankara", "temp_c": 12}',   "answer": "Ankara'da 12°C."},
    {"q": "Paris hava?",            "tool": "get_weather", "args": {"city": "Paris"},    "result": '{"city": "Paris", "temp_c": 15}',    "answer": "Paris'te 15°C."},
    {"q": "Tokyo hava?",            "tool": "get_weather", "args": {"city": "Tokyo"},    "result": '{"city": "Tokyo", "temp_c": 22}',    "answer": "Tokyo'da 22°C."},
    {"q": "London hava?",           "tool": "get_weather", "args": {"city": "London"},   "result": '{"city": "London", "temp_c": 10}',   "answer": "Londra'da 10°C."},
    {"q": "Hesapla 7*8",            "tool": "calculator",  "args": {"expression": "7*8"}, "result": "56",                                  "answer": "Sonuc 56."},
    {"q": "Hesapla 100/4",          "tool": "calculator",  "args": {"expression": "100/4"}, "result": "25",                                "answer": "Sonuc 25."},
    {"q": "Hesapla 2**10",          "tool": "calculator",  "args": {"expression": "2**10"}, "result": "1024",                              "answer": "Sonuc 1024."},
    {"q": "Hesapla 15+27",          "tool": "calculator",  "args": {"expression": "15+27"}, "result": "42",                                "answer": "Sonuc 42."},
    {"q": "Hesapla 99-33",          "tool": "calculator",  "args": {"expression": "99-33"}, "result": "66",                                "answer": "Sonuc 66."},
] * 5  # 50 rows

def to_messages(r):
    return [
        {"role": "system",    "content": "You are a helpful assistant with tool access."},
        {"role": "user",      "content": r["q"]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": r["tool"], "arguments": r["args"]}}]},
        {"role": "tool", "tool_call_id": "c1", "name": r["tool"], "content": r["result"]},
        {"role": "assistant", "content": r["answer"]},
    ]

# 4. Format function — passes tools= to native template; STRIP <bos>
def fmt(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, tools=tools, add_generation_prompt=False,
        )
        texts.append(text.removeprefix("<bos>"))   # CRITICAL: bos strip for Gemma 4
    return {"text": texts}

dataset = Dataset.from_list([{"messages": to_messages(r)} for r in raw])
print(f"\nDataset rows: {len(dataset)}")
dataset = dataset.map(fmt, batched=True)

print("\n=== SAMPLE FORMATTED ===")
print(dataset[0]["text"][:1500])
print("...[TRUNCATED]")

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=3,                       # SMOKE
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        max_length=1024,
    ),
)

# 6. Masking — Gemma 4 markers
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

print("\n=== MASKING CHECK (sample 0) ===")
sample = trainer.train_dataset[0]
unmasked = [tokenizer.pad_token_id if x == -100 else x for x in sample["labels"]]
print(tokenizer.decode(unmasked).replace(tokenizer.pad_token, " ")[:800])

# 7. Train
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nGPU = {gpu_stats.name} | start memory = {start_mem} GB")

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n{trainer_stats.metrics['train_runtime']} seconds")
print(f"Peak reserved memory = {used} GB")
print(f"Train loss = {trainer_stats.metrics['train_loss']}")

# 8. Inference test
from transformers import TextStreamer
print("\n=== INFERENCE TEST ===")
test_msgs = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant with tool access."}]},
    {"role": "user",   "content": [{"type": "text", "text": "Hesapla 12*7"}]},
]
inputs = tokenizer.apply_chat_template(
    test_msgs,
    add_generation_prompt=True,
    tools=tools,
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
).to("cuda")

prompt_text = tokenizer.apply_chat_template(test_msgs, tokenize=False, tools=tools, add_generation_prompt=True)
print(f"Prompt (last 300 chars):\n{prompt_text[-300:]}")
print("\n--- Generation ---")
_ = model.generate(
    **inputs, max_new_tokens=128,
    temperature=1.0, top_p=0.95, top_k=64,    # Gemma 4 önerisi
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print("\n=== SMOKE TEST DONE ===")
