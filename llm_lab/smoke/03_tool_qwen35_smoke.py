"""Tool calling SFT smoke test — Qwen3.5-2B with NATIVE chat template.

Strategy:
- Use AutoTokenizer's native chat_template (NOT get_chat_template)
- Mini synthetic dataset (10 examples) with tool_calls field
- 3 training steps to verify pipeline works
- Verify masking with correct markers
"""
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Model — Qwen3.5-2B is actually multimodal (FastVisionModel) but we can load with FastLanguageModel
#    for text-only tool calling SFT
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-2B",
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,         # Text-only tool calling
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, lora_alpha=8, lora_dropout=0,
    bias="none",
    random_state=3407,
)

# 2. Inspect native template — DO NOT call get_chat_template
print(f"\nNative chat_template length: {len(tokenizer.chat_template)} chars")

# 3. Mini synthetic tool dataset
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

# 4. Format function — passes tools= to native template
def fmt(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, tools=tools, add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

dataset = Dataset.from_list([{"messages": to_messages(r)} for r in raw])
print(f"\nDataset rows: {len(dataset)}")
dataset = dataset.map(fmt, batched=True)

# Show 1 formatted sample
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
        max_length=2048,
    ),
)

# 6. Masking — Qwen3.5 uses <|im_start|>user\n / <|im_start|>assistant\n
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
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

# 8. Quick inference test
from transformers import TextStreamer
print("\n=== INFERENCE TEST ===")
test_msgs = [
    {"role": "system", "content": "You are a helpful assistant with tool access."},
    {"role": "user", "content": "Hesapla 12*7"},
]
text = tokenizer.apply_chat_template(test_msgs, tokenize=False, tools=tools, add_generation_prompt=True)
print(f"Prompt:\n{text[-300:]}")
print("\n--- Generation ---")
inputs = tokenizer(None, text, return_tensors="pt").to("cuda") if hasattr(tokenizer, "image_processor") else tokenizer(text, return_tensors="pt").to("cuda")
_ = model.generate(
    **inputs, max_new_tokens=128,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print("\n=== SMOKE TEST DONE ===")
