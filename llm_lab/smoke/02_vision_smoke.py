import unsloth                                  # MUTLAKA EN BAŞTA
from unsloth import FastVisionModel               # FastLanguageModel DEĞİL
from unsloth.trainer import UnslothVisionDataCollator   # Vision için ZORUNLU
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-2B",
    load_in_4bit = False,                  # 16-bit LoRA için False; 4-bit için True
    use_gradient_checkpointing = "unsloth", # Long context için "unsloth" veya True
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,    # Vision encoder tune
    finetune_language_layers   = True,    # Text LM tune
    finetune_attention_modules = True,    # Attention tune
    finetune_mlp_modules       = True,    # MLP tune

    r = 16,                                # 8/16/32/64/128
    lora_alpha = 16,                       # alpha == r
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,                    # RSLoRA destekleniyor
    loftq_config = None,                   # LoftQ destekleniyor
    # target_modules = "all-linear",       # opsiyonel
)

dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
print(dataset)
print(f"\nIlk ornek anahtarlari: {list(dataset[0].keys())}")

# Image'i goruntule
dataset[2]["image"]

# Karsilik gelen LaTeX
print(dataset[2]["text"])

# Vision conversation format'ina cevir
instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text":  instruction},
                {"type": "image", "image": sample["image"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text",  "text": sample["text"]},
            ]},
        ]
    }

# Tüm dataset'i çevir (list comprehension; not Dataset.map for vision)
converted_dataset = [convert_to_conversation(s) for s in dataset]
print(f'Converted: {len(converted_dataset)}')
print(f'\nIlk ornek (mesajlar):')
for m in converted_dataset[0]["messages"]:
    print(f'  role={m["role"]}, content_types={[c["type"] for c in m["content"]]}')

FastVisionModel.for_inference(model)            # Inference modu

image = dataset[2]["image"]
expected = dataset[2]["text"]

messages = [
    {"role": "user", "content": [
        {"type": "image"},                      # Inference'ta sadece placeholder
        {"type": "text", "text": instruction},
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Image tokenizer'a ILK arg olarak gecer (text_only'den farkli)
inputs = tokenizer(
    image,                                      # PIL.Image
    input_text,                                 # text
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 128,
    use_cache = True,
    temperature = 1.5,                          # Qwen3.5 Vision önerisi
    min_p = 0.1,
)

print(f'\n\nExpected: {expected}')

FastVisionModel.for_training(model)             # Training moduna geri al

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),  # ZORUNLU
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 3,                          # demo; production: num_train_epochs=1
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "vision_outputs",
        report_to = "none",

        # Vision SFT için ZORUNLU üçlü:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

# Memory snapshot
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

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

FastVisionModel.for_inference(model)            # Inference modu

image = dataset[2]["image"]
expected = dataset[2]["text"]

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction},
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 128,
    use_cache = True,
    temperature = 1.5, min_p = 0.1,
)

print(f'\n\nExpected: {expected}')

# A. LoRA adapter
# model.save_pretrained("qwen35_vision_lora")
# tokenizer.save_pretrained("qwen35_vision_lora")

# B. Merged 16-bit (vLLM)
# model.save_pretrained_merged(
#     "qwen35_vision_merged",
#     tokenizer,
#     save_method = "merged_16bit",
# )

# C. Hub push
# model.push_to_hub("USER/qwen35_vision_lora", token="hf_xxx")

print("LoRA saved: qwen35_vision_lora/")
