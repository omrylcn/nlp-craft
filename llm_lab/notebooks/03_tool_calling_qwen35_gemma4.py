import unsloth                                # MUTLAKA EN BASTA
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

print(f'torch: {torch.__version__} | cuda: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 2 fonksiyon tanimla — JSON Schema
tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
        },
    },
}, {
    'type': 'function',
    'function': {
        'name': 'calculator',
        'description': 'Evaluate a math expression',
        'parameters': {
            'type': 'object',
            'properties': {'expression': {'type': 'string'}},
            'required': ['expression'],
        },
    },
}]
print(f'{len(tools)} tools tanimli')

# Sentetik mini dataset
raw = [
    {'q': 'Istanbul hava nasil?',  'tool': 'get_weather', 'args': {'city': 'Istanbul'}, 'result': '{"city": "Istanbul", "temp_c": 18}', 'answer': "Istanbul'da 18°C."},
    {'q': 'Ankara hava nasil?',    'tool': 'get_weather', 'args': {'city': 'Ankara'},   'result': '{"city": "Ankara", "temp_c": 12}',   'answer': "Ankara'da 12°C."},
    {'q': 'Paris hava?',           'tool': 'get_weather', 'args': {'city': 'Paris'},    'result': '{"city": "Paris", "temp_c": 15}',    'answer': "Paris'te 15°C."},
    {'q': 'Tokyo hava?',           'tool': 'get_weather', 'args': {'city': 'Tokyo'},    'result': '{"city": "Tokyo", "temp_c": 22}',    'answer': "Tokyo'da 22°C."},
    {'q': 'London hava?',          'tool': 'get_weather', 'args': {'city': 'London'},   'result': '{"city": "London", "temp_c": 10}',   'answer': "Londra'da 10°C."},
    {'q': 'Hesapla 7*8',           'tool': 'calculator',  'args': {'expression': '7*8'},   'result': '56',   'answer': 'Sonuc 56.'},
    {'q': 'Hesapla 100/4',         'tool': 'calculator',  'args': {'expression': '100/4'}, 'result': '25',   'answer': 'Sonuc 25.'},
    {'q': 'Hesapla 2**10',         'tool': 'calculator',  'args': {'expression': '2**10'}, 'result': '1024', 'answer': 'Sonuc 1024.'},
    {'q': 'Hesapla 15+27',         'tool': 'calculator',  'args': {'expression': '15+27'}, 'result': '42',   'answer': 'Sonuc 42.'},
    {'q': 'Hesapla 99-33',         'tool': 'calculator',  'args': {'expression': '99-33'}, 'result': '66',   'answer': 'Sonuc 66.'},
] * 5  # 50 satir

def to_messages(r):
    return [
        {'role': 'system',    'content': 'You are a helpful assistant with tool access.'},
        {'role': 'user',      'content': r['q']},
        {'role': 'assistant', 'content': '',
         'tool_calls': [{'id': 'c1', 'type': 'function',
                         'function': {'name': r['tool'], 'arguments': r['args']}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'name': r['tool'], 'content': r['result']},
        {'role': 'assistant', 'content': r['answer']},
    ]

raw_messages = [{'messages': to_messages(r)} for r in raw]
print(f'Toplam {len(raw_messages)} satir')
print('\nIlk ornek:')
for m in raw_messages[0]['messages']:
    print(f"  {m['role']:10} {str(m).replace(chr(10), ' ')[:120]}")

from unsloth import FastVisionModel

model_q, tokenizer_q = FastVisionModel.from_pretrained(
    'unsloth/Qwen3.5-2B',
    load_in_4bit = False,                       # 16-bit LoRA (4.7 GB sığar)
    use_gradient_checkpointing = 'unsloth',
)

model_q = FastVisionModel.get_peft_model(
    model_q,
    finetune_vision_layers     = False,         # Text-only tool calling
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 8, lora_alpha = 8, lora_dropout = 0,
    bias = 'none',
    random_state = 3407,
)

# NATIVE template — DO NOT call get_chat_template
print(f'Native template length: {len(tokenizer_q.chat_template)} chars')
print(f'tools branch present:   {"tools" in tokenizer_q.chat_template}')

# Format function — tools= native template'e gecir
def fmt_q(examples):
    return {'text': [
        tokenizer_q.apply_chat_template(
            m, tokenize=False, tools=tools, add_generation_prompt=False,
        )
        for m in examples['messages']
    ]}

dataset_q = Dataset.from_list(raw_messages).map(fmt_q, batched=True)
print(dataset_q[0]['text'][:1000])
print('\n...[TRUNCATED]')

trainer_q = SFTTrainer(
    model = model_q,
    tokenizer = tokenizer_q,
    train_dataset = dataset_q,
    args = SFTConfig(
        dataset_text_field = 'text',
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,                         # demo; production: num_train_epochs=1
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = 'adamw_8bit',
        weight_decay = 0.001,
        lr_scheduler_type = 'linear',
        seed = 3407,
        report_to = 'none',
        max_length = 2048,
    ),
)

# Qwen3.5 markerlari (Qwen3 ile ayni)
trainer_q = train_on_responses_only(
    trainer_q,
    instruction_part = '<|im_start|>user\n',
    response_part    = '<|im_start|>assistant\n',
)

# Masking dogrulama
sample = trainer_q.train_dataset[0]
print('=== MASKED LABELS (sadece response gorunmeli) ===')
print(tokenizer_q.decode([
    tokenizer_q.pad_token_id if x == -100 else x
    for x in sample['labels']
]).replace(tokenizer_q.pad_token, ' ')[:600])

trainer_stats_q = trainer_q.train()
print(f"\nTrain loss: {trainer_stats_q.metrics['train_loss']:.4f}")
print(f"Peak VRAM: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

from transformers import TextStreamer

test_msgs = [
    {'role': 'system', 'content': 'You are a helpful assistant with tool access.'},
    {'role': 'user', 'content': 'Hesapla 12*7'},
]
text = tokenizer_q.apply_chat_template(
    test_msgs, tokenize=False, tools=tools, add_generation_prompt=True,
)
inputs = tokenizer_q(text, return_tensors='pt').to('cuda')
_ = model_q.generate(
    **inputs, max_new_tokens=128,
    temperature=0.7, top_p=0.8, top_k=20,         # Qwen önerisi
    streamer=TextStreamer(tokenizer_q, skip_prompt=True),
)

# Onceki Qwen3.5 model'i bellekten cikar — tek GPU'da iki model sigmaz
import gc
del model_q, tokenizer_q, trainer_q, dataset_q
gc.collect()
torch.cuda.empty_cache()

from unsloth import FastModel

model_g, tokenizer_g = FastModel.from_pretrained(
    model_name = 'unsloth/gemma-4-E2B-it',
    dtype = None,                                # auto bf16
    max_seq_length = 1024,
    load_in_4bit = False,                        # 16-bit (16GB'de 9.9GB peak)
    full_finetuning = False,
)
model_g = FastModel.get_peft_model(
    model_g,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 8, lora_alpha = 8, lora_dropout = 0,
    bias = 'none',
    random_state = 3407,
)

# NATIVE template — 16317 char
print(f'Native template length: {len(tokenizer_g.chat_template)} chars')
print(f'tools branch:           {"tools" in tokenizer_g.chat_template}')

# Format — bos strip ZORUNLU
def fmt_g(examples):
    return {'text': [
        tokenizer_g.apply_chat_template(
            m, tokenize=False, tools=tools, add_generation_prompt=False,
        ).removeprefix('<bos>')                  # KRITIK
        for m in examples['messages']
    ]}

dataset_g = Dataset.from_list(raw_messages).map(fmt_g, batched=True)
print(dataset_g[0]['text'][:1000])
print('\n...[TRUNCATED]')

trainer_g = SFTTrainer(
    model = model_g,
    tokenizer = tokenizer_g,
    train_dataset = dataset_g,
    args = SFTConfig(
        dataset_text_field = 'text',
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = 'adamw_8bit',
        weight_decay = 0.001,
        lr_scheduler_type = 'linear',
        seed = 3407,
        report_to = 'none',
        max_length = 1024,
    ),
)

# Gemma 4 markerlari
trainer_g = train_on_responses_only(
    trainer_g,
    instruction_part = '<|turn>user\n',
    response_part    = '<|turn>model\n',
)

# Masking dogrulama
sample = trainer_g.train_dataset[0]
print('=== MASKED LABELS ===')
print(tokenizer_g.decode([
    tokenizer_g.pad_token_id if x == -100 else x
    for x in sample['labels']
]).replace(tokenizer_g.pad_token, ' ')[:600])

trainer_stats_g = trainer_g.train()
print(f"\nTrain loss: {trainer_stats_g.metrics['train_loss']:.4f}")
print(f"Peak VRAM: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

# Gemma 4 inference — content LIST format zorunlu
test_msgs = [
    {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant with tool access.'}]},
    {'role': 'user',   'content': [{'type': 'text', 'text': 'Hesapla 12*7'}]},
]

inputs = tokenizer_g.apply_chat_template(
    test_msgs,
    add_generation_prompt = True,
    tools = tools,
    return_tensors = 'pt',
    tokenize = True,
    return_dict = True,
).to('cuda')

_ = model_g.generate(
    **inputs, max_new_tokens=128,
    temperature=1.0, top_p=0.95, top_k=64,        # Gemma 4 önerisi
    streamer=TextStreamer(tokenizer_g, skip_prompt=True),
)

# Aktif modelin adapter'ini kaydet
# A. LoRA (en kucuk)
model_g.save_pretrained('gemma4_tool_lora')
tokenizer_g.save_pretrained('gemma4_tool_lora')
print('Gemma 4 tool LoRA saved')

# B. Merged 16-bit (vLLM)
# model_g.save_pretrained_merged('gemma4_tool_merged', tokenizer_g, save_method='merged_16bit')

# C. GGUF (Ollama / llama.cpp)
# model_g.save_pretrained_gguf('gemma4_tool_gguf', tokenizer_g, quantization_method='q4_k_m')
