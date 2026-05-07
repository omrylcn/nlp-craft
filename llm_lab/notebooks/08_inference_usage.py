import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread

MODEL = 'unsloth/Qwen3-4B-Instruct-2507'

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length = 2048,
    load_in_4bit = True,                       # QLoRA — 4 GB peak
    full_finetuning = False,
)

# Chat template — for the correct format
tokenizer = get_chat_template(tokenizer, chat_template='qwen3-instruct')

FastLanguageModel.for_inference(model)            # 2x faster

# Helper — chat template + generate + decode-only-new-tokens
def gen_text(messages, **gen_kwargs):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs['input_ids'].shape[1]:]   # CRITICAL: skip the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

msgs = [{'role': 'user', 'content': 'What is interest?'}]
print(gen_text(msgs, max_new_tokens=80, do_sample=False))   # greedy

# Greedy (deterministic)
print('--- Greedy ---')
print(gen_text(msgs, max_new_tokens=60, do_sample=False))

# Low temperature (focused)
print('\n--- T=0.3 (focused) ---')
print(gen_text(msgs, max_new_tokens=60, temperature=0.3, do_sample=True))

# High temperature (creative)
print('\n--- T=1.5 (creative) ---')
print(gen_text(msgs, max_new_tokens=60, temperature=1.5, top_p=0.95, do_sample=True))

# Repetition penalty
print('\n--- repetition_penalty=1.3 ---')
print(gen_text(msgs, max_new_tokens=60,
               temperature=0.7, top_p=0.8, top_k=20,
               repetition_penalty=1.3, do_sample=True))

# A. TextStreamer — simple
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')
_ = model.generate(
    **inputs, max_new_tokens=80,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# B. TextIteratorStreamer — production pattern
streamer = TextIteratorStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True,
)
gen_kwargs = dict(
    **inputs, max_new_tokens=80,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    streamer=streamer,
)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

# Token-by-token yield
collected = ''
for chunk in streamer:
    collected += chunk
    print(chunk, end='', flush=True)
thread.join()
print(f'\n\nTotal: {len(collected)} chars')

prompts = [
    'Where is Istanbul?',
    'How does snow form?',
    'What is quantum?',
]
formatted = [
    tokenizer.apply_chat_template(
        [{'role': 'user', 'content': p}],
        tokenize=False, add_generation_prompt=True,
    ) for p in prompts
]

tokenizer.padding_side = 'left'                  # REQUIRED for batching
batch_inputs = tokenizer(
    formatted, return_tensors='pt', padding=True,
).to('cuda')

out = model.generate(
    **batch_inputs, max_new_tokens=50,
    temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)

# Decode each sample
for i, prompt in enumerate(prompts):
    new_tokens = out[i][batch_inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f'\nQ: {prompt}')
    print(f'A: {response[:120]}...')

tokenizer.padding_side = 'right'                 # restore

# Load a thinking model (we loaded Qwen3-Instruct in earlier sections)
# Clear the model to simulate a fresh session
import gc
del model
gc.collect()
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Qwen3-4B-Thinking-2507',
    max_seq_length = 4096,                       # think can be long
    load_in_4bit = True,
)
tokenizer = get_chat_template(tokenizer, chat_template='qwen3-thinking')
FastLanguageModel.for_inference(model)

# Math problem — ideal for a thinking model
msgs = [{'role': 'user', 'content': 'Calculate 137 * 49'}]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')

out = model.generate(
    **inputs, max_new_tokens=1024,                # leave room for think
    temperature=0.6, top_p=0.95, top_k=20,        # recommended for thinking models
    do_sample=True,
)
full = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f'--- Full output ({len(full)} chars) ---')
print(full[:800])

import re

# Strategy 1: capture the full <think>...</think>
think_match = re.search(r'<think>\s*(.*?)\s*</think>', full, re.DOTALL)
if think_match:
    think_part = think_match.group(1).strip()
    final_part = re.sub(r'<think>.*?</think>\s*', '', full, count=1, flags=re.DOTALL).strip()
else:
    # Strategy 2: split on </think> (template prepended <think>, decoder dropped it)
    if '</think>' in full:
        think_part, final_part = full.split('</think>', 1)
        think_part = think_part.strip()
        final_part = final_part.strip()
    else:
        think_part = ''
        final_part = full.strip()

print(f'--- Thinking ({len(think_part)} chars) ---')
print(think_part[:400] + ('...' if len(think_part) > 400 else ''))
print(f'\n--- Final Answer ({len(final_part)} chars) ---')
print(final_part[:400])

print(f'\n--- For UI (only final to user): ---')
print(final_part)

# Load Qwen3-Instruct (ideal for tool calling)
import gc
del model
gc.collect()
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Qwen3-4B-Instruct-2507',
    max_seq_length = 2048,
    load_in_4bit = True,
)
tokenizer = get_chat_template(tokenizer, chat_template='qwen3-instruct')
FastLanguageModel.for_inference(model)

# Define tools — OpenAI function-calling JSON schema
import json

tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get current weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string', 'description': 'City name'}},
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

# Mock tool implementations
def get_weather(city: str) -> dict:
    fake = {'Istanbul': 18, 'Ankara': 12, 'Paris': 15, 'Tokyo': 22}
    return {'city': city, 'temp_c': fake.get(city, 20), 'condition': 'cloudy'}

def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {'__builtins__': {}}, {}))
    except Exception as e:
        return f'Error: {e}'

TOOL_FNS = {'get_weather': get_weather, 'calculator': calculator}

# Turn 1 — User asks, model decides to call tool
conv = [
    {'role': 'system', 'content': 'You are a helpful assistant with tool access.'},
    {'role': 'user',   'content': 'How is the weather in Istanbul?'},
]

text = tokenizer.apply_chat_template(
    conv, tokenize=False, tools=tools, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors='pt').to('cuda')
out = model.generate(
    **inputs, max_new_tokens=200,
    temperature=0.3, top_p=0.8, top_k=20, do_sample=True,    # low T — JSON precision
)
response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('--- Model output ---')
print(response)

# Parse <tool_call>{json}</tool_call> (Qwen3 format) + execute
tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)

if tool_match:
    call = json.loads(tool_match.group(1))
    tool_name = call['name']
    tool_args = call['arguments']
    print(f'Parsed call: {tool_name}({tool_args})')

    # Execute (real Python function)
    if tool_name in TOOL_FNS:
        result = TOOL_FNS[tool_name](**tool_args)
        print(f'Executed → {result}')
    else:
        result = {'error': f'Unknown tool: {tool_name}'}
else:
    print('No tool call detected — model answered directly')
    tool_name = None
    result = None

# Turn 2 — Feed tool result, get final answer
if tool_name:
    conv_full = conv + [
        # Assistant's tool call
        {'role': 'assistant', 'content': '',
         'tool_calls': [{
             'id': 'call_1', 'type': 'function',
             'function': {'name': tool_name, 'arguments': tool_args},
         }]},
        # Tool execution result (role='tool', tool_call_id must match)
        {'role': 'tool', 'tool_call_id': 'call_1', 'name': tool_name,
         'content': json.dumps(result)},
    ]

    text2 = tokenizer.apply_chat_template(
        conv_full, tokenize=False, tools=tools, add_generation_prompt=True,
    )
    inputs2 = tokenizer(text2, return_tensors='pt').to('cuda')
    out2 = model.generate(
        **inputs2, max_new_tokens=150,
        temperature=0.7, top_p=0.8, top_k=20, do_sample=True,
    )
    final = tokenizer.decode(out2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=True)
    print('--- Final answer ---')
    print(final)
