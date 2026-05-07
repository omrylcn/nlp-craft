import unsloth                                # MUST be the very first import
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
    CHAT_TEMPLATES,                           # raw dict
)
from transformers import AutoTokenizer
from datasets import Dataset

# Available templates
print(f'{len(CHAT_TEMPLATES)} templates registered')
print(sorted(CHAT_TEMPLATES.keys()))

TOK_FOR = {
    'qwen3-instruct':   'unsloth/Qwen3-4B-Instruct-2507',
    'qwen3-thinking':   'unsloth/Qwen3-4B-Thinking-2507',
    'gemma-4':          'unsloth/gemma-4-E2B-it',
    'gemma-4-thinking': 'unsloth/gemma-4-E2B-it',     # same tokenizer, different template
    'llama-3.1':        'unsloth/Llama-3.1-8B-Instruct',
}

tokenizers = {}
for name, path in TOK_FOR.items():
    tok = AutoTokenizer.from_pretrained(path)
    tok = get_chat_template(tok, chat_template=name)
    tokenizers[name] = tok
    print(f'OK: {name}')

msgs = [
    {'role': 'user',      'content': 'What is interest?'},
    {'role': 'assistant', 'content': 'Interest is the time value of money.'},
]

for name, tok in tokenizers.items():
    print('=' * 60)
    print(name)
    print('=' * 60)
    print(tok.apply_chat_template(msgs, tokenize=False))
    print()

user_only = [{'role': 'user', 'content': 'What is interest?'}]

for name, tok in tokenizers.items():
    no_gen   = tok.apply_chat_template(user_only, tokenize=False, add_generation_prompt=False)
    with_gen = tok.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    suffix   = with_gen[len(no_gen):]
    print(f'{name:20} appended = {suffix!r}')

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
}]
test_msgs = [{'role': 'user', 'content': 'What is the weather in Istanbul?'}]

for name, tok in tokenizers.items():
    without = tok.apply_chat_template(test_msgs, tokenize=False)
    with_t  = tok.apply_chat_template(test_msgs, tokenize=False, tools=tools)
    supported = with_t != without
    diff = len(with_t) - len(without)
    print(f'{name:20} tools= supported = {supported}  (diff: +{diff} chars)')

for name in ['qwen3-instruct', 'qwen3-thinking', 'gemma-4', 'gemma-4-thinking']:
    tok = tokenizers[name]
    print(f'\n--- {name} ---')
    for et in [False, True]:
        try:
            out = tok.apply_chat_template(
                [{'role': 'user', 'content': 'What is interest?'}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=et,
            )
            tail = out[-80:]
            print(f'  enable_thinking={et}: ...{tail!r}')
        except Exception as e:
            print(f'  enable_thinking={et}: FAIL {e}')

multi = [
    {'role': 'system',    'content': 'You are a helpful Turkish assistant.'},
    {'role': 'user',      'content': 'What is interest?'},
    {'role': 'assistant', 'content': 'Interest is the time value of money.'},
    {'role': 'user',      'content': 'Give an example.'},
]

for name in ['qwen3-instruct', 'gemma-4']:
    print('=' * 60)
    print(name)
    print('=' * 60)
    print(tokenizers[name].apply_chat_template(multi, tokenize=False, add_generation_prompt=True))
    print()

# 8a. Alpaca -> unchanged (no 'conversations' col)
alpaca = Dataset.from_list([
    {'instruction': 'Define interest', 'input': '', 'output': 'Interest is the time value of money.'},
    {'instruction': 'Translate',       'input': 'hello', 'output': 'merhaba'},
])
print('--- Alpaca ---')
print(f'BEFORE cols: {alpaca.column_names}')
out = standardize_data_formats(alpaca)
print(f'AFTER  cols: {out.column_names}')
print(f'Sample[0]:   {out[0]}')
print('=> Alpaca UNCHANGED (manual conversion required)')

# 8b. ShareGPT -> converted (needs at least 2 rows for role detection)
sharegpt = Dataset.from_list([
    {'conversations': [
        {'from': 'human', 'value': 'Define interest'},
        {'from': 'gpt',   'value': 'Interest is the time value of money.'},
    ]},
    {'conversations': [
        {'from': 'human', 'value': 'Translate hello'},
        {'from': 'gpt',   'value': 'merhaba'},
    ]},
])
print('--- ShareGPT ---')
print(f'BEFORE: {sharegpt[0]}')
out = standardize_data_formats(sharegpt)
print(f'AFTER:  {out[0]}')
print('=> ShareGPT (from/value/human/gpt) -> OpenAI (role/content/user/assistant)')

# 8c. OpenAI messages -> unchanged (no 'conversations' col, function exits early)
openai = Dataset.from_list([
    {'messages': [
        {'role': 'user',      'content': 'Define interest'},
        {'role': 'assistant', 'content': 'Interest is the time value of money.'},
    ]},
])
print('--- OpenAI ---')
print(f'BEFORE cols: {openai.column_names}')
out = standardize_data_formats(openai)
print(f'AFTER  cols: {out.column_names}')
print('=> OpenAI UNCHANGED (already in the correct format)')

# 9a. Qwen3 - standard pattern
tok = tokenizers['qwen3-instruct']

def fmt_qwen3(examples):
    return {'text': [
        tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples['conversations']
    ]}

# Test
data = Dataset.from_list([
    {'conversations': [
        {'role': 'user', 'content': 'What is interest?'},
        {'role': 'assistant', 'content': 'Interest is the time value of money.'},
    ]},
])
out = data.map(fmt_qwen3, batched=True)
print(out[0]['text'])

# 9b. Gemma 4 - '<bos>' STRIP is mandatory
# Double bos -> broken model. The processor will add it itself.
tok = tokenizers['gemma-4']

def fmt_gemma4(examples):
    return {'text': [
        tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
        for c in examples['conversations']
    ]}

out = data.map(fmt_gemma4, batched=True)
print('After Gemma 4 format (no bos at start):')
print(repr(out[0]['text'][:100]))

# Vision conversation - image + text mixed
vision_msg = {
    'messages': [
        {'role': 'user', 'content': [
            {'type': 'text',  'text': 'What is the LaTeX in this image?'},
            {'type': 'image', 'image': 'https://example.com/img.png'},  # must be a real PIL.Image
        ]},
        {'role': 'assistant', 'content': [
            {'type': 'text', 'text': r'\frac{a}{b}'},
        ]},
    ]
}
print('Vision format:')
import json
print(json.dumps(vision_msg, indent=2, ensure_ascii=False))

# Tool calling SFT - multi-turn assistant -> tool -> assistant
tool_conv = [
    {'role': 'system', 'content': 'You are a weather assistant.'},
    {'role': 'user', 'content': 'What is the weather in Istanbul?'},
    {'role': 'assistant', 'content': '',
     'tool_calls': [{
         'id': 'call_1', 'type': 'function',
         'function': {'name': 'get_weather', 'arguments': {'city': 'Istanbul'}}
     }]},
    {'role': 'tool', 'tool_call_id': 'call_1', 'name': 'get_weather',
     'content': '{"city": "Istanbul", "temp_c": 18, "condition": "cloudy"}'},
    {'role': 'assistant', 'content': 'It is currently 18°C and cloudy in Istanbul.'},
]

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
}]

tok = tokenizers['qwen3-instruct']
rendered = tok.apply_chat_template(tool_conv, tokenize=False, tools=tools)
print(rendered)
