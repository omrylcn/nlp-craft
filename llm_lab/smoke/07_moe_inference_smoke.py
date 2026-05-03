"""MoE inference + save smoke test.

Goal: verify after MoE SFT we can:
1. Pre-train inference (model.generate)
2. Post-train inference (after SFT)
3. Save LoRA
4. Save merged 16-bit (does this work for MoE?)
5. Reload + inference from saved
"""
import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 1. Load
MODEL = 'imdatta0/tiny_qwen3_moe_2.8B_0.7B'
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length = 2048,
    load_in_4bit = False,
    fast_inference = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','gate_up_proj'],
    lora_alpha = 64,
    use_gradient_checkpointing = True,
    random_state = 3407, bias = 'none',
)
tokenizer = get_chat_template(tokenizer, chat_template='qwen-3')

# 2. Pre-train inference
print("\n=== PRE-TRAIN INFERENCE ===")
test_msgs = [{'role': 'user', 'content': 'Hesapla 12 * 7'}]
text = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to('cuda')
_ = model.generate(
    **inputs, max_new_tokens=80,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# 3. Quick training (3 step)
print("\n=== TRAINING (3 steps) ===")
dataset = load_dataset('unsloth/OpenMathReasoning-mini', split='cot[:50]')
def to_messages(e):
    return {'conversations': [
        {'role': 'user', 'content': e['problem']},
        {'role': 'assistant', 'content': e['generated_solution']},
    ]}
dataset = dataset.map(to_messages, remove_columns=dataset.column_names)
dataset = dataset.map(lambda e: {'text': [
    tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
    for c in e['conversations']
]}, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field='text',
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        warmup_steps=2, max_steps=3,
        learning_rate=2e-4, logging_steps=1,
        optim='adamw_8bit', weight_decay=0.001,
        lr_scheduler_type='linear', seed=3407,
        report_to='none',
    ),
)
trainer.train()

# 4. Post-train inference
print("\n=== POST-TRAIN INFERENCE ===")
inputs = tokenizer(text, return_tensors='pt').to('cuda')
_ = model.generate(
    **inputs, max_new_tokens=80,
    temperature=0.7, top_p=0.8, top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# 5. Save LoRA
print("\n=== SAVE LoRA ===")
model.save_pretrained('moe_lora_test')
tokenizer.save_pretrained('moe_lora_test')
print("LoRA saved: moe_lora_test/")

# 6. Try merged save (might fail for MoE)
print("\n=== TRY MERGED 16-bit ===")
try:
    model.save_pretrained_merged(
        'moe_merged_test', tokenizer, save_method='merged_16bit',
    )
    print("Merged 16-bit saved successfully!")
except Exception as e:
    print(f"Merged save FAILED: {type(e).__name__}: {str(e)[:200]}")

print("\n=== SMOKE DONE ===")
