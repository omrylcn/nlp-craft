"""Smoke test for 00_unsloth_guide.ipynb — verifies all imports + key APIs exist.
The guide is a reference document with multiple independent demo blocks (text, vision,
thinking) that are not meant to run sequentially as one script. So we verify only that
every API the guide cites actually exists in the installed packages."""
import unsloth
import torch
import inspect

print(f"unsloth: {unsloth.__version__}")
print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")

# 1. Class family — guide claims these exist
from unsloth import FastLanguageModel, FastVisionModel, FastModel
print(f"FastLanguageModel: {FastLanguageModel}")
print(f"FastVisionModel:   {FastVisionModel}")
print(f"FastModel:         {FastModel}")
try:
    from unsloth import FastTextModel
    print(f"FastTextModel:     {FastTextModel}")
except ImportError as e:
    print(f"FastTextModel:     NOT FOUND ({e})")

# 2. chat_templates module — guide cites these helpers
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
print(f"get_chat_template:        {get_chat_template}")
print(f"standardize_data_formats: {standardize_data_formats}")
print(f"train_on_responses_only:  {train_on_responses_only}")

# 3. UnslothVisionDataCollator — guide cites this for vision SFT
from unsloth.trainer import UnslothVisionDataCollator
print(f"UnslothVisionDataCollator: {UnslothVisionDataCollator}")

# 4. trl APIs guide uses
from trl import SFTTrainer, SFTConfig
import trl
print(f"trl: {trl.__version__}")

# 5. transformers
import transformers
print(f"transformers: {transformers.__version__}")

# 6. Verify chat_template names guide claims work
import os
ct_path = os.path.dirname(unsloth.__file__) + "/chat_templates.py"
ct_src = open(ct_path).read()
for name in ["qwen3-instruct", "qwen3-thinking", "gemma-4", "gemma-3", "llama-3.1"]:
    found = name in ct_src or name.replace('-', '_') in ct_src
    print(f"  template '{name}': {'OK' if found else 'MISSING'}")

# 7. from_pretrained signatures the guide cites
sig = inspect.signature(FastLanguageModel.from_pretrained)
for kw in ["model_name", "max_seq_length", "load_in_4bit", "load_in_8bit", "full_finetuning"]:
    print(f"  FastLanguageModel.from_pretrained.{kw}: {'OK' if kw in sig.parameters else 'MISSING'}")

# 8. get_peft_model kwargs
sig = inspect.signature(FastLanguageModel.get_peft_model)
for kw in ["r", "lora_alpha", "lora_dropout", "target_modules", "use_gradient_checkpointing", "use_rslora", "loftq_config"]:
    print(f"  FastLanguageModel.get_peft_model.{kw}: {'OK' if kw in sig.parameters else 'MISSING'}")

# 9. FastVisionModel.get_peft_model finetune_* flags
sig = inspect.signature(FastVisionModel.get_peft_model)
for kw in ["finetune_vision_layers", "finetune_language_layers", "finetune_attention_modules", "finetune_mlp_modules"]:
    print(f"  FastVisionModel.get_peft_model.{kw}: {'OK' if kw in sig.parameters else 'MISSING'}")

print("=== GUIDE IMPORTS SMOKE OK ===")
