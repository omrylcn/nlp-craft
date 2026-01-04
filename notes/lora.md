# LoRA: Zero to Hero - Comprehensive Guide

## Parameter-Efficient Fine-Tuning from Scratch

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [LoRA Theory: Mathematical Foundations](#2-lora-theory-mathematical-foundations)
3. [Getting Started: Installation](#3-getting-started-installation)
4. [Data Preparation and Preprocessing](#4-data-preparation-and-preprocessing)
5. [Model Loading and Configuration](#5-model-loading-and-configuration)
6. [LoRA Config: Every Parameter in Detail](#6-lora-config-every-parameter-in-detail)
7. [Training Pipeline](#7-training-pipeline)
8. [Model Saving and Loading](#8-model-saving-and-loading)
9. [Inference and Production](#9-inference-and-production)
10. [Advanced: Optimization and Best Practices](#10-advanced-optimization-and-best-practices)

---

## 1. Introduction and Motivation

### 1.1 The Fine-Tuning Problem

Modern large language models (LLMs) contain billions of parameters. For example:

- GPT-3: 175 billion parameters
- LLaMA-7B: 7 billion parameters
- Mistral-7B: 7.3 billion parameters

**Cost of Traditional Fine-Tuning:**

To fine-tune a 7B parameter model in FP32 (32-bit float) format:

- **Memory:** 7B × 4 bytes = 28 GB (model weights only)
- **Gradients:** 28 GB (for backward pass)
- **Optimizer States:** 56 GB (for Adam optimizer)
- **Total:** ~112 GB GPU memory required!

These costs are inaccessible for most researchers and companies. This is where **LoRA** comes in.

### 1.2 LoRA's Solution

LoRA (Low-Rank Adaptation) trains only a small set of parameters by adding **low-rank matrices** instead of updating all model parameters.

**Results:**

- ✅ 99.9% fewer trainable parameters
- ✅ 75% less memory consumption
- ✅ Similar or better performance
- ✅ Multiple adapters can be used with the same base model

---

## 2. LoRA Theory: Mathematical Foundations

### 2.1 Core Idea: Rank Decomposition

In Transformer models, each attention and feedforward layer works with the following formula:

```
h = W₀ · x
```

Where:
- `W₀` → Original pre-trained weight matrix (d × d dimensions)
- `x` → Input vector (d dimensions)
- `h` → Output vector (d dimensions)

**Traditional Fine-Tuning:**

```
W = W₀ + ΔW
```

- `ΔW` → Update the entire d × d matrix (d² parameters)

**LoRA Approach:**

```
W = W₀ + BA
```

Where:
- `B` → Low-rank matrix (d × r dimensions)
- `A` → Low-rank matrix (r × d dimensions)
- `r` → Rank (r << d, e.g., r=8, d=4096)

**Parameter Savings:**

- Traditional: d² = 4096² = 16,777,216 parameters
- LoRA: 2×d×r = 2×4096×8 = 65,536 parameters
- **256x fewer parameters!**

### 2.2 Mathematical Derivation

Forward pass computation:

```
h = W · x
  = (W₀ + BA) · x
  = W₀·x + B·(A·x)
```

**Key Points:**

1. **Initialization:**
   - `A` → Gaussian distribution (N(0, σ²))
   - `B` → Zero matrix
   - Initially: `BA = 0`, so `W = W₀` (identity transform)

2. **Scaling Factor (α):**

   ```
   h = W₀·x + (α/r)·B·(A·x)
   ```

   - `α` → Typical values: 16, 32
   - `α/r` ratio controls learning rate

3. **Gradient Flow:**

   ```
   ∂L/∂B = ∂L/∂h · A·x
   ∂L/∂A = B^T · ∂L/∂h · x
   ```

   - Only `B` and `A` are updated
   - `W₀` remains frozen

### 2.3 Which Layers to Apply?

Transformer Architecture:

```
Input → Embedding
     ↓
┌─────────────────┐
│  Transformer    │
│  Block 1        │
│  ├─ Self-Attn   │ ← LoRA here
│  │  ├─ Q_proj   │ ← LoRA target
│  │  ├─ K_proj   │ ← LoRA target
│  │  ├─ V_proj   │ ← LoRA target
│  │  └─ O_proj   │ ← LoRA target
│  └─ FFN         │
│     ├─ up_proj  │ ← LoRA target (optional)
│     └─ down_proj│ ← LoRA target (optional)
└─────────────────┘
     ↓
   (N blocks)
     ↓
   Output
```

**Common Configurations:**

1. **Minimal:** Only `q_proj` and `v_proj`
   - Fewest parameters
   - Sufficient performance for most tasks

2. **Standard:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Balanced parameter/performance

3. **Aggressive:** All linear layers (attention + FFN)
   - Best performance
   - More parameters

---

## 3. Getting Started: Installation

### 3.1 Requirements

```bash
# Python 3.8+ required
python --version  # Should be Python 3.8+
```

### 3.2 Library Installation

```bash
# Core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face ecosystem
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# LoRA and quantization
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0  # For QLoRA

# Training utilities
pip install trl>=0.7.0  # Supervised Fine-Tuning Trainer
pip install wandb  # Experiment tracking (optional)

# Utilities
pip install scipy
pip install sentencepiece  # For LLaMA tokenizer
pip install protobuf
```

### 3.3 Installation Verification

```python
import torch
import transformers
import peft
import bitsandbytes as bnb

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 4. Data Preparation and Preprocessing

### 4.1 Data Format

Typical data formats for LoRA fine-tuning:

**1. Instruction-Following Format:**

```json
{
  "instruction": "Translate the following text to English:",
  "input": "Bonjour le monde.",
  "output": "Hello world."
}
```

**2. Conversational Format:**

```json
{
  "messages": [
    {"role": "user", "content": "How do I create a list in Python?"},
    {"role": "assistant", "content": "In Python, you can create a list like this: my_list = [1, 2, 3]"}
  ]
}
```

### 4.2 Dataset Loading

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Option 1: From Hugging Face Hub
dataset = load_dataset("databricks/databricks-dolly-15k")

# Option 2: From local CSV
df = pd.read_csv("my_training_data.csv")
dataset = Dataset.from_pandas(df)

# Option 3: From JSON file
dataset = load_dataset("json", data_files="train.json")

print(f"Dataset size: {len(dataset)}")
print(f"Dataset columns: {dataset.column_names}")
print(f"First example:\n{dataset[0]}")
```

### 4.3 Creating Prompt Template

Prompt template is critical for the model to understand the input format.

```python
def create_prompt_template(example):
    """
    Create prompt for instruction-following format.

    Args:
        example: Dict containing 'instruction', 'input', 'output'

    Returns:
        Formatted string
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return prompt
```

### 4.4 Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",  # Important for LoRA!
)

# Add padding token (some models don't have one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    """Batch tokenization function."""
    prompts = [create_prompt_template(ex) for ex in examples]

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )

    # Labels = input_ids (for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# Apply to dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)
```

---

## 5. Model Loading and Configuration

### 5.1 Standard Base Model Loading

```python
from transformers import AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"

# Load model - with FP16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Memory footprint: {total_params * 2 / 1e9:.2f} GB")  # FP16 = 2 bytes
```

### 5.2 QLoRA Model Loading (4-bit Quantization)

QLoRA reduces memory by 75% by quantizing the model to 4-bit.

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.float16,  # Computations in FP16
    bnb_4bit_use_double_quant=True,  # Double quantization (extra 1% savings)
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model loaded in 4-bit")
# 4-bit = 0.5 bytes per parameter
print(f"Estimated memory: {total_params * 0.5 / 1e9:.2f} GB")
```

### 5.3 Gradient Checkpointing for Memory Savings

```python
from peft import prepare_model_for_kbit_training

# Prepare model for kbit training
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# What is gradient checkpointing?
# Instead of saving activations, recompute during backward pass
# Memory: 30-50% reduction
# Time: 20% increase

print("Model prepared for k-bit training with gradient checkpointing")
```

---

## 6. LoRA Config: Every Parameter in Detail

### 6.1 Basic LoraConfig

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    # === RANK AND SCALING ===
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor

    # === TARGET MODULES ===
    target_modules=[
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
    ],

    # === DROPOUT ===
    lora_dropout=0.05,  # Dropout in LoRA layers

    # === BIAS ===
    bias="none",  # "none", "all", or "lora_only"

    # === TASK TYPE ===
    task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling
)
```

### 6.2 Detailed Explanation of Each Parameter

#### 6.2.1 `r` (Rank)

Rank of LoRA matrices. **Most important hyperparameter!**

```python
# What does rank mean?
# B matrix: (d × r)
# A matrix: (r × d)
# BA product: (d × d)

# As rank increases:
# ✅ More expressiveness
# ✅ Better performance (generally)
# ❌ More parameters
# ❌ More memory
```

**Recommended Values:**

- `r=4`: Very small tasks, minimal parameters
- `r=8`: **Standard**, ideal for most tasks
- `r=16`: More complex tasks
- `r=32`: Very complex tasks or domain shift
- `r=64+`: Rarely needed

#### 6.2.2 `lora_alpha` (Scaling Factor)

Controls how effective the LoRA update will be.

```python
# Formula: scaling = lora_alpha / r
# Output: h = W₀·x + (lora_alpha/r) · BA·x

# Example:
# r=8, lora_alpha=16 → scaling=2.0
# r=8, lora_alpha=32 → scaling=4.0
```

**Recommended Values:**

- `lora_alpha = 2 * r` (common rule)
- `r=8` → `lora_alpha=16`
- `r=16` → `lora_alpha=32`

#### 6.2.3 `target_modules`

Which layers to apply LoRA?

```python
# === MINIMAL CONFIG ===
target_modules = ["q_proj", "v_proj"]
# Fewest parameters, fast training
# Sufficient for most tasks

# === STANDARD CONFIG ===
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Balanced performance/parameters
# Recommended starting point

# === AGGRESSIVE CONFIG ===
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"  # FFN
]
# Best performance
# More parameters
```

**Model-Specific Target Modules:**

```python
# LLaMA/Mistral
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# GPT-2/GPT-NeoX
target_modules = ["c_attn", "c_proj"]

# BLOOM
target_modules = ["query_key_value", "dense"]

# T5
target_modules = ["q", "k", "v", "o"]
```

### 6.3 Config Examples: Different Scenarios

```python
# === SCENARIO 1: Minimal Memory (4GB GPU) ===
minimal_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# === SCENARIO 2: Balanced (8-16GB GPU) ===
balanced_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# === SCENARIO 3: High Performance (24GB+ GPU) ===
performance_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="lora_only",
    task_type=TaskType.CAUSAL_LM,
)
```

---

## 7. Training Pipeline

### 7.1 Creating PEFT Model

```python
from peft import get_peft_model

# Base model + LoRA config → PEFT model
peft_model = get_peft_model(model, lora_config)

# Model statistics
peft_model.print_trainable_parameters()
```

**Output:**

```
trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243
```

This output means:
- Only **8.4M parameters** are trained
- Total model has **6.7B parameters**
- **0.12%** parameters are trained (800x fewer!)

### 7.2 Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # === OUTPUT ===
    output_dir="./lora-llama2-7b-output",
    overwrite_output_dir=True,

    # === TRAINING HYPERPARAMETERS ===
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 4*4 = 16

    # === LEARNING RATE ===
    learning_rate=2e-4,  # Typical for LoRA: 1e-4 to 3e-4
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # === OPTIMIZATION ===
    optim="paged_adamw_32bit",  # Optimized for QLoRA
    weight_decay=0.01,
    max_grad_norm=1.0,

    # === PRECISION ===
    fp16=False,
    bf16=False,
    tf32=True,

    # === LOGGING ===
    logging_steps=10,
    logging_dir="./logs",
    report_to="wandb",

    # === CHECKPOINTING ===
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # === EVALUATION ===
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # === EFFICIENCY ===
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)
```

### 7.3 Training with SFTTrainer

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],

    tokenizer=tokenizer,
    formatting_func=create_prompt_template,

    data_collator=data_collator,
    args=training_args,

    packing=False,
    max_seq_length=512,
)

# Start training
trainer.train()
```

---

## 8. Model Saving and Loading

### 8.1 Saving LoRA Adapter

```python
# Save only LoRA weights (very small: ~10-50 MB)
output_dir = "./lora-llama2-adapter"

# Method 1: With Trainer
trainer.save_model(output_dir)

# Method 2: Manual
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Saved files:
# lora-llama2-adapter/
# ├── adapter_config.json  # LoRA config
# ├── adapter_model.bin    # LoRA weights (~10-50 MB)
# └── tokenizer files
```

### 8.2 Saving Merged Model

```python
# Merge base model + LoRA adapter
# Result: Standalone model (no adapter loading needed at inference)

merged_model = peft_model.merge_and_unload()

merged_output_dir = "./lora-llama2-merged"
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)

# File size: ~13-14 GB (full model)
```

**Merge vs Adapter-Only:**

| Feature | Adapter-Only | Merged Model |
|---------|--------------|--------------|
| File size | ~10-50 MB | ~13-14 GB |
| Inference latency | +5-10% overhead | No latency |
| Multiple adapters | ✅ Same base model | ❌ Full model per task |
| Deployment | Base model + adapter required | Standalone |
| Recommended | Development, multi-task | Production, single-task |

### 8.3 Loading LoRA Adapter

```python
from peft import PeftModel, PeftConfig

adapter_path = "./lora-llama2-adapter"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Add LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)

print("LoRA adapter loaded successfully")
```

### 8.4 Multiple Adapter Management

```python
# Scenario: Different adapters for different tasks

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load adapter 1
model = PeftModel.from_pretrained(
    base_model,
    "./adapter-sentiment",
    adapter_name="sentiment"
)

# Add adapter 2
model.load_adapter("./adapter-translation", adapter_name="translation")

# Add adapter 3
model.load_adapter("./adapter-summarization", adapter_name="summarization")

# Select active adapter
model.set_adapter("sentiment")  # For sentiment analysis
# Inference...

model.set_adapter("translation")  # For translation
# Inference...

# Merge multiple adapters (weighted combination)
model.add_weighted_adapter(
    adapters=["sentiment", "translation"],
    weights=[0.7, 0.3],
    adapter_name="sentiment_translation_combo",
    combination_type="cat"  # "cat", "linear", "svd"
)
model.set_adapter("sentiment_translation_combo")
```

---

## 9. Inference and Production

### 9.1 Basic Inference

```python
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")

def generate_response(prompt, max_new_tokens=256, temperature=0.7):
    """Generate response from the model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    return response
```

### 9.2 Generation Parameters

```python
# === GREEDY DECODING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,  # Greedy: always pick highest prob token
)

# === SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,  # Low = conservative, High = creative
)

# === TOP-K SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,  # Sample from top 50 tokens
)

# === TOP-P (NUCLEUS) SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.95,  # Select tokens until cumulative prob reaches 95%
)

# === BEST PRACTICES ===
# Creative tasks (story, poem): temperature=0.8-1.0, top_p=0.95
# Factual tasks (QA, summary): temperature=0.3-0.5, top_p=0.9
# Code generation: temperature=0.2, top_p=0.95, repetition_penalty=1.1
```

### 9.3 Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(prompt, max_new_tokens=256):
    """Stream tokens as they're generated (ChatGPT-style)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

    thread.join()

# Usage
for token in stream_generate(prompt):
    print(token, end="", flush=True)
```

---

## 10. Advanced: Optimization and Best Practices

### 10.1 Hyperparameter Tuning

```python
import optuna

def objective(trial):
    r = trial.suggest_int("r", 4, 32, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", r, r*4, step=r)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Train and evaluate...
    return metrics["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best hyperparameters:", study.best_params)
```

### 10.2 Memory Optimization Techniques

```python
# === TECHNIQUE 1: Flash Attention ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# === TECHNIQUE 2: Gradient Accumulation ===
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    # Effective batch = 1 * 32 = 32
)

# === TECHNIQUE 3: Mixed Precision Training ===
training_args = TrainingArguments(
    bf16=True,  # For Ampere GPUs (A100, H100) - recommended
)

# === TECHNIQUE 4: Activation Checkpointing ===
model.gradient_checkpointing_enable()
# Memory: -40%, Time: +20%
```

### 10.3 Common Issues and Solutions

```python
# === PROBLEM 1: OOM (Out of Memory) ===
# Solution 1: Reduce batch size
training_args.per_device_train_batch_size = 1
# Solution 2: Increase gradient accumulation
training_args.gradient_accumulation_steps = 16
# Solution 3: Enable gradient checkpointing
model.gradient_checkpointing_enable()
# Solution 4: Use QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# === PROBLEM 2: Loss Divergence ===
# Solution 1: Lower learning rate
training_args.learning_rate = 5e-5
# Solution 2: Increase warmup
training_args.warmup_steps = 500
# Solution 3: Add gradient clipping
training_args.max_grad_norm = 0.3

# === PROBLEM 3: Poor Generation Quality ===
# Solution 1: Increase LoRA rank
lora_config.r = 16
# Solution 2: More target modules
lora_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Solution 3: More training epochs
training_args.num_train_epochs = 5
```

---

## 11. Summary and Conclusion

### 11.1 LoRA Advantages

✅ **Parameter Efficiency:** 99.9% fewer trainable parameters
✅ **Memory Savings:** 60-75% less GPU memory
✅ **Fast Training:** Fewer parameters → faster convergence
✅ **Modularity:** Multiple adapters, single base model
✅ **No Catastrophic Forgetting:** Base model frozen → knowledge preserved
✅ **Easy Deployment:** Adapter swap, multi-task serving

### 11.2 LoRA Disadvantages

❌ **Inference Overhead:** 5-10% latency (if not merged)
❌ **Limited Expressiveness:** Less powerful than full fine-tuning
❌ **Learning New Knowledge:** May struggle with domain shift
❌ **Hyperparameter Sensitivity:** r, alpha selection critical

### 11.3 When to Use LoRA?

**LoRA Ideal:**
- ✅ Limited GPU resources (8-16 GB)
- ✅ Instruction-following, alignment tasks
- ✅ Domain adaptation (similar domain)
- ✅ Multi-task scenarios
- ✅ Rapid prototyping

**Prefer Full Fine-Tuning:**
- ❌ Completely new domain (code → medical)
- ❌ Sufficient resources available (100+ GB GPU)
- ❌ Maximum accuracy required
- ❌ Single task, no adapter switching needed

### 11.4 LoRA Variants (2023-2024)

| Variant | Trainable Params | Feature | Use Case |
|---------|------------------|---------|----------|
| LoRA | 2×d×r | Standard low-rank | General purpose |
| QLoRA | 2×d×r | 4-bit quantization | Low memory |
| DoRA | 2×d×r + d | Magnitude-direction | High performance |
| LoRA+ | 2×d×r | Different lr for A and B | Faster convergence |
| rsLoRA | 2×d×r | √r scaling | Stable for high rank |
| VeRA | 2×d | Random frozen + scaling | Ultra low parameter |
| AdaLoRA | Variable | Adaptive rank | Automatic rank selection |

### 11.5 Further Reading and Resources

**Original Papers:**
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- DoRA: https://arxiv.org/abs/2402.09353

**Hugging Face Docs:**
- PEFT: https://huggingface.co/docs/peft
- Transformers: https://huggingface.co/docs/transformers

**Code Repositories:**
- PEFT: https://github.com/huggingface/peft
- TRL: https://github.com/huggingface/trl
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

---

## Quick Reference

```python
# === MINIMAL LoRA PIPELINE ===

# 1. Install
# pip install transformers peft bitsandbytes trl

# 2. Load Model (QLoRA)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 3. LoRA Config
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 4. Train
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()

# 5. Save
model.save_pretrained("./lora-adapter")

# 6. Load & Inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

inputs = tokenizer("Your prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

**Happy Fine-Tuning!**
