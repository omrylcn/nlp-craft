# LoRA: Zero to Hero - Kapsamlı Rehber

## Sıfırdan İleriye Parameter-Efficient Fine-Tuning

---

## 📚 İçindekiler

1. [Giriş ve Motivasyon](#1-giriş-ve-motivasyon)
2. [LoRA Teorisi: Matematiksel Temeller](#2-lora-teorisi-matematiksel-temeller)
3. [Getting Started: Kurulum](#3-getting-started-kurulum)
4. [Veri Hazırlama ve Preprocessing](#4-veri-hazırlama-ve-preprocessing)
5. [Model Yükleme ve Konfigürasyon](#5-model-yükleme-ve-konfigürasyon)
6. [LoRA Config: Her Parametre Detaylı](#6-lora-config-her-parametre-detaylı)
7. [Training Pipeline](#7-training-pipeline)
8. [Model Kaydetme ve Yükleme](#8-model-kaydetme-ve-yükleme)
9. [Inference ve Production](#9-inference-ve-production)
10. [İleri Seviye: Optimizasyon ve Best Practices](#10-i̇leri-seviye-optimizasyon-ve-best-practices)

---

## 1. Giriş ve Motivasyon

### 1.1 Fine-Tuning Problemi

Modern büyük dil modelleri (LLM) milyarlarca parametre içerir. Örneğin:

- GPT-3: 175 milyar parametre
- LLaMA-7B: 7 milyar parametre
- Mistral-7B: 7.3 milyar parametre

**Geleneksel Fine-Tuning'in Maliyeti:**

Bir 7B parametreli modeli FP32 (32-bit float) formatında fine-tune etmek için:

- **Bellek:** 7B × 4 byte = 28 GB (sadece model ağırlıkları)
- **Gradients:** 28 GB (backward pass için)
- **Optimizer States:** 56 GB (Adam optimizer için)
- **Toplam:** ~112 GB GPU belleği gerekir!

Bu maliyetler çoğu araştırmacı ve şirket için erişilemez. İşte bu noktada **LoRA** devreye giriyor.

### 1.2 LoRA'nın Çözümü

LoRA (Low-Rank Adaptation), modelin tüm parametrelerini güncellemek yerine, **düşük rankli (low-rank) matrisler** ekleyerek sadece küçük bir parametre kümesini eğitir.

**Sonuç:**

- ✅ %99.9 daha az eğitilebilir parametre
- ✅ %75 daha az bellek tüketimi
- ✅ Benzer veya daha iyi performans
- ✅ Birden fazla adapter aynı base model ile kullanılabilir

---

## 2. LoRA Teorisi: Matematiksel Temeller

### 2.1 Temel Fikir: Rank Decomposition

Transformer modellerinde her attention ve feedforward layer aşağıdaki formülle çalışır:

```
h = W₀ · x
```

Burada:

- `W₀` → Orijinal pre-trained weight matrix (d × d boyutunda)
- `x` → Input vektörü (d boyutunda)
- `h` → Output vektörü (d boyutunda)

**Geleneksel Fine-Tuning:**

```
W = W₀ + ΔW
```

- `ΔW` → Tüm d × d matrisini güncelle (d² parametre)

**LoRA Yaklaşımı:**

```
W = W₀ + BA
```

Burada:

- `B` → Düşük rankli matrix (d × r boyutunda)
- `A` → Düşük rankli matrix (r × d boyutunda)
- `r` → Rank (r << d, örneğin r=8, d=4096)

**Parametre Tasarrufu:**

- Geleneksel: d² = 4096² = 16,777,216 parametre
- LoRA: 2×d×r = 2×4096×8 = 65,536 parametre
- **256x daha az parametre!**

### 2.2 Matematiksel Derivation

Forward pass hesaplaması:

```
h = W · x
  = (W₀ + BA) · x
  = W₀·x + B·(A·x)
```

**Önemli Noktalar:**

1. **Initialization:**
   - `A` → Gaussian dağılımı (N(0, σ²))
   - `B` → Sıfır matris
   - İlk durumda: `BA = 0`, yani `W = W₀` (identity transform)

2. **Scaling Factor (α):**

   ```
   h = W₀·x + (α/r)·B·(A·x)
   ```

   - `α` → Tipik değer: 16, 32
   - `α/r` oranı öğrenme hızını kontrol eder

3. **Gradient Flow:**

   ```
   ∂L/∂B = ∂L/∂h · A·x
   ∂L/∂A = B^T · ∂L/∂h · x
   ```

   - Sadece `B` ve `A` güncellenir
   - `W₀` frozen (donmuş) kalır

### 2.3 Hangi Layer'lara Uygulanır?

Transformer Architecture:

```
Input → Embedding
     ↓
┌─────────────────┐
│  Transformer    │
│  Block 1        │
│  ├─ Self-Attn   │ ← LoRA burada
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

**Yaygın Konfigürasyonlar:**

1. **Minimal:** Sadece `q_proj` ve `v_proj`
   - En az parametre
   - Yeterli performans çoğu task için

2. **Standard:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Dengeli parametre/performans

3. **Aggressive:** Tüm linear layers (attention + FFN)
   - En iyi performans
   - Daha fazla parametre

---

## 3. Getting Started: Kurulum

### 3.1 Gereksinimler

```bash
# Python 3.8+ gerekli
python --version  # Python 3.8+ olmalı
```

### 3.2 Kütüphane Kurulumu

```bash
# Temel kütüphaneler
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face ekosistemi
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# LoRA ve quantization
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0  # QLoRA için

# Training utilities
pip install trl>=0.7.0  # Supervised Fine-Tuning Trainer
pip install wandb  # Experiment tracking (optional)

# Utilities
pip install scipy
pip install sentencepiece  # LLaMA tokenizer için
pip install protobuf
```

### 3.3 Kurulum Doğrulama

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

**Beklenen Çıktı:**

```
PyTorch version: 2.1.0+cu118
Transformers version: 4.35.2
PEFT version: 0.7.1
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA A100-SXM4-40GB
```

---

## 4. Veri Hazırlama ve Preprocessing

### 4.1 Veri Formatı

LoRA fine-tuning için tipik veri formatları:

**1. Instruction-Following Format:**

```json
{
  "instruction": "Şu metni Türkçeye çevir:",
  "input": "The weather is nice today.",
  "output": "Bugün hava güzel."
}
```

**2. Conversational Format:**

```json
{
  "messages": [
    {"role": "user", "content": "Python'da liste nasıl oluşturulur?"},
    {"role": "assistant", "content": "Python'da liste şu şekilde oluşturulur: my_list = [1, 2, 3]"}
  ]
}
```

### 4.2 Dataset Yükleme

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Seçenek 1: Hugging Face Hub'dan
dataset = load_dataset("databricks/databricks-dolly-15k")

# Seçenek 2: Yerel CSV'den
df = pd.read_csv("my_training_data.csv")
dataset = Dataset.from_pandas(df)

# Seçenek 3: JSON dosyasından
dataset = load_dataset("json", data_files="train.json")

print(f"Dataset size: {len(dataset)}")
print(f"Dataset columns: {dataset.column_names}")
print(f"First example:\n{dataset[0]}")
```

### 4.3 Prompt Template Oluşturma

Prompt template, modelin girdi formatını anlaması için kritiktir.

```python
def create_prompt_template(example):
    """
    Instruction-following format için prompt oluştur.
    
    Args:
        example: Dict containing 'instruction', 'input', 'output'
    
    Returns:
        Formatted string
    """
    # Template yapısı
    # ### Instruction: <görev tanımı>
    # ### Input: <opsiyonel girdi>
    # ### Response: <beklenen çıktı>
    
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Input varsa ekle, yoksa sadece instruction
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

# Örnek kullanım
example = {
    "instruction": "Aşağıdaki cümleyi olumlu mu olumsuz mu sınıflandır.",
    "input": "Bu film harikaydı, çok beğendim!",
    "output": "Olumlu"
}

formatted_prompt = create_prompt_template(example)
print(formatted_prompt)
```

**Çıktı:**

```
### Instruction:
Aşağıdaki cümleyi olumlu mu olumsuz mu sınıflandır.

### Input:
Bu film harikaydı, çok beğendim!

### Response:
Olumlu
```

### 4.4 Tokenization

```python
from transformers import AutoTokenizer

# Tokenizer yükle
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",  # LoRA için önemli!
)

# Padding token ekle (bazı modellerde yok)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    """
    Batch tokenization fonksiyonu.
    
    Args:
        examples: Dict of lists, her key bir column
        
    Returns:
        Tokenized dict with input_ids, attention_mask, labels
    """
    # Promptları oluştur
    prompts = [create_prompt_template(ex) for ex in examples]
    
    # Tokenize et
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,  # Model'e göre ayarla
        padding="max_length",
        return_tensors=None  # List olarak dön
    )
    
    # Labels = input_ids (causal LM için)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Dataset'e uygula
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names,  # Orijinal columnları kaldır
    desc="Tokenizing dataset"
)

print(f"Tokenized dataset: {tokenized_dataset}")
print(f"First tokenized example shape: {len(tokenized_dataset[0]['input_ids'])}")
```

### 4.5 Data Collator

Data collator, batch'leri dinamik olarak oluşturur.

```python
from transformers import DataCollatorForLanguageModeling

# Causal LM için data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Masked LM değil, Causal LM
)

# Test et
sample_batch = [tokenized_dataset[i] for i in range(3)]
collated = data_collator(sample_batch)

print("Collated batch keys:", collated.keys())
print("Batch input_ids shape:", collated["input_ids"].shape)
print("Batch attention_mask shape:", collated["attention_mask"].shape)
```

---

## 5. Model Yükleme ve Konfigürasyon

### 5.1 Base Model Yükleme (Standard)

```python
from transformers import AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"

# Model yükle - FP16 ile
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",  # Otomatik GPU dağıtımı
    trust_remote_code=True,
)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")

# Model parametrelerini say
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Memory footprint: {total_params * 2 / 1e9:.2f} GB")  # FP16 = 2 bytes
```

### 5.2 QLoRA ile Model Yükleme (4-bit Quantization)

QLoRA, modeli 4-bit'e indirerek belleği %75 azaltır.

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization aktif
    bnb_4bit_quant_type="nf4",  # NormalFloat4 veri tipi
    bnb_4bit_compute_dtype=torch.float16,  # Hesaplamalar FP16'da
    bnb_4bit_use_double_quant=True,  # Double quantization (ekstra %1 tasarruf)
)

# Model yükle
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

**BitsAndBytesConfig Parametreleri:**

- `load_in_4bit`: 4-bit quantization aktif et
- `bnb_4bit_quant_type`:
  - `"nf4"` → NormalFloat4 (önerilen)
  - `"fp4"` → Regular Float4
- `bnb_4bit_compute_dtype`: Matmul için dtype (FP16 veya BF16)
- `bnb_4bit_use_double_quant`: Quantization constants'ı da quantize et

### 5.3 Model için Gradient Checkpointing

Bellek tasarrufu için gradient checkpointing aktif et.

```python
from peft import prepare_model_for_kbit_training

# Model'i kbit training için hazırla
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True  # Gradient checkpointing aktif
)

# Gradient checkpointing nedir?
# Aktivasyonları kaydetmek yerine, backward pass'te yeniden hesapla
# Bellek: %30-50 azaltma
# Süre: %20 artış

print("Model prepared for k-bit training with gradient checkpointing")
```

---

## 6. LoRA Config: Her Parametre Detaylı

### 6.1 Temel LoraConfig

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    # === RANK VE SCALING ===
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
    lora_dropout=0.05,  # LoRA layer'larında dropout
    
    # === BIAS ===
    bias="none",  # "none", "all", veya "lora_only"
    
    # === TASK TYPE ===
    task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling
)
```

### 6.2 Her Parametrenin Detaylı Açıklaması

#### 6.2.1 `r` (Rank)

LoRA matrislerinin rankı. **En önemli hyperparameter!**

```python
# Rank ne demek?
# B matrix: (d × r)
# A matrix: (r × d)
# BA product: (d × d)

# Rank arttıkça:
# ✅ Daha fazla expressiveness
# ✅ Daha iyi performans (genelde)
# ❌ Daha fazla parametre
# ❌ Daha fazla bellek
```

**Önerilen Değerler:**

- `r=4`: Çok küçük tasklar, minimal parametre
- `r=8`: **Standard**, çoğu task için ideal
- `r=16`: Daha karmaşık tasklar
- `r=32`: Çok karmaşık tasklar veya domain shift
- `r=64+`: Nadiren gerekli

**Parametre Hesabı:**

```python
def calculate_lora_params(d_model, r, num_target_modules):
    """
    Args:
        d_model: Model hidden dimension (örn: 4096)
        r: LoRA rank
        num_target_modules: Kaç modül hedeflenecek (örn: 4 = q,k,v,o)
    """
    params_per_module = 2 * d_model * r  # B ve A matrisleri
    total_params = params_per_module * num_target_modules
    return total_params

# LLaMA-7B için hesap (d_model=4096, 32 transformer layers)
d_model = 4096
num_layers = 32
num_target_modules = 4  # q,k,v,o

for r in [4, 8, 16, 32]:
    params_per_layer = calculate_lora_params(d_model, r, num_target_modules)
    total_params = params_per_layer * num_layers
    print(f"r={r}: {total_params:,} parameters ({total_params / 1e6:.2f}M)")
```

**Çıktı:**

```
r=4: 4,194,304 parameters (4.19M)
r=8: 8,388,608 parameters (8.39M)
r=16: 16,777,216 parameters (16.78M)
r=32: 33,554,432 parameters (33.55M)
```

#### 6.2.2 `lora_alpha` (Scaling Factor)

LoRA güncellemesinin ne kadar etkili olacağını kontrol eder.

```python
# Formül: scaling = lora_alpha / r
# Output: h = W₀·x + (lora_alpha/r) · BA·x

# Örnek:
# r=8, lora_alpha=16 → scaling=2.0
# r=8, lora_alpha=32 → scaling=4.0

# Yüksek alpha:
# ✅ Daha agresif güncelleme
# ❌ Instability riski

# Düşük alpha:
# ✅ Daha stable training
# ❌ Yavaş öğrenme
```

**Önerilen Değerler:**

- `lora_alpha = 2 * r` (yaygın kural)
- `r=8` → `lora_alpha=16`
- `r=16` → `lora_alpha=32`

#### 6.2.3 `target_modules`

Hangi layer'lara LoRA uygulanacak?

```python
# === MINIMAL CONFIG ===
target_modules = ["q_proj", "v_proj"]
# En az parametre, hızlı training
# Çoğu task için yeterli

# === STANDARD CONFIG ===
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Dengeli performans/parametre
# Önerilen başlangıç

# === AGGRESSIVE CONFIG ===
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"  # FFN
]
# En iyi performans
# Daha fazla parametre

# === REGEX PATTERN ===
target_modules = ".*attn.*"  # Tüm attention layer'ları
target_modules = ".*proj"  # 'proj' ile biten tüm layer'lar
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

#### 6.2.4 `lora_dropout`

LoRA layer'larında dropout oranı.

```python
# Dropout, overfitting'i önler
# LoRA layer'larının output'una uygulanır

# Değer aralığı: 0.0 - 0.2
lora_dropout = 0.0  # Dropout yok
lora_dropout = 0.05  # Standard (önerilen)
lora_dropout = 0.1  # Daha agresif regularization
```

#### 6.2.5 `bias`

Bias parametrelerini nasıl eğiteceğiz?

```python
# === "none" ===
bias = "none"
# Hiçbir bias eğitilmez
# En az parametre
# Çoğu durumda yeterli

# === "all" ===
bias = "all"
# Model'deki tüm bias'lar eğitilir
# Daha fazla parametre
# Daha iyi performans (bazı durumlarda)

# === "lora_only" ===
bias = "lora_only"
# Sadece LoRA layer'larındaki bias'lar eğitilir
# Orta yol

# Önerilen: "none" (çoğu durumda)
```

#### 6.2.6 `task_type`

Ne tür bir task için eğitim yapılıyor?

```python
from peft import TaskType

# === CAUSAL_LM ===
task_type = TaskType.CAUSAL_LM
# GPT-style modeller için
# LLaMA, Mistral, GPT-2, etc.
# Next-token prediction

# === SEQ_2_SEQ_LM ===
task_type = TaskType.SEQ_2_SEQ_LM
# T5, BART, etc.
# Sequence-to-sequence tasks

# === QUESTION_ANSWERING ===
task_type = TaskType.QUESTION_ANS
# QA-specific models

# === TOKEN_CLASSIFICATION ===
task_type = TaskType.TOKEN_CLS
# NER, POS tagging
```

### 6.3 İleri Seviye Parametreler

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    
    # === İLERİ SEVİYE ===
    
    # Layer-specific rank
    rank_pattern={
        ".*layers.0.*": 4,  # İlk layer'lara düşük rank
        ".*layers.(1[0-9]|2[0-9]|3[01]).*": 16,  # Son layer'lara yüksek rank
    },
    
    # Layer-specific alpha
    alpha_pattern={
        ".*layers.0.*": 8,
        ".*layers.(1[0-9]|2[0-9]|3[01]).*": 32,
    },
    
    # Specific layers to transform
    layers_to_transform=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    
    # Modules to save (model head gibi)
    modules_to_save=["lm_head"],
    
    # Initialization method
    init_lora_weights=True,  # True, False, veya "gaussian"
    
    # RS-LoRA (Rank-Stabilized LoRA)
    use_rslora=False,  # True ise scaling = lora_alpha / sqrt(r)
)
```

### 6.4 Config Örnekleri: Farklı Senaryolar

```python
# === SENARYO 1: Minimal Bellek (4GB GPU) ===
minimal_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# === SENARYO 2: Balanced (8-16GB GPU) ===
balanced_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# === SENARYO 3: High Performance (24GB+ GPU) ===
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

# === SENARYO 4: Domain Shift (Yeni domain öğrenme) ===
domain_shift_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="all",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["embed_tokens", "lm_head"],  # Embeddings de eğit
)
```

---

## 7. Training Pipeline

### 7.1 PEFT Model Oluşturma

```python
from peft import get_peft_model

# Base model + LoRA config → PEFT model
peft_model = get_peft_model(model, lora_config)

# Model istatistikleri
peft_model.print_trainable_parameters()
```

**Çıktı:**

```
trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243
```

Bu çıktı şu anlama gelir:

- Sadece **8.4M parametre** eğitiliyor
- Toplam model **6.7B parametre**
- **%0.12** parametre eğitiliyor (800x daha az!)

### 7.2 Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # === OUTPUT ===
    output_dir="./lora-llama2-7b-output",  # Checkpoint dizini
    overwrite_output_dir=True,
    
    # === TRAINING HYPERPARAMETERS ===
    num_train_epochs=3,  # Epoch sayısı
    per_device_train_batch_size=4,  # GPU başına batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 4*4 = 16
    
    # === LEARNING RATE ===
    learning_rate=2e-4,  # LoRA için tipik: 1e-4 to 3e-4
    lr_scheduler_type="cosine",  # "linear", "cosine", "constant"
    warmup_steps=100,  # Warmup step sayısı
    
    # === OPTIMIZATION ===
    optim="paged_adamw_32bit",  # QLoRA için optimize edilmiş
    weight_decay=0.01,
    max_grad_norm=1.0,  # Gradient clipping
    
    # === PRECISION ===
    fp16=False,  # QLoRA için False
    bf16=False,  # A100/H100 için True
    tf32=True,  # Ampere GPU'larda hızlandırma
    
    # === LOGGING ===
    logging_steps=10,
    logging_dir="./logs",
    report_to="wandb",  # "wandb", "tensorboard", veya "none"
    
    # === CHECKPOINTING ===
    save_strategy="steps",  # "steps" veya "epoch"
    save_steps=500,
    save_total_limit=3,  # En fazla 3 checkpoint sakla
    
    # === EVALUATION ===
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # === EFFICIENCY ===
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,  # Multi-GPU için
)
```

#### 7.2.1 Batch Size ve Gradient Accumulation

```python
# Effective Batch Size hesabı:
# effective_batch = per_device_batch * num_gpus * gradient_accumulation_steps

# Örnek 1: Single GPU
per_device_batch = 4
gradient_accumulation = 4
effective_batch = 4 * 1 * 4 = 16

# Örnek 2: 4 GPU
per_device_batch = 2
gradient_accumulation = 8
num_gpus = 4
effective_batch = 2 * 4 * 8 = 64

# Neden gradient accumulation?
# - Büyük batch size için yetersiz VRAM
# - Küçük batch'leri accumulate et → büyük batch effect
```

#### 7.2.2 Learning Rate Seçimi

```python
# LoRA için önerilen LR:
# - Full fine-tuning: 1e-5 to 5e-5
# - LoRA: 1e-4 to 3e-4 (10x daha yüksek!)

# Neden daha yüksek LR?
# - LoRA, düşük rankli bir subspace'te öğreniyor
# - Daha az parametre → daha hızlı convergence gerekli
# - Orijinal weights frozen → overfitting riski düşük

# LR scheduler türleri:
lr_scheduler_type = "cosine"  # Smooth decay (önerilen)
lr_scheduler_type = "linear"  # Linear decay
lr_scheduler_type = "constant"  # Sabit LR
lr_scheduler_type = "constant_with_warmup"  # Warmup + sabit
```

#### 7.2.3 Optimizer Seçimi

```python
# === STANDART ADAMW ===
optim = "adamw_torch"  # PyTorch native AdamW
#장점: Stable, well-tested
# 단점: Yüksek bellek (optimizer states)

# === PAGED ADAMW (QLoRA için) ===
optim = "paged_adamw_32bit"
optim = "paged_adamw_8bit"
# Özellik: Optimizer states'i CPU'ya page eder
#장점: %50 daha az GPU memory
# 단점:約간 더 yavaş

# === ADAFACTOR ===
optim = "adafactor"
# Özellik: Stateless optimizer
#장점: Çok düşük memory
# 단점: Bazen unstable

# QLoRA için önerilen: paged_adamw_32bit
```

### 7.3 SFTTrainer ile Training

```python
from trl import SFTTrainer

# SFTTrainer: Supervised Fine-Tuning için optimize edilmiş trainer
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    
    # Tokenizer ve formatting
    tokenizer=tokenizer,
    formatting_func=create_prompt_template,  # Prompt template func
    
    # Data collator
    data_collator=data_collator,
    
    # Training args
    args=training_args,
    
    # Packing (isteğe bağlı)
    packing=False,  # True ise multiple examples'ı bir sequence'e pack et
    max_seq_length=512,
)

# Training başlat
trainer.train()
```

#### 7.3.1 SFTTrainer vs Standard Trainer

```python
# === STANDARD TRAINER ===
from transformers import Trainer

# Manuel preprocessing gerekli
# Daha fazla kod yazmak gerekiyor

# === SFT TRAINER ===
from trl import SFTTrainer

#장점:
# ✅ Instruction-following için optimize edilmiş
# ✅ Otomatik prompt formatting
# ✅ Packing desteği (efficiency)
# ✅ Dataset formatting utilities
# ✅ Daha az kod

# SFTTrainer, özellikle instruction-tuning için önerilen
```

### 7.4 Training Loop İçinde Neler Oluyor?

```python
# Pseudo-code: Her training step
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        
        # 2. Backward pass
        loss.backward()  # Sadece LoRA params için gradient hesapla
        
        # 3. Gradient accumulation
        if step % gradient_accumulation_steps == 0:
            # 4. Optimizer step
            optimizer.step()  # Sadece LoRA weights güncelle (B, A)
            optimizer.zero_grad()
            
        # 5. Logging
        if step % logging_steps == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        
        # 6. Checkpoint
        if step % save_steps == 0:
            trainer.save_model(f"checkpoint-{step}")
```

### 7.5 Monitoring Training

#### 7.5.1 WandB Integration

```python
import wandb

# WandB başlat
wandb.init(
    project="lora-llama2-finetuning",
    name="llama2-7b-instruction-tuning",
    config={
        "learning_rate": 2e-4,
        "batch_size": 16,
        "lora_r": 8,
        "lora_alpha": 16,
    }
)

# Training args'da report_to ayarla
training_args = TrainingArguments(
    # ...
    report_to="wandb",
    run_name="llama2-7b-instruction-tuning",
)

# Training başlat (otomatik log olur)
trainer.train()
```

#### 7.5.2 TensorBoard

```python
# Training args
training_args = TrainingArguments(
    # ...
    report_to="tensorboard",
    logging_dir="./logs",
)

# TensorBoard başlat (terminal'de)
# tensorboard --logdir=./logs
```

### 7.6 Training Tips ve Best Practices

```python
# === TIP 1: Gradient Checkpointing ===
# Bellek tasarrufu: %30-50
# Süre artışı: %20
# Önerilen: Her zaman açık (özellikle QLoRA ile)
model.gradient_checkpointing_enable()

# === TIP 2: Batch Size Tuning ===
# En büyük batch size'ı bul (OOM olmadan)
# Büyük batch → daha stable training
for batch_size in [1, 2, 4, 8, 16]:
    try:
        trainer = SFTTrainer(
            # ...
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                # ...
            )
        )
        print(f"Batch size {batch_size} works!")
    except RuntimeError as e:
        print(f"Batch size {batch_size} OOM")
        break

# === TIP 3: Learning Rate Finder ===
# LR çok yüksek → divergence
# LR çok düşük → yavaş convergence
# Optimal LR bul:
from transformers.trainer_utils import get_scheduler

# Test different LRs
for lr in [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]:
    print(f"Testing LR: {lr}")
    # Small training run
    # Monitor loss curve

# === TIP 4: Warmup Steps ===
# Warmup: LR'yi 0'dan target'a kademeli artır
# Önerilen: total_steps'in %5-10'u
total_steps = len(train_dataset) / batch_size * num_epochs
warmup_steps = int(0.05 * total_steps)

# === TIP 5: Early Stopping ===
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # 3 eval'da improvement yoksa dur
    early_stopping_threshold=0.001,  # Minimum improvement threshold
)

trainer = SFTTrainer(
    # ...
    callbacks=[early_stopping],
)
```

---

## 8. Model Kaydetme ve Yükleme

### 8.1 LoRA Adapter'ı Kaydetme

```python
# Sadece LoRA weights'leri kaydet (çok küçük: ~10-50 MB)
output_dir = "./lora-llama2-adapter"

# Yöntem 1: Trainer ile
trainer.save_model(output_dir)

# Yöntem 2: Manuel
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Kaydedilen dosyalar:
# lora-llama2-adapter/
# ├── adapter_config.json  # LoRA config
# ├── adapter_model.bin    # LoRA weights (~10-50 MB)
# └── tokenizer files

print(f"LoRA adapter saved to {output_dir}")

# Dosya boyutunu kontrol et
import os
adapter_size = os.path.getsize(f"{output_dir}/adapter_model.bin") / (1024**2)
print(f"Adapter size: {adapter_size:.2f} MB")
```

### 8.2 Merged Model Kaydetme

```python
# Base model + LoRA adapter'ı birleştir
# Sonuç: Standalone model (inference'ta adapter yüklemeye gerek yok)

# Yöntem 1: merge_and_unload
merged_model = peft_model.merge_and_unload()

# Birleştirilmiş modeli kaydet
merged_output_dir = "./lora-llama2-merged"
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)

# Dosya boyutu: ~13-14 GB (full model)
print(f"Merged model saved to {merged_output_dir}")

# Uyarı: Merge sonrası adapter'ı unload edemezsiniz!
```

**Merge vs Adapter-Only:**

| Özellik | Adapter-Only | Merged Model |
|---------|--------------|--------------|
| Dosya boyutu | ~10-50 MB | ~13-14 GB |
| Inference latency | +5-10% overhead | Latency yok |
| Çoklu adapter | ✅ Aynı base model ile | ❌ Her task için full model |
| Deployment | Base model + adapter gerekli | Standalone |
| Önerilen | Development, multi-task | Production, single-task |

### 8.3 LoRA Adapter'ı Yükleme

```python
from peft import PeftModel, PeftConfig

# Yöntem 1: Direkt yükleme
adapter_path = "./lora-llama2-adapter"

# Base model yükle
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# LoRA adapter'ı ekle
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)

print("LoRA adapter loaded successfully")

# Yöntem 2: QLoRA ile yükleme
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)
```

### 8.4 Çoklu Adapter Yönetimi

```python
# Senaryo: Farklı task'lar için farklı adapter'lar

# Adapter 1: Sentiment Analysis
# Adapter 2: Translation
# Adapter 3: Summarization

# Base model bir kez yükle
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Adapter 1 yükle
model = PeftModel.from_pretrained(
    base_model,
    "./adapter-sentiment",
    adapter_name="sentiment"
)

# Adapter 2 ekle
model.load_adapter("./adapter-translation", adapter_name="translation")

# Adapter 3 ekle
model.load_adapter("./adapter-summarization", adapter_name="summarization")

# Aktif adapter'ı seç
model.set_adapter("sentiment")  # Sentiment analysis için
# Inference...

model.set_adapter("translation")  # Translation için
# Inference...

# Birden fazla adapter'ı birleştir (weighted combination)
model.add_weighted_adapter(
    adapters=["sentiment", "translation"],
    weights=[0.7, 0.3],
    adapter_name="sentiment_translation_combo",
    combination_type="cat"  # "cat", "linear", "svd"
)
model.set_adapter("sentiment_translation_combo")

# Adapter'ı kaldır
model.delete_adapter("sentiment")
```

### 8.5 Hugging Face Hub'a Yükleme

```python
from huggingface_hub import login, HfApi

# 1. Login
login(token="your_hf_token_here")

# 2. Model'i Hub'a push et
model_id = "your-username/lora-llama2-sentiment"

# Adapter-only
peft_model.push_to_hub(
    model_id,
    commit_message="Initial commit: LoRA adapter for sentiment analysis",
    private=False,  # Public repo
)

tokenizer.push_to_hub(model_id)

# 3. Model card oluştur
model_card = f"""
---
license: llama2
base_model: meta-llama/Llama-2-7b-hf
tags:
- peft
- lora
- sentiment-analysis
---

# LoRA Adapter: LLaMA-2 7B Sentiment Analysis

This is a LoRA adapter for sentiment analysis, trained on [dataset_name].

## Training Details
- Base model: meta-llama/Llama-2-7b-hf
- LoRA rank: 8
- LoRA alpha: 16
- Training dataset: [dataset_name]
- Training time: [duration]

## Usage
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "{model_id}")
```

"""

# Model card'ı kaydet

with open("README.md", "w") as f:
    f.write(model_card)

# Hub'dan yükleme

# model = PeftModel.from_pretrained(base_model, "your-username/lora-llama2-sentiment")

```

---

## 9. Inference ve Production

### 9.1 Basic Inference

```python
# Model ve tokenizer yükle
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
model.eval()  # Evaluation mode

tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")

def generate_response(prompt, max_new_tokens=256, temperature=0.7):
    """
    Generate response from the model.
    
    Args:
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy, 1.0 = random)
    
    Returns:
        Generated text
    """
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,  # Sampling mode
            top_p=0.95,  # Nucleus sampling
            top_k=50,  # Top-k sampling
            repetition_penalty=1.1,  # Tekrarı önle
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove input prompt from output
    response = generated_text[len(prompt):].strip()
    
    return response

# Örnek kullanım
prompt = """### Instruction:
Aşağıdaki cümlenin duygusunu belirle.

### Input:
Bu film gerçekten harikaydı!

### Response:
"""

response = generate_response(prompt)
print(response)
```

### 9.2 Generation Parameters Detaylı

```python
# === GREEDY DECODING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,  # Greedy: her zaman en yüksek prob token seç
)
#장점: Deterministik, tutarlı
# 단점: Tekrarlayıcı, yaratıcı değil

# === SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,  # Sampling aktif
    temperature=0.7,  # Düşük = conservative, Yüksek = creative
)
# temperature=0.1 → neredeyse greedy
# temperature=1.0 → neutral sampling
# temperature=2.0 → çok random

# === TOP-K SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,  # En yüksek 50 token'dan sample et
)
# top_k=1 → greedy
# top_k=50 → dengeli
# top_k=100 → daha çeşitli

# === TOP-P (NUCLEUS) SAMPLING ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.95,  # Kümülatif prob %95'e ulaşana kadar token seç
)
# top_p=0.9 → güvenli
# top_p=0.95 → dengeli (önerilen)
# top_p=1.0 → tüm vocabulary

# === REPETITION PENALTY ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    repetition_penalty=1.2,  # >1.0: tekrarı cezalandır
)
# 1.0: penalty yok
# 1.2: hafif penalty (önerilen)
# 1.5+: agresif penalty

# === BEAM SEARCH ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,  # 5 beam
    early_stopping=True,
    no_repeat_ngram_size=2,  # 2-gram tekrarını engelle
)
# num_beams=1: greedy
# num_beams=3-5: dengeli
# num_beams=10+: çok yavaş

# === BEST PRACTICES ===
# Yaratıcı task (story, poem): temperature=0.8-1.0, top_p=0.95
# Faktual task (QA, summary): temperature=0.3-0.5, top_p=0.9
# Code generation: temperature=0.2, top_p=0.95, repetition_penalty=1.1
```

### 9.3 Batch Inference

```python
def batch_generate(prompts, batch_size=8, **generation_kwargs):
    """
    Batch inference for multiple prompts.
    
    Args:
        prompts: List of input prompts
        batch_size: Number of prompts per batch
        **generation_kwargs: Arguments for model.generate()
    
    Returns:
        List of generated responses
    """
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,  # Pad to same length
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        
        # Decode
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove input prompts
        for prompt, response in zip(batch_prompts, batch_responses):
            clean_response = response[len(prompt):].strip()
            responses.append(clean_response)
    
    return responses

# Kullanım
prompts = [
    "### Instruction: Summarize this text.\n### Input: ...\n### Response:",
    "### Instruction: Translate to Turkish.\n### Input: ...\n### Response:",
    # ... 100 prompts
]

responses = batch_generate(
    prompts,
    batch_size=16,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
)
```

### 9.4 Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(prompt, max_new_tokens=256):
    """
    Stream tokens as they're generated (ChatGPT-style).
    
    Args:
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
    
    Yields:
        Generated tokens one by one
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    # Generation kwargs
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "streamer": streamer,
    }
    
    # Start generation in separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens
    for new_text in streamer:
        yield new_text
    
    thread.join()

# Kullanım
prompt = "### Instruction: Write a story about a robot.\n### Response:"

print("Generating story...")
for token in stream_generate(prompt):
    print(token, end="", flush=True)
print("\n\nDone!")
```

### 9.5 Production Deployment: FastAPI Örneği

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# === MODEL YÜKLEME (STARTUP) ===
app = FastAPI(title="LoRA LLM API")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    metadata: dict

# Global variables
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """Model'i startup'ta yükle"""
    global model, tokenizer
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(base_model, "./lora-adapter")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./lora-adapter")
    print("Model loaded successfully!")

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Text generation endpoint"""
    try:
        # Tokenize
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(request.prompt):].strip()
        
        return GenerationResponse(
            generated_text=response_text,
            prompt=request.prompt,
            metadata={
                "model": "llama2-7b-lora",
                "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0]),
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Client kullanımı:
# import requests
# response = requests.post(
#     "http://localhost:8000/generate",
#     json={
#         "prompt": "### Instruction: ...\n### Response:",
#         "max_tokens": 128,
#         "temperature": 0.7,
#     }
# )
# print(response.json()["generated_text"])
```

---

## 10. İleri Seviye: Optimizasyon ve Best Practices

### 10.1 Hyperparameter Tuning

```python
import optuna

def objective(trial):
    """Optuna objective function for hyperparameter search"""
    
    # Hyperparameters to tune
    r = trial.suggest_int("r", 4, 32, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", r, r*4, step=r)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    
    # Create config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=f"./optuna-trial-{trial.number}",
        num_train_epochs=1,  # Hızlı deneme
        per_device_train_batch_size=4,
        learning_rate=learning_rate,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="no",  # Save yapma (hız için)
    )
    
    # Train
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=small_train_dataset,  # Küçük subset
        eval_dataset=small_eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    return metrics["eval_loss"]  # Minimize loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Best hyperparameters
print("Best hyperparameters:")
print(study.best_params)
print(f"Best eval loss: {study.best_value:.4f}")

# Visualization
import optuna.visualization as vis
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
```

### 10.2 Quantization Strategies

#### 10.2.1 QLoRA (4-bit)

```python
# 4-bit quantization ile maksimum bellek tasarrufu
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Bellek kullanımı:
# FP16: ~14 GB
# 8-bit: ~7 GB
# 4-bit: ~3.5 GB
# 4-bit + double quant: ~3.2 GB
```

#### 10.2.2 8-bit Quantization

```python
# 8-bit: 4-bit'ten daha az agresif, daha stable
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outlier detection threshold
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

#### 10.2.3 GPTQ Quantization

```python
# GPTQ: Inference için optimize edilmiş quantization
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    model_basename="model",
    use_safetensors=True,
    device_map="auto",
)

# GPTQ장점:
# ✅ Inference'ta hızlı (custom CUDA kernels)
# ✅ 4-bit ama QLoRA'dan daha az accuracy loss
# 단점:
# ❌ Quantization süreci uzun (calibration data gerekli)
# ❌ Training için uygun değil
```

### 10.3 Multi-GPU Training

```python
# Seçenek 1: DataParallel (kolay ama yavaş)
model = torch.nn.DataParallel(model)

# Seçenek 2: DistributedDataParallel (önerilen)
# Launch script: torchrun veya accelerate

# train.py
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Training loop
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# Launch:
# accelerate launch --multi_gpu --num_processes=4 train.py

# Seçenek 3: FSDP (Fully Sharded Data Parallel)
# En büyük modeller için (70B+)
training_args = TrainingArguments(
    # ...
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"
    },
)
```

### 10.4 Memory Optimization Techniques

```python
# === TEKNIK 1: Flash Attention ===
# %30-50 hızlanma, %30-40 bellek tasarrufu
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Flash Attention 2
    device_map="auto",
)

# === TEKNIK 2: Gradient Accumulation ===
# Küçük batch'leri accumulate et → büyük effective batch
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Her GPU'da 1
    gradient_accumulation_steps=32,  # 32 step accumulate
    # Effective batch = 1 * 32 = 32
)

# === TEKNIK 3: CPU Offloading ===
# Infrequent layer'ları CPU'da tut
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
)

# === TEKNIK 4: Mixed Precision Training ===
training_args = TrainingArguments(
    # ...
    fp16=True,  # Pascal/Volta GPU için
    # veya
    bf16=True,  # Ampere GPU (A100, H100) için - önerilen
)

# === TEKNIK 5: Activation Checkpointing ===
model.gradient_checkpointing_enable()
# Activations'ı save etme, backward'da recompute et
# Bellek: -40%, Süre: +20%
```

### 10.5 Evaluation Metrics

```python
from datasets import load_metric
import numpy as np

# === PERPLEXITY ===
def compute_perplexity(model, eval_dataloader):
    """
    Compute perplexity on evaluation set.
    Lower is better.
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    
    return perplexity

# === ROUGE (Summarization) ===
rouge = load_metric("rouge")

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores for summarization.
    """
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    
    return {
        "rouge1": results["rouge1"].mid.fmeasure,
        "rouge2": results["rouge2"].mid.fmeasure,
        "rougeL": results["rougeL"].mid.fmeasure,
    }

# === BLEU (Translation) ===
bleu = load_metric("sacrebleu")

def compute_bleu(predictions, references):
    """
    Compute BLEU score for translation.
    """
    results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )
    
    return results["score"]

# === CUSTOM METRICS ===
def compute_metrics(eval_pred):
    """
    Custom metrics function for Trainer.
    """
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE
    rouge_scores = compute_rouge(decoded_preds, decoded_labels)
    
    # Compute accuracy (exact match)
    exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    accuracy = exact_match / len(decoded_preds)
    
    return {
        **rouge_scores,
        "accuracy": accuracy,
    }

# Trainer ile kullanım
trainer = SFTTrainer(
    # ...
    compute_metrics=compute_metrics,
)
```

### 10.6 Common Issues and Solutions

```python
# === PROBLEM 1: OOM (Out of Memory) ===
# Çözüm 1: Batch size azalt
training_args.per_device_train_batch_size = 1

# Çözüm 2: Gradient accumulation artır
training_args.gradient_accumulation_steps = 16

# Çözüm 3: Gradient checkpointing aktif et
model.gradient_checkpointing_enable()

# Çözüm 4: QLoRA kullan
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# Çözüm 5: Max sequence length azalt
max_seq_length = 256  # 512 yerine

# === PROBLEM 2: Loss Divergence ===
# Çözüm 1: Learning rate azalt
training_args.learning_rate = 5e-5  # 2e-4 yerine

# Çözüm 2: Warmup artır
training_args.warmup_steps = 500

# Çözüm 3: Gradient clipping ekle
training_args.max_grad_norm = 0.3  # 1.0 yerine

# Çözüm 4: LoRA alpha azalt
lora_config.lora_alpha = 8  # 16 yerine

# === PROBLEM 3: Slow Training ===
# Çözüm 1: Flash Attention kullan
model = AutoModelForCausalLM.from_pretrained(
    ..., attn_implementation="flash_attention_2"
)

# Çözüm 2: TF32 aktif et (Ampere GPU)
torch.backends.cuda.matmul.allow_tf32 = True

# Çözüm 3: Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Çözüm 4: Dataloader workers artır
training_args.dataloader_num_workers = 4

# === PROBLEM 4: Poor Generation Quality ===
# Çözüm 1: LoRA rank artır
lora_config.r = 16  # 8 yerine

# Çözüm 2: Daha fazla target module
lora_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Çözüm 3: Training epoch artır
training_args.num_train_epochs = 5  # 3 yerine

# Çözüm 4: Data quality kontrol et
# - Duplikat veri temizle
# - Düşük kaliteli örnekleri çıkar
# - Data augmentation uygula
```

### 10.7 Model Merging Strategies

```python
# === STRATEJİ 1: Simple Merge ===
merged_model = peft_model.merge_and_unload()
# En basit, en hızlı

# === STRATEJİ 2: Weighted Merge ===
# Birden fazla adapter'ı ağırlıklı birleştir
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2", "adapter3"],
    weights=[0.5, 0.3, 0.2],
    adapter_name="merged_adapter",
    combination_type="linear",
)

# === STRATEJİ 3: SVD Merge ===
# SVD ile low-rank approximation
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2"],
    weights=[0.6, 0.4],
    adapter_name="svd_merged",
    combination_type="svd",
    svd_rank=8,  # Yeni rank
)

# === STRATEJİ 4: Task Arithmetic ===
# Model soup: Çoklu adapter'ları optimize et
from peft import TaskArithmeticMerger

merger = TaskArithmeticMerger()
merged_weights = merger.merge(
    adapters=[adapter1, adapter2, adapter3],
    weights=[0.4, 0.3, 0.3],
)
```

### 10.8 Continual Learning with LoRA

```python
# Senaryo: Model'i yeni task'lara adapt et, eski bilgiyi unutma

# === APPROACH 1: Sequential Adapter Training ===
# Task 1
lora_config_task1 = LoraConfig(r=8, ...)
model_task1 = get_peft_model(base_model, lora_config_task1)
# Train on Task 1
trainer.train()
model_task1.save_pretrained("./adapter-task1")

# Task 2 (yeni adapter, base model frozen)
model = PeftModel.from_pretrained(base_model, "./adapter-task1")
lora_config_task2 = LoraConfig(r=8, ...)
model.add_adapter(lora_config_task2, adapter_name="task2")
model.set_adapter("task2")
# Train on Task 2
trainer.train()
model.save_pretrained("./adapter-task2")

# Inference: Task'a göre adapter seç
model.set_adapter("task1")  # Task 1 için
model.set_adapter("task2")  # Task 2 için

# === APPROACH 2: Elastic Weight Consolidation (EWC) ===
# Fisher information ile önemli weights'leri koru
from peft import EWCCallback

ewc_callback = EWCCallback(
    lambda_ewc=0.4,  # EWC loss weight
    num_samples=1000,  # Fisher hesabı için sample sayısı
)

trainer = SFTTrainer(
    # ...
    callbacks=[ewc_callback],
)

# === APPROACH 3: Progressive LoRA ===
# Her task için rank'i artır
# Task 1: r=4
# Task 2: r=8
# Task 3: r=16
```

### 10.9 Interpretability: LoRA Weights Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_lora_weights(model):
    """
    Analyze LoRA weight matrices to understand what model learned.
    """
    # LoRA weights'leri topla
    lora_weights = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            # A ve B matrislerini al
            A = module.lora_A.default.weight.detach().cpu()
            B = module.lora_B.default.weight.detach().cpu()
            
            # BA product
            BA = B @ A
            
            lora_weights[name] = {
                'A': A,
                'B': B,
                'BA': BA,
            }
    
    return lora_weights

def visualize_lora_magnitude(lora_weights):
    """
    Visualize magnitude of LoRA updates across layers.
    """
    magnitudes = []
    layer_names = []
    
    for name, weights in lora_weights.items():
        BA = weights['BA']
        magnitude = torch.norm(BA).item()
        magnitudes.append(magnitude)
        layer_names.append(name.split('.')[-2])  # Layer name
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(magnitudes)), magnitudes)
    plt.xticks(range(len(magnitudes)), layer_names, rotation=45)
    plt.xlabel('Layer')
    plt.ylabel('LoRA Update Magnitude')
    plt.title('Magnitude of LoRA Updates Across Layers')
    plt.tight_layout()
    plt.show()

def visualize_singular_values(lora_weights):
    """
    Visualize singular values of BA products.
    Shows effective rank of LoRA updates.
    """
    for name, weights in lora_weights.items():
        BA = weights['BA']
        
        # SVD
        U, S, V = torch.svd(BA)
        
        plt.figure(figsize=(8, 5))
        plt.plot(S.numpy(), 'o-')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Magnitude')
        plt.title(f'Singular Values: {name}')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
        
        # Effective rank (95% variance explained)
        cumsum = torch.cumsum(S, dim=0)
        total = cumsum[-1]
        effective_rank = torch.sum(cumsum < 0.95 * total).item()
        print(f"{name}: Effective rank = {effective_rank} / {len(S)}")

# Kullanım
weights = analyze_lora_weights(peft_model)
visualize_lora_magnitude(weights)
visualize_singular_values(weights)
```

### 10.10 Deployment Best Practices

```python
# === PRACTICE 1: Model Optimization ===
# 1. Merge adapter (inference latency düşür)
merged_model = peft_model.merge_and_unload()

# 2. Convert to FP16 or BF16
merged_model = merged_model.half()  # FP16

# 3. Export to ONNX (optional)
import torch.onnx
dummy_input = torch.randint(0, 1000, (1, 512)).to(model.device)
torch.onnx.export(
    merged_model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
)

# === PRACTICE 2: Model Serving ===
# Seçenek A: TorchServe
# - Production-grade serving
# - Auto-scaling, metrics, logging

# Seçenek B: TensorRT-LLM (NVIDIA)
# - En hızlı inference (CUDA optimizations)
# - Karmaşık setup

# Seçenek C: vLLM
# - PagedAttention, continuous batching
# - High throughput

# Seçenek D: Text Generation Inference (HuggingFace)
from text_generation import Client

client = Client("http://localhost:8080")
response = client.generate("Hello, world!", max_new_tokens=100)

# === PRACTICE 3: Monitoring ===
import time
import psutil

class InferenceMonitor:
    def __init__(self):
        self.latencies = []
        self.throughputs = []
    
    def monitor_inference(self, model, inputs, num_runs=100):
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model.generate(**inputs)
            
            latency = time.time() - start_time
            self.latencies.append(latency)
        
        # Compute stats
        avg_latency = np.mean(self.latencies)
        p50_latency = np.percentile(self.latencies, 50)
        p95_latency = np.percentile(self.latencies, 95)
        p99_latency = np.percentile(self.latencies, 99)
        
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"P50 latency: {p50_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")
        print(f"P99 latency: {p99_latency:.3f}s")
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak GPU memory: {gpu_mem:.2f} GB")

# === PRACTICE 4: A/B Testing ===
# Base model vs LoRA fine-tuned model
def ab_test(model_a, model_b, test_prompts):
    results = {"model_a": [], "model_b": []}
    
    for prompt in test_prompts:
        # Generate from both models
        response_a = generate_response(model_a, prompt)
        response_b = generate_response(model_b, prompt)
        
        # Collect user preference (in production: real users)
        # Here: simulated with heuristics
        score_a = compute_quality_score(response_a)
        score_b = compute_quality_score(response_b)
        
        results["model_a"].append(score_a)
        results["model_b"].append(score_b)
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(
        results["model_a"],
        results["model_b"]
    )
    
    print(f"Model A avg score: {np.mean(results['model_a']):.3f}")
    print(f"Model B avg score: {np.mean(results['model_b']):.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Statistically significant difference!")
```

---

## 11. Özet ve Sonuç

### 11.1 LoRA Avantajları

✅ **Parametre Verimliliği:** %99.9 daha az trainable parametre
✅ **Bellek Tasarrufu:** %60-75 daha az GPU bellek
✅ **Hızlı Training:** Daha az parametre → daha hızlı convergence
✅ **Modülerlik:** Birden fazla adapter, tek base model
✅ **Katastrofik Unutma Yok:** Base model frozen → bilgi korunur
✅ **Kolay Deployment:** Adapter swap, multi-task serving

### 11.2 LoRA Dezavantajları

❌ **Inference Overhead:** %5-10 latency (merge edilmezse)
❌ **Limitli Expressiveness:** Full fine-tuning'den daha az güçlü
❌ **Yeni Bilgi Öğrenme:** Domain shift'te zorlanabilir
❌ **Hyperparameter Sensitivity:** r, alpha seçimi kritik

### 11.3 Ne Zaman LoRA Kullanmalı?

**LoRA İdeal:**

- ✅ Limited GPU resources (8-16 GB)
- ✅ Instruction-following, alignment tasks
- ✅ Domain adaptation (benzer domain)
- ✅ Multi-task scenarios
- ✅ Rapid prototyping

**Full Fine-Tuning Tercih Et:**

- ❌ Completely new domain (code → medical)
- ❌ Sufficient resources available (100+ GB GPU)
- ❌ Maximum accuracy required
- ❌ Single task, no adapter switching needed

### 11.4 Quick Reference

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

### 11.5 İleri Okuma ve Kaynaklar

**Orijinal Papers:**

- LoRA: <https://arxiv.org/abs/2106.09685>
- QLoRA: <https://arxiv.org/abs/2305.14314>
- Prefix Tuning: <https://arxiv.org/abs/2101.00190>

**Hugging Face Docs:**

- PEFT: <https://huggingface.co/docs/peft>
- Transformers: <https://huggingface.co/docs/transformers>

**Code Repositories:**

- PEFT: <https://github.com/huggingface/peft>
- TRL: <https://github.com/huggingface/trl>
- BitsAndBytes: <https://github.com/TimDettmers/bitsandbytes>

**Blog Posts:**

- <https://huggingface.co/blog/peft>
- <https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms>

---

## Son Notlar

Bu rehber, LoRA'yı sıfırdan öğrenmeniz için kapsamlı bir kaynak sunmaktadır. Teorik temeller, pratik kod örnekleri, production best practices ve troubleshooting tüm adımlarıyla anlatılmıştır.

**Önerilen Öğrenme Yolu:**

1. Bölüm 1-3: Teori ve setup
2. Bölüm 4-6: Veri hazırlama ve config
3. Bölüm 7: Kendi dataset'inizle train edin
4. Bölüm 8-9: Inference ve deployment
5. Bölüm 10: Optimize edin ve production'a alın

**Başarılar! Happy Fine-Tuning! 🚀**
