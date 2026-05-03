# lab_env — Yeniden Kullanılabilir LLM Lab Ortamı

Yapı:
```
lab_env/
├── pyproject.toml          # uv proje (transformers, trl, unsloth, peft, jupyterlab)
└── notebooks/
    ├── 01_sft_modern.ipynb     # ch07'nin 2026 versiyonu (Unsloth + TRL SFTTrainer)
    └── ref/                    # referans notebook'lar (read-only)
        ├── pre-training-evaluation.ipynb
        ├── model_analysis.ipynb
        └── tokenizer.ipynb
```

## Kurulum

Yerelden server'a kopyala:
```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' lab_env/ vast-ai:/root/lab_env/
```

Server'da:
```bash
ssh vast-ai
cd /root/lab_env
uv sync                                 # paketleri indir + venv kur
uv run python -c "import torch, unsloth; print('OK')"
```

## JupyterLab

```bash
cd /root/lab_env
uv run jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root
```

SSH config'de `LocalForward 8080 localhost:8080` varsa: `http://localhost:8080`

## Notebook Akışı

| # | Notebook | Amaç |
|---|----------|------|
| 1 | `01_sft_modern.ipynb` | SFT — Qwen3-0.6B-Base'i instruction-following hale getir |
| ref | `pre-training-evaluation.ipynb` | 18-bölümlük evaluation (base modeli ölç) |
| ref | `model_analysis.ipynb` | Mimari/parametre/weight analizi |
| ref | `tokenizer.ipynb` | Tokenizer fertility (Türkçe için kritik) |

## Kritik Notlar (TRL v1.3 + Unsloth 2025.7)

- **`import unsloth` EN BAŞTA** — yoksa optimizasyonlar uygulanmaz
- **`assistant_only_loss=True`** — chat template `{% generation %}` keyword'lerine ihtiyaç duyar (Qwen3 için TRL otomatik patch yapar)
- **`use_gradient_checkpointing='unsloth'`** — uzun context + düşük VRAM için
- **`processing_class=tokenizer`** — TRL v1.3'te `tokenizer=` deprecated
- **`packing=True`** — kısa örneklerde 2-4x hız ama `assistant_only_loss` ile uyumsuz olabilir
- **`completion_only_loss`** prompt-completion format için (messages değil)

## Yeni Server İçin

1. `~/.ssh/config` `vast-ai` host güncelle (yeni IP/port)
2. `rsync` ile bu klasörü kopyala
3. `uv sync`
4. Notebook'lar zaten içinde

## Versiyon Pinleri

| Paket | Pin Sebebi |
|-------|-----------|
| `transformers>=5.2` | Qwen3.5 model_type için zorunlu |
| `trl` | latest (1.3+) — Qwen3.5 support |
| `unsloth` | latest 2026.4+ — yeni transformers ile uyumlu |

xformers warnings normal (Python sürüm uyumsuzluğu) — Unsloth kendi kernel'larını kullanır.
