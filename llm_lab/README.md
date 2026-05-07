# lab_env — Reusable LLM Lab Environment

Layout:
```
lab_env/
├── pyproject.toml          # uv project (transformers, trl, unsloth, peft, jupyterlab)
└── notebooks/
    ├── 01_sft_modern.ipynb     # 2026 version of ch07 (Unsloth + TRL SFTTrainer)
    └── ref/                    # reference notebooks (read-only)
        ├── pre-training-evaluation.ipynb
        ├── model_analysis.ipynb
        └── tokenizer.ipynb
```

## Setup

Copy from local to server:
```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' lab_env/ vast-ai:/root/lab_env/
```

On the server:
```bash
ssh vast-ai
cd /root/lab_env
uv sync                                 # download packages + create venv
uv run python -c "import torch, unsloth; print('OK')"
```

## JupyterLab

```bash
cd /root/lab_env
uv run jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root
```

If your SSH config has `LocalForward 8080 localhost:8080`: `http://localhost:8080`

## Notebook Flow

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_sft_modern.ipynb` | SFT — turn Qwen3-0.6B-Base into an instruction-following model |
| ref | `pre-training-evaluation.ipynb` | 18-section evaluation suite (measure the base model) |
| ref | `model_analysis.ipynb` | Architecture / parameter / weight analysis |
| ref | `tokenizer.ipynb` | Tokenizer fertility (critical for Turkish) |

## Critical Notes (TRL v1.3 + Unsloth 2025.7)

- **`import unsloth` MUST come first** — otherwise the optimizations don't get applied
- **`assistant_only_loss=True`** — the chat template has to contain the `{% generation %}` keywords (TRL patches this automatically for Qwen3)
- **`use_gradient_checkpointing='unsloth'`** — for long context + low VRAM
- **`processing_class=tokenizer`** — `tokenizer=` is deprecated in TRL v1.3
- **`packing=True`** — 2-4x speed-up on short examples but may be incompatible with `assistant_only_loss`
- **`completion_only_loss`** — for prompt-completion format (not messages)

## For a New Server

1. Update the `vast-ai` host in `~/.ssh/config` (new IP/port)
2. `rsync` this folder over
3. `uv sync`
4. The notebooks are already inside

## Version Pins

| Package | Pin Reason |
|---------|------------|
| `transformers>=5.2` | Required for the Qwen3.5 model_type |
| `trl` | latest (1.3+) — Qwen3.5 support |
| `unsloth` | latest 2026.4+ — compatible with the new transformers |

xformers warnings are normal (Python version mismatch) — Unsloth uses its own kernels.
