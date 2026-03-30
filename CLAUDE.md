# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI's **Parameter Golf** challenge: train the best language model that fits in a 16MB artifact (code + compressed model) and trains in under 10 minutes on 8xH100s. The metric is **val_bpb** (bits per byte) on the FineWeb validation set, evaluated tokenizer-agnostically. Lower is better. Adapted from modded-nanogpt.

## Key Commands

### Data Download
```bash
# Download cached FineWeb with 1024-token SentencePiece vocab (default)
python3 data/cached_challenge_fineweb.py --variant sp1024
# Smaller subset for local iteration
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training (CUDA, single GPU)
```bash
RUN_ID=my_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Training (CUDA, 8xH100 for leaderboard)
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training (MLX on Apple Silicon)
```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

### Useful Environment Variable Overrides
- `MAX_WALLCLOCK_SECONDS=600` (default; set to `0` for unlimited)
- `VAL_LOSS_EVERY=200` for periodic validation
- `TRAIN_LOG_EVERY=200` for train loss logging
- `ITERATIONS=20000` (default step count)
- `SEED=1337` (default)

## Architecture

### Two Training Scripts
- **`train_gpt.py`** (~1126 lines) — CUDA/PyTorch, distributed via `torchrun`+DDP. This is the submission script.
- **`train_gpt_mlx.py`** (~1104 lines) — Apple Silicon/MLX port for local iteration. Same model architecture, different backend.

Both scripts are starter baselines (<1500 lines hard limit), not SOTA configs. PRs improving these are accepted, but competitive submissions go in `/records`.

### Model (GPT class in train_gpt.py)
- Transformer with **U-Net style skip connections**: first half (encoder) stores skip activations, second half (decoder) consumes them in reverse order via learnable `skip_weights`.
- **GQA** (Grouped Query Attention): 8 query heads, 4 KV heads by default.
- **RoPE** positional encoding with configurable base frequency.
- **ReLU^2 MLP** (squared relu activation from modded-nanogpt).
- **Tied embeddings** by default (input embedding reused as output projection).
- **Logit softcapping** (tanh-based, default 30.0).
- Per-block learnable `attn_scale`, `mlp_scale`, and `resid_mix` (residual stream mixing with initial embedding `x0`).
- **CastedLinear**: weights stored in fp32 for optimizer quality, cast to bf16 at compute time.

### Optimizer Setup
Three optimizer groups with separate learning rates:
1. **Muon** (Newton-Schulz orthogonalization) for 2D matrix params in transformer blocks — `MATRIX_LR=0.04`
2. **Adam** for scalar/vector params and skip weights — `SCALAR_LR=0.04`
3. **Adam** for token embeddings — `TIED_EMBED_LR=0.05` (or `EMBED_LR`/`HEAD_LR` when untied)

Muon momentum warms up from 0.85 to 0.95 over 500 steps. LR warmdown is wallclock-aware (scales down as training approaches the 10-minute cap).

### Post-Training Quantization & Serialization
After training, the model is quantized to **int8** (per-row scales for 2D tensors, per-tensor for vectors) and compressed with **zlib**. The round-tripped quantized model is re-evaluated to produce the final `val_bpb` score. Small tensors (<65536 elements) and control tensors are kept as fp16.

Output files: `final_model.pt` (raw), `final_model.int8.ptz` (compressed artifact).

### Evaluation (BPB Calculation)
- **val_bpb** = bits_per_token * tokens_per_byte — tokenizer-agnostic compression metric.
- Validation runs on the fixed first-50k-document FineWeb validation split.
- SentencePiece LUTs compute per-token byte counts, handling leading-space merges and boundary tokens.
- Evaluation can use any sequence length and sliding window strategies.

## Data Layout
```
data/
  datasets/fineweb10B_sp1024/    # Binary shards: fineweb_train_*.bin, fineweb_val_*.bin
  tokenizers/fineweb_1024_bpe.model  # SentencePiece model
  cached_challenge_fineweb.py    # HuggingFace downloader (manifest-driven)
```

Shards use a custom binary format: 256-int32 header (magic=20240520, version=1, num_tokens) followed by uint16 token IDs.

## Submission Structure
Each submission is a PR adding a folder under `records/track_10min_16mb/` or `records/track_non_record_16mb/`:
```
records/track_10min_16mb/YYYY-MM-DD_DescriptiveName/
  README.md          # Explains the approach
  submission.json    # Name, GitHub ID, val_bpb, metadata
  train.log          # Training log (3+ runs for statistical significance)
  train_gpt.py       # Must compile and run within the records folder
```

### Leaderboard Rules
- New SOTA must beat existing by **>= 0.005 nats** at **p < 0.01**.
- Total artifact (code + compressed model) must be **<= 16,000,000 bytes** (decimal, not MiB).
- Training must complete in **<= 10 minutes on 8xH100 SXM**.
- No network access during evaluation. No training on the validation set.
- Test-time training is allowed only on validation tokens already graded.

---

# Role: Plan Agent

当用户给出需求时，你的职责是分析并输出结构化执行计划。

## 输出规范
- 计划写入 `.agent/plan.md`（如不存在则创建，`.agent/` 目录同理）
- 如果 `.agent/plan.md` 已有内容，将新计划追加到末尾（用 `---` 分隔），不覆盖已有内容
- 每个子任务用 `## Task N: 标题` 格式
- 每个 task 包含：目标、要修改的文件列表、具体改动描述、依赖关系
- 无依赖的任务标记 `parallel: true`

## Review 规范
- 确认已完成的任务追加到 `.agent/plan_done.md` 归档
- 从 `.agent/plan.md` 中删除已归档的任务内容

## 约束
- 不要自己执行代码修改，只输出计划
- 保持任务粒度适中（每个 task 对应 1-3 个文件）
- 标注风险点和需要 review 的关键逻辑
