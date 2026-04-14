"""
PhysLM training scaffold for Phase 4.

This script reuses the proven Parameter Golf training infrastructure while swapping
the model body to a physics-inspired hybrid:
- 12 Hamiltonian SSM blocks with chunked scan
- 2 causal attention "measurement" blocks
- bigram hash input prior
- Boltzmann-style output head with Zipf prior bias

Slimmed default budget target:
- tok_emb (tied): ~524K
- BigramHash: ~328K
- 12x SSM blocks: ~20.1M
- 2x attention blocks with GQA: ~3.67M
- skip weights + Boltzmann output: ~4K
- total: ~24.6M params, targeting <16MB after int6-class compression
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from train_eft import (
    BigramHashEmbedding,
    CONTROL_TENSOR_NAME_PATTERNS,
    CastedLinear,
    CausalSelfAttention,
    DistributedTokenLoader,
    Muon,
    RMSNorm,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    quantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
)


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"physlm_{uuid.uuid4()}")
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_ssm_layers = int(os.environ.get("NUM_SSM_LAYERS", 12))
    num_attn_layers = int(os.environ.get("NUM_ATTN_LAYERS", 2))
    state_dim = int(os.environ.get("STATE_DIM", 64))
    mlp_mult = float(os.environ.get("MLP_MULT", 2.0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    zipf_alpha = float(os.environ.get("ZIPF_ALPHA", 1.37))

    embed_lr = float(os.environ.get("EMBED_LR", 0.035))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    ssm_lr = float(os.environ.get("SSM_LR", 0.001))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32_768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "0")))


class CastedLinear(nn.Linear):
    _qat_phase = "off"
    _qat_alpha = 0.0
    _qat_enabled = False

    def __init__(self, in_features: int, out_features: int, bias: bool = False, quant_bits: int = 6):
        super().__init__(in_features, out_features, bias=bias)
        self.quant_bits = quant_bits

    @classmethod
    def set_qat_state(cls, progress: float) -> tuple[str, float]:
        if progress < 0.5:
            cls._qat_phase = "off"
            cls._qat_alpha = 0.0
        elif progress < 0.8:
            cls._qat_phase = "soft"
            cls._qat_alpha = (progress - 0.5) / 0.3
        else:
            cls._qat_phase = "hard"
            cls._qat_alpha = 1.0
        cls._qat_enabled = cls._qat_phase != "off"
        return cls._qat_phase, cls._qat_alpha

    def _fake_quantize_weight(self, weight: Tensor) -> Tensor:
        qmax = (1 << (self.quant_bits - 1)) - 1
        if weight.ndim == 2:
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        else:
            scale = weight.abs().amax().clamp_min(1e-8) / qmax
        clipped = torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale
        ste = weight + (clipped - weight).detach()
        if self._qat_phase == "soft":
            return torch.lerp(weight, ste, self._qat_alpha)
        if self._qat_phase == "hard":
            return ste
        return weight

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.float()
        if self.training and self.quant_bits > 0 and self._qat_enabled:
            weight = self._fake_quantize_weight(weight)
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), weight, bias).to(dtype=x.dtype)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def repeat_kv(x: Tensor, num_heads: int) -> Tensor:
    bsz, kv_heads, seqlen, head_dim = x.shape
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={kv_heads}")
    repeat_factor = num_heads // kv_heads
    return (
        x[:, :, None, :, :]
        .expand(bsz, kv_heads, repeat_factor, seqlen, head_dim)
        .reshape(bsz, num_heads, seqlen, head_dim)
    )


class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, embed_dim: int, output_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, output_dim, bias=False, quant_bits=6)
        self.proj._zero_init = True

    def forward(self, input_ids: Tensor) -> Tensor:
        prev_tokens = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
        bigram_ids = (prev_tokens.long() * 1_000_003 + input_ids.long()) % self.num_buckets
        return self.proj(self.embedding(bigram_ids))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim != num_heads * head_dim:
            raise ValueError("dim must equal num_heads * head_dim")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = CastedLinear(dim, dim, bias=False, quant_bits=8)
        self.k_proj = CastedLinear(dim, kv_dim, bias=False, quant_bits=8)
        self.v_proj = CastedLinear(dim, kv_dim, bias=False, quant_bits=8)
        self.out_proj = CastedLinear(dim, dim, bias=False, quant_bits=8)
        self.out_proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        k = repeat_kv(k, self.num_heads)
        v = repeat_kv(v, self.num_heads)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.out_proj(y)


class LeakySquaredMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up = CastedLinear(dim, hidden_dim, bias=False)
        self.down = CastedLinear(hidden_dim, dim, bias=False)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.up(x), negative_slope=0.5)
        return self.down(x.square())


@torch.jit.script
def _sequential_hamiltonian_scan(phase: Tensor, drive: Tensor, decay: Tensor) -> tuple[Tensor, Tensor]:
    """Dissipative Hamiltonian scan: h_t = decay * exp(-i*phase) * h_{t-1} + drive_t"""
    batch = phase.size(0)
    seqlen = phase.size(1)
    state_dim = phase.size(2)
    out_real = torch.empty_like(phase)
    out_imag = torch.empty_like(phase)
    state_real = torch.zeros(batch, state_dim, device=phase.device, dtype=phase.dtype)
    state_imag = torch.zeros(batch, state_dim, device=phase.device, dtype=phase.dtype)
    for t in range(seqlen):
        d = decay[:, t, :]
        ar = d * torch.cos(phase[:, t, :])
        ai = d * torch.sin(phase[:, t, :])
        new_real = ar * state_real + ai * state_imag + drive[:, t, :]
        new_imag = -ai * state_real + ar * state_imag
        state_real = new_real
        state_imag = new_imag
        out_real[:, t, :] = state_real
        out_imag[:, t, :] = state_imag
    return out_real, out_imag


def _chunked_hamiltonian_scan_impl(phase: Tensor, drive: Tensor, decay: Tensor, chunk_size: int) -> tuple[Tensor, Tensor]:
    """Dissipative chunked scan: A_t = decay_t * exp(-i*phase_t), with |A_t| < 1."""
    batch, seqlen, state_dim = phase.shape
    pad_tokens = (-seqlen) % chunk_size
    if pad_tokens > 0:
        phase = F.pad(phase, (0, 0, 0, pad_tokens))
        drive = F.pad(drive, (0, 0, 0, pad_tokens))
        decay = F.pad(decay, (0, 0, 0, pad_tokens), value=1.0)
    padded_len = phase.size(1)
    num_chunks = padded_len // chunk_size
    phase_chunks = phase.reshape(batch, num_chunks, chunk_size, state_dim)
    drive_chunks = drive.reshape(batch, num_chunks, chunk_size, state_dim)
    decay_chunks = decay.reshape(batch, num_chunks, chunk_size, state_dim)

    state_real = torch.zeros(batch, state_dim, device=phase.device, dtype=phase.dtype)
    state_imag = torch.zeros_like(state_real)
    out_real = torch.empty(batch, num_chunks, chunk_size, state_dim, device=phase.device, dtype=phase.dtype)
    out_imag = torch.empty_like(out_real)

    for chunk_idx in range(num_chunks):
        chunk_phase = phase_chunks[:, chunk_idx]
        chunk_drive = drive_chunks[:, chunk_idx]
        chunk_decay = decay_chunks[:, chunk_idx]

        # Log-space cumulative product for dissipative evolution:
        # log|A_t| = log(decay_t), angle(A_t) = -phase_t
        log_decay_prefix = torch.cumsum(torch.log(chunk_decay + 1e-8), dim=1)
        prefix_mag = torch.exp(log_decay_prefix)
        prefix_phase = torch.cumsum(chunk_phase, dim=1)
        prefix_real = prefix_mag * torch.cos(prefix_phase)
        prefix_imag = -prefix_mag * torch.sin(prefix_phase)

        # Inverse of prefix for transforming drive into "unwound" space
        inv_mag = torch.exp(-log_decay_prefix)
        inv_prefix_real = inv_mag * torch.cos(prefix_phase)
        inv_prefix_imag = inv_mag * torch.sin(prefix_phase)

        transformed_real = chunk_drive * inv_prefix_real
        transformed_imag = chunk_drive * inv_prefix_imag
        accum_real = torch.cumsum(transformed_real, dim=1)
        accum_imag = torch.cumsum(transformed_imag, dim=1)

        # Carry boundary state scaled by chunk's total decay
        total_decay = prefix_mag[:, -1:, :]
        carry_real = state_real[:, None, :] + accum_real
        carry_imag = state_imag[:, None, :] + accum_imag
        chunk_out_real = prefix_real * carry_real - prefix_imag * carry_imag
        chunk_out_imag = prefix_imag * carry_real + prefix_real * carry_imag

        out_real[:, chunk_idx] = chunk_out_real
        out_imag[:, chunk_idx] = chunk_out_imag
        # Boundary state for next chunk is in the "unwound" coordinate of last step
        state_real = carry_real[:, -1, :]
        state_imag = carry_imag[:, -1, :]

    out_real = out_real.reshape(batch, padded_len, state_dim)[:, :seqlen, :]
    out_imag = out_imag.reshape(batch, padded_len, state_dim)[:, :seqlen, :]
    return out_real, out_imag


class HamiltonianSSM(nn.Module):
    def __init__(self, dim: int, state_dim: int, chunk_size: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.input_gate = CastedLinear(dim, dim, bias=True)
        self.dt_proj = CastedLinear(dim, state_dim, bias=True)
        self.b_proj = CastedLinear(dim, state_dim, bias=False)
        self.c_proj = CastedLinear(state_dim, dim, bias=False)
        self.output_gate = CastedLinear(dim, dim, bias=True)
        self.direct_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.omega = nn.Parameter(torch.logspace(-2.0, 0.3, state_dim, dtype=torch.float16))
        # Learnable damping rate (Lindbladian dissipation): initialized so decay ≈ 0.95/step
        self.log_gamma = nn.Parameter(torch.full((state_dim,), -3.0, dtype=torch.float32))
        nn.init.zeros_(self.input_gate.bias)
        nn.init.constant_(self.output_gate.bias, -1.0)
        # _sequential_hamiltonian_scan is @torch.jit.script — no compile wrapper needed

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        gate = torch.sigmoid(self.input_gate(x_fp32))
        u = gate * x_fp32
        dt = F.softplus(self.dt_proj(u)).clamp(max=5.0) + 1e-3
        drive = torch.tanh(self.b_proj(u)) / math.sqrt(self.state_dim)
        omega = self.omega.float().abs()[None, None, :]
        phase = (dt * omega).clamp(max=50.0)
        gamma = F.softplus(self.log_gamma.float())[None, None, :]
        decay = torch.exp(-gamma * dt).clamp(min=0.5, max=0.9999)
        stacked_states, _ = _sequential_hamiltonian_scan(phase, drive, decay)
        y = self.c_proj(stacked_states)
        y = torch.sigmoid(self.output_gate(x_fp32)) * y
        direct = self.direct_scale[None, None, :] * gate * x_fp32
        return (y + direct).to(dtype=x.dtype)


class PhysLMBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, mlp_mult: float):
        super().__init__()
        hidden_dim = int(dim * mlp_mult)
        self.ssm_norm = RMSNorm()
        self.ssm = HamiltonianSSM(dim, state_dim)
        self.ssm_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_norm = RMSNorm()
        self.mlp = LeakySquaredMLP(dim, hidden_dim)
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.ssm_scale.to(dtype=x.dtype)[None, None, :] * self.ssm(self.ssm_norm(mixed))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class AttentionMeasurementBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_mult: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must divide num_heads")
        hidden_dim = int(dim * mlp_mult)
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=dim // num_heads,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_norm = RMSNorm()
        self.mlp = LeakySquaredMLP(dim, hidden_dim)
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(mixed))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class BoltzmannOutput(nn.Module):
    def __init__(self, vocab_size: int, dim: int, tie_embeddings: bool, zipf_alpha: float, logit_softcap: float):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
        self.self_energy = nn.Parameter(-zipf_alpha * torch.log(ranks))
        self.proj = None if tie_embeddings else CastedLinear(dim, vocab_size, bias=False, quant_bits=6)

    def forward(self, x: Tensor, tied_weight: Tensor | None) -> Tensor:
        if self.tie_embeddings:
            if tied_weight is None:
                raise ValueError("tied_weight is required when tie_embeddings=True")
            logits = F.linear(x, tied_weight.to(dtype=x.dtype))
        else:
            logits = self.proj(x)
        logits = logits + self.self_energy.to(dtype=x.dtype)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits


class PhysLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        num_ssm_layers: int,
        num_attn_layers: int,
        state_dim: int,
        mlp_mult: float,
        bigram_vocab_size: int,
        bigram_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        rope_base: float,
        qk_gain_init: float,
        logit_softcap: float,
        zipf_alpha: float,
    ):
        super().__init__()
        if num_ssm_layers % 2 != 0:
            raise ValueError("num_ssm_layers must be even for symmetric U-Net skips")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
        self.input_norm = RMSNorm()
        self.ssm_blocks = nn.ModuleList(
            [PhysLMBlock(model_dim, state_dim, mlp_mult) for _ in range(num_ssm_layers)]
        )
        self.skip_weights = nn.Parameter(torch.zeros(num_ssm_layers // 2, model_dim, dtype=torch.float32))
        self.attn_blocks = nn.ModuleList(
            [
                AttentionMeasurementBlock(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init, mlp_mult)
                for _ in range(num_attn_layers)
            ]
        )
        self.output = BoltzmannOutput(vocab_size, model_dim, tie_embeddings, zipf_alpha, logit_softcap)
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward_features(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids).to(dtype=x.dtype)
        x = self.input_norm(x)
        x0 = x
        half = len(self.ssm_blocks) // 2
        skip_bank: list[Tensor] = []
        for idx, block in enumerate(self.ssm_blocks):
            x = block(x, x0)
            if idx < half:
                skip_bank.append(x)
            else:
                skip_idx = len(self.ssm_blocks) - 1 - idx
                x = x + self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skip_bank[skip_idx]
        for block in self.attn_blocks:
            x = block(x, x0)
        return x

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        hidden = self.forward_features(input_ids)
        return self.output(hidden, self.tok_emb.weight if self.tie_embeddings else None)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        logits = self.forward_logits(input_ids)
        if target_ids is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")


def eval_val_sliding(
    args: Hyperparameters,
    base_model: PhysLM,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= 1
    ]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: PhysLM,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]

    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(
        f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
        f"total_windows={len(window_starts)} stride={stride} "
        f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
        f"freeze_blocks={args.ttt_freeze_blocks}"
    )

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.ssm_blocks) + len(base_model.attn_blocks))))
    ttt_params: list[nn.Parameter] = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"ssm_blocks.{bi}." in name or f"attn_blocks.{bi - len(base_model.ssm_blocks)}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(
        f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
        f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}"
    )

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci + 1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(
        f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )
    return val_loss, val_bpb


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    default_grad_accum_steps = 8 // world_size
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps > 0 else default_grad_accum_steps
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(sys.platform == "win32")

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = PhysLM(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_ssm_layers=args.num_ssm_layers,
        num_attn_layers=args.num_attn_layers,
        state_dim=args.state_dim,
        mlp_mult=args.mlp_mult,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        logit_softcap=args.logit_softcap,
        zipf_alpha=args.zipf_alpha,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name.endswith("omega"):
                param.data = param.data.to(dtype=torch.float16)
    if args.compile_model and sys.platform != "win32":
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    else:
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    embed_params = [base_model.tok_emb.weight, base_model.bigram.embedding.weight]
    head_named_params = [(f"output.{name}", p) for name, p in base_model.output.named_parameters()]
    head_param_names = {name for name, _ in head_named_params}
    body_named_params = [
        (name, p)
        for name, p in base_model.named_parameters()
        if name not in {"tok_emb.weight", "bigram.embedding.weight"} and name not in head_param_names
    ]
    matrix_params = [
        p
        for name, p in body_named_params
        if p.ndim >= 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in body_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    head_params = [p for _, p in head_named_params]

    optimizer_embed = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_head = torch.optim.Adam(
        [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_embed, optimizer_head, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"model_kind:physlm ssm_layers:{args.num_ssm_layers} attn_layers:{args.num_attn_layers} "
        f"model_dim:{args.model_dim} state_dim:{args.state_dim} "
        f"num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{args.embed_lr} head_lr:{args.head_lr} "
        f"matrix_lr:{args.matrix_lr} ssm_lr:{args.ssm_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f} ema_enabled:{args.ema_enabled} "
        f"swa_enabled:{args.swa_enabled} eval_stride:{args.eval_stride} ttt_enabled:{args.ttt_enabled}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    training_time_ms = 0.0
    stop_after_step: int | None = None
    qat_phase, qat_alpha = CastedLinear.set_qat_state(0.0)
    log0(f"qat:phase={qat_phase} alpha={qat_alpha:.3f}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        progress = step / max(args.iterations, 1)
        next_qat_phase, next_qat_alpha = CastedLinear.set_qat_state(progress)
        if next_qat_phase != qat_phase or abs(next_qat_alpha - qat_alpha) >= 0.05:
            qat_phase, qat_alpha = next_qat_phase, next_qat_alpha
            if qat_phase != "soft" or step % max(args.train_log_every, 1) == 0:
                log0(f"qat:phase={qat_phase} alpha={qat_alpha:.3f} step:{step}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        should_log_train = args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if ema_state is not None:
        log0("ema:applying EMA weights")
        avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    elif args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        avg_state = {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype) for name, t in swa_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if args.eval_stride > 0:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
