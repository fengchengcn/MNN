"""Qwen3-ASR custom model — built from safetensors weights.

Qwen3-ASR is not in the standard transformers library. This module
loads the model directly from safetensors into a torch.nn.Module
structure that the MNN llmexport.py pipeline can consume.

Architecture:
  thinker                    — top-level wrapper
  ├── lm_head               — Linear(hidden_size, vocab_size, bias=False)
  ├── model.embed_tokens    — Embedding(vocab_size, hidden_size)
  ├── model.layers[0..N-1]  — N× Qwen3 decoder layers
  │   ├── input_layernorm          — RMSNorm(hidden_size)
  │   ├── self_attn.q_proj         — Linear(hidden_size, n_heads*head_dim, bias=False)
  │   ├── self_attn.k_proj         — Linear(hidden_size, n_kv*head_dim, bias=False)
  │   ├── self_attn.v_proj         — Linear(hidden_size, n_kv*head_dim, bias=False)
  │   ├── self_attn.o_proj         — Linear(n_heads*head_dim, hidden_size, bias=False)
  │   ├── self_attn.q_norm         — RMSNorm(head_dim)
  │   ├── self_attn.k_norm         — RMSNorm(head_dim)
  │   ├── post_attention_layernorm — RMSNorm(hidden_size)
  │   └── mlp
  │       ├── gate_proj  — Linear(intermediate_size, hidden_size, bias=False)
  │       ├── up_proj    — Linear(intermediate_size, hidden_size, bias=False)
  │       └── down_proj  — Linear(hidden_size, intermediate_size, bias=False)
  ├── model.norm           — RMSNorm(hidden_size)
  └── audio_tower          — Whisper-style audio encoder
      ├── conv2d1   — Conv2d(1, conv_hidden, 3, stride=2)
      ├── conv2d2   — Conv2d(conv_hidden, conv_hidden, 3, stride=2)
      ├── conv2d3   — Conv2d(conv_hidden, conv_hidden, 3, stride=2)
      ├── conv_out  — Linear(conv_hidden*16, d_model)
      ├── layers[0..M-1] — M× Transformer encoder layers
      │   ├── self_attn_layer_norm  — LayerNorm(d_model)
      │   ├── self_attn             — QKV attention (n_audio_heads heads, d_model//n_audio_heads dim)
      │   │   ├── q_proj, k_proj, v_proj, out_proj
      │   ├── fc1           — Linear(d_model, audio_ffn)
      │   ├── fc2           — Linear(audio_ffn, d_model)
      │   └── final_layer_norm — LayerNorm(d_model)
      ├── ln_post  — LayerNorm(d_model)
      ├── proj1    — Linear(d_model, d_model)
      └── proj2    — Linear(d_model, hidden_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from safetensors import safe_open


# ---------------------------------------------------------------------------
# RMSNorm — used by Qwen3 text decoder
# ---------------------------------------------------------------------------

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


# ---------------------------------------------------------------------------
# Text decoder layer (matches Qwen3 / LLaMA structure)
# ---------------------------------------------------------------------------

class Qwen3DecoderLayer(nn.Module):
    """Single transformer decoder layer with GQA + Q/K-Norm + SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        hs = config["hidden_size"]
        n_heads = config["num_attention_heads"]
        n_kv = config.get("num_key_value_heads", n_heads)
        head_dim = config.get("head_dim", hs // n_heads)
        ffn = config["intermediate_size"]

        # Pre-attention norm
        self.input_layernorm = QwenRMSNorm(hs)

        # Attention (GQA) — use plain Modules so getattr-based mapper works
        self.self_attn = _AttentionSubModule(hs, n_heads, n_kv, head_dim)

        # Post-attention norm
        self.post_attention_layernorm = QwenRMSNorm(hs)

        # SwiGLU MLP
        self.mlp = _MLPSubModule(hs, ffn)

    def load_weights(self, state_dict, prefix):
        """Copy weights from flat state dict under the given prefix."""
        self.input_layernorm.weight.data.copy_(state_dict[f"{prefix}.input_layernorm.weight"])
        self.post_attention_layernorm.weight.data.copy_(state_dict[f"{prefix}.post_attention_layernorm.weight"])
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(self.self_attn, name).weight.data.copy_(state_dict[f"{prefix}.self_attn.{name}.weight"])
        for name in ["q_norm", "k_norm"]:
            getattr(self.self_attn, name).weight.data.copy_(state_dict[f"{prefix}.self_attn.{name}.weight"])
        for name in ["gate_proj", "up_proj", "down_proj"]:
            getattr(self.mlp, name).weight.data.copy_(state_dict[f"{prefix}.mlp.{name}.weight"])


class _AttentionSubModule(nn.Module):
    """Plain Module with attribute-based submodules for GQA attention."""

    def __init__(self, hs, n_heads, n_kv, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hs, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hs, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hs, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hs, bias=False)
        self.q_norm = QwenRMSNorm(head_dim)
        self.k_norm = QwenRMSNorm(head_dim)


class _MLPSubModule(nn.Module):
    """Plain Module with attribute-based submodules for SwiGLU MLP."""

    def __init__(self, hs, ffn):
        super().__init__()
        self.gate_proj = nn.Linear(hs, ffn, bias=False)
        self.up_proj = nn.Linear(hs, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hs, bias=False)

    def forward(self, x):
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        # During ONNX export, gate/up/down_proj are replaced by FakeLinear (zeros OK)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Audio encoder (Whisper-style: 3×Conv2d + N×Transformer + Projector)
# ---------------------------------------------------------------------------

class AudioEncoderLayer(nn.Module):
    """Transformer encoder layer used in Qwen3-ASR audio encoder."""

    def __init__(self, d_model=896, n_heads=14, ffn=3584):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn_q_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_k_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_v_proj = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_out_proj = nn.Linear(d_model, d_model, bias=True)

        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn, bias=True)
        self.fc2 = nn.Linear(ffn, d_model, bias=True)

    def forward(self, x, attention_mask=None):
        # Pre-LN self-attention with multi-head
        residual = x
        x = self.self_attn_layer_norm(x)
        B, L, D = x.shape

        q = self.self_attn_q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn_k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn_v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None:
            attn_bias = (1.0 - attention_mask) * -10000.0
            attn_bias = attn_bias.unsqueeze(1).unsqueeze(2)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.self_attn_out_proj(attn_out)
        x = residual + attn_out

        # Pre-LN FFN
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x

    def load_weights(self, state_dict, prefix):
        self.self_attn_layer_norm.weight.data.copy_(state_dict[f"{prefix}.self_attn_layer_norm.weight"])
        self.self_attn_layer_norm.bias.data.copy_(state_dict[f"{prefix}.self_attn_layer_norm.bias"])
        self.final_layer_norm.weight.data.copy_(state_dict[f"{prefix}.final_layer_norm.weight"])
        self.final_layer_norm.bias.data.copy_(state_dict[f"{prefix}.final_layer_norm.bias"])
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            attr = f"self_attn_{proj}"
            getattr(self, attr).weight.data.copy_(state_dict[f"{prefix}.self_attn.{proj}.weight"])
            getattr(self, attr).bias.data.copy_(state_dict[f"{prefix}.self_attn.{proj}.bias"])
        for name in ["fc1", "fc2"]:
            getattr(self, name).weight.data.copy_(state_dict[f"{prefix}.{name}.weight"])
            getattr(self, name).bias.data.copy_(state_dict[f"{prefix}.{name}.bias"])


class AudioEncoder(nn.Module):
    """Whisper-style audio encoder: Conv2d ×3 → Transformer ×N → Projection.

    Configurable via kwargs to support different model sizes (0.6B, 1.7B, etc.).

    Uses a simplified forward (no chunking/windowing) which is equivalent to
    Wasser1462's conv_frontend+encoder combined path for short audio (≤30s).
    For short audio, a single window covers the full sequence so full-attention
    is correct. For longer audio, chunking/windowing would need to be added.
    """

    def __init__(self, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", 896)
        n_heads = kwargs.get("n_audio_heads", 14)
        ffn = kwargs.get("audio_ffn", 3584)
        n_layers = kwargs.get("n_audio_layers", 18)
        conv_hidden = kwargs.get("conv_hidden_size", 480)
        output_dim = kwargs.get("output_dim", 1024)
        n_window = kwargs.get("n_window", 50)
        n_window_infer = kwargs.get("n_window_infer", 800)
        conv_chunksize = kwargs.get("conv_chunksize", 500)

        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn = ffn
        self.n_layers = n_layers
        self.conv_hidden = conv_hidden
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize

        # Convolutional front-end (mel: [B, 128, T] → [B, conv_hidden, 16, T//8])
        self.conv2d1 = nn.Conv2d(1, conv_hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1)
        # Project from conv features to transformer dimension
        # After 3× stride-2: 128 mel bins → 16, so conv_out input = conv_hidden * 16
        self.conv_out = nn.Linear(conv_hidden * 16, d_model, bias=False)
        # Positional embeddings (sinusoidal)
        self.register_buffer("embed_positions", self._create_sinusoidal_positions(3000, d_model))
        # Transformer encoder layers
        self.layers = nn.ModuleList([AudioEncoderLayer(d_model, n_heads, ffn) for _ in range(n_layers)])
        # Output projection
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model, bias=True)
        self.proj2 = nn.Linear(d_model, output_dim, bias=True)

    @staticmethod
    def _create_sinusoidal_positions(max_len, dim):
        """Whisper-style sinusoidal PE: [all_sins | all_cosines].

        Uses Whisper's frequency formula: log_inc = ln(10000) / (dim/2 - 1).
        Verified against Wasser1462 encoder.int8.onnx PE table (cosim 1.0).
        """
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal position embedding requires even dim, got {dim}")
        log_inc = math.log(10000.0) / (dim // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(dim // 2, dtype=torch.float))
        scaled = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * inv.unsqueeze(0)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
        return pe

    def forward(self, input_features):
        # input_features: [B, 128, T] (mel spectrogram)
        x = input_features.unsqueeze(1)  # [B, 1, 128, T]

        # Conv front-end: stride=2 each, so T → T//2 → T//4 → T//8
        x = F.gelu(self.conv2d1(x))     # [B, C, 64, T//2]
        x = F.gelu(self.conv2d2(x))     # [B, C, 32, T//4]
        x = F.gelu(self.conv2d3(x))     # [B, C, 16, T//8]

        # Flatten spatial dims
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(B, W, C * H)                  # [B, T_enc, C*H]
        x = self.conv_out(x)                      # [B, T_enc, d_model]

        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.embed_positions[:seq_len, :].unsqueeze(0)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # [B, T_enc, hidden_size]
        return x

    def load_weights(self, state_dict, prefix="thinker.audio_tower"):
        """Load weights from flat state dict."""
        self.conv2d1.weight.data.copy_(state_dict[f"{prefix}.conv2d1.weight"])
        self.conv2d1.bias.data.copy_(state_dict[f"{prefix}.conv2d1.bias"])
        self.conv2d2.weight.data.copy_(state_dict[f"{prefix}.conv2d2.weight"])
        self.conv2d2.bias.data.copy_(state_dict[f"{prefix}.conv2d2.bias"])
        self.conv2d3.weight.data.copy_(state_dict[f"{prefix}.conv2d3.weight"])
        self.conv2d3.bias.data.copy_(state_dict[f"{prefix}.conv2d3.bias"])
        self.conv_out.weight.data.copy_(state_dict[f"{prefix}.conv_out.weight"])

        for i in range(self.n_layers):
            self.layers[i].load_weights(state_dict, f"{prefix}.layers.{i}")

        self.ln_post.weight.data.copy_(state_dict[f"{prefix}.ln_post.weight"])
        self.ln_post.bias.data.copy_(state_dict[f"{prefix}.ln_post.bias"])
        self.proj1.weight.data.copy_(state_dict[f"{prefix}.proj1.weight"])
        self.proj1.bias.data.copy_(state_dict[f"{prefix}.proj1.bias"])
        self.proj2.weight.data.copy_(state_dict[f"{prefix}.proj2.weight"])
        self.proj2.bias.data.copy_(state_dict[f"{prefix}.proj2.bias"])


# ---------------------------------------------------------------------------
# Dual-model components: Qwen3ASRConvFrontend + Qwen3ASREncoder
#
# Replicates the same I/O signatures as the verified Wasser1462 ONNX models:
#   conv_frontend.onnx:  [B, T, 128] → fold-into-batch → [B, T_valid, 896]
#   encoder.int8.onnx:   ([B, T_valid, 896], [B, T_valid]) → [B, T_valid, 1024]
# ---------------------------------------------------------------------------

class Qwen3ASRConvFrontend(nn.Module):
    """Fold-into-batch conv frontend matching Wasser1462 conv_frontend.onnx.

    Input:  [B, T, 128]     time-major mel features
    Output: [B, T_valid, 896]  all valid frames (pad/slice handled internally)

    Key design:
    - Dynamic padding to multiple of chunk_size via torch.cat (no .item() calls)
    - Fold-into-Batch for parallel conv across chunks
    - Internal Slice to remove ghost frames from partial last chunk
    - conv_out (bias=False) applied after unfold
    """

    def __init__(self, chunk_size=100, conv_hidden=480, d_model=896):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.conv_hidden = conv_hidden

        # Conv2d×3: k=3, s=2, p=1 (same as official & Wasser1462)
        self.conv1 = nn.Conv2d(1, conv_hidden, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_hidden, conv_hidden, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_hidden, conv_hidden, 3, stride=2, padding=1)

        # Linear projection: 7680 → d_model (bias=False — critical!)
        self.conv_out = nn.Linear(conv_hidden * 16, d_model, bias=False)

    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        """Official formula per stride-2 conv layer: L_out = (L-1)//2 + 1."""
        for _ in range(3):
            input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    def load_weights_from_audio_encoder(self, audio_encoder):
        self.conv1.weight.data.copy_(audio_encoder.conv2d1.weight.data)
        self.conv1.bias.data.copy_(audio_encoder.conv2d1.bias.data)
        self.conv2.weight.data.copy_(audio_encoder.conv2d2.weight.data)
        self.conv2.bias.data.copy_(audio_encoder.conv2d2.bias.data)
        self.conv3.weight.data.copy_(audio_encoder.conv2d3.weight.data)
        self.conv3.bias.data.copy_(audio_encoder.conv2d3.bias.data)
        self.conv_out.weight.data.copy_(audio_encoder.conv_out.weight.data)

    def forward(self, x):
        # x: [B, T, 128]  time-major
        B, T, C = x.shape
        chunk = self.chunk_size

        # ── Step 1: Dynamic pad to multiple of chunk_size ──
        # pad_len = (chunk - T % chunk) % chunk  (kept as int / symbolic)
        pad_len = (chunk - T % chunk) % chunk

        # Use concat with zeros for dynamic padding (F.pad breaks with Tensor args)
        # For ONNX export with dynamic T: use expand fallback if torch.zeros fails
        try:
            zeros = torch.zeros(B, pad_len, C, dtype=x.dtype, device=x.device)
        except (TypeError, RuntimeError):
            zeros = torch.zeros(1, 1, C, dtype=x.dtype, device=x.device).expand(B, pad_len, C)
        x = torch.cat([x, zeros], dim=1)  # [B, T_pad, 128]
        T_pad = T + pad_len

        # ── Step 2: Fold-into-Batch ──
        N = T_pad // chunk  # number of chunks (int / symbolic)
        # [B, N*chunk, 128] → [B, N, chunk, 128]
        x = x.reshape(B, N, chunk, C)
        # → [B, N, 128, chunk] → [B*N, 1, 128, chunk]  (Conv2d expects NCHW)
        x = x.permute(0, 1, 3, 2)               # [B, N, 128, chunk]
        x = x.reshape(B * N, 1, C, chunk)       # [B*N, 1, 128, chunk]

        # ── Step 3: Conv2d×3 (parallel across all chunks) ──
        x = F.gelu(self.conv1(x))   # [B*N, 480, 64, T_chunk_1]
        x = F.gelu(self.conv2(x))   # [B*N, 480, 32, T_chunk_2]
        x = F.gelu(self.conv3(x))   # [B*N, 480, 16, T_out_chunk]

        _, H_conv, H_freq, T_out_chunk = x.shape
        # H_conv=480, H_freq=16, T_out_chunk=f³(chunk)

        # ── Step 4: Unfold ──
        # [B*N, 480, 16, T_out] → [B, N, 480, 16, T_out]
        x = x.reshape(B, N, H_conv, H_freq, T_out_chunk)
        # → [B, 480, 16, N, T_out] → [B, 7680, N * T_out]
        x = x.permute(0, 2, 3, 1, 4)             # [B, 480, 16, N, T_out_chunk]
        x = x.reshape(B, H_conv * H_freq, N * T_out_chunk)  # [B, 7680, enc_T_padded]
        x = x.permute(0, 2, 1)                    # [B, enc_T_padded, 7680]

        # ── Step 5: Linear projection (Wasser1462: no partial-chunk ghost frame removal) ──
        # The model was trained with chunk_size=100 fold-into-batch, which retains all
        # N * T_out_chunk frames including those from zero-padded partial chunks.
        # Removing ghost frames (Slice) would deviate from training distribution.
        x = self.conv_out(x)  # [B, N * T_out_chunk, d_model=896]

        return x


class Qwen3ASREncoder(nn.Module):
    """Transformer encoder matching Wasser1462 encoder.int8.onnx.

    Input 1: input_features       [B, n_audio_tokens, 896]
    Input 2: feature_attention_mask [B, n_audio_tokens]  (1=valid, 0=padding)
    Output:  audio_features [B, n_audio_tokens, 1024]

    Uses Whisper-style concat PE (pre-computed buffer, sliced at runtime).
    attention_mask is passed through to all transformer layers.
    """

    def __init__(self, d_model=896, n_layers=18, n_heads=14,
                 ffn_dim=3584, output_dim=1024, max_pe_len=5000,
                 chunk_output_len=13):
        """chunk_output_len: conv output frames per chunk (fff(chunk_size)=13 for chunk=100).
        PE repeats positions 0..chunk_output_len-1 for each chunk, matching training."""
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_pe_len = max_pe_len
        self.chunk_output_len = chunk_output_len
        pe = self._create_sinusoidal_positions(max_pe_len, d_model)
        self.register_buffer("positional_embedding", pe, persistent=False)
        self.encoder_layers = nn.ModuleList([
            AudioEncoderLayer(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model, bias=True)
        self.proj2 = nn.Linear(d_model, output_dim, bias=True)

    @staticmethod
    def _create_sinusoidal_positions(max_len, dim):
        """Whisper-style: [all_sins | all_cosines] with log_inc = ln(10000)/(dim/2-1).

        Verified against Wasser1462 encoder.int8.onnx PE table (cosim 1.0).
        """
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal PE requires even dim, got {dim}")
        log_inc = math.log(10000.0) / (dim // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(dim // 2, dtype=torch.float))
        scaled = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * inv.unsqueeze(0)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
        return pe

    def load_weights_from_audio_encoder(self, audio_encoder):
        """Copy transformer/projection weights. PE is generated independently (do not copy)."""
        if len(self.encoder_layers) != len(audio_encoder.layers):
            raise ValueError(f"Layer count mismatch: {len(self.encoder_layers)} vs {len(audio_encoder.layers)}")
        for i, (d, s) in enumerate(zip(self.encoder_layers, audio_encoder.layers)):
            d.self_attn_layer_norm.weight.data.copy_(s.self_attn_layer_norm.weight.data)
            d.self_attn_layer_norm.bias.data.copy_(s.self_attn_layer_norm.bias.data)
            d.final_layer_norm.weight.data.copy_(s.final_layer_norm.weight.data)
            d.final_layer_norm.bias.data.copy_(s.final_layer_norm.bias.data)
            for p in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                getattr(d, f'self_attn_{p}').weight.data.copy_(getattr(s, f'self_attn_{p}').weight.data)
                getattr(d, f'self_attn_{p}').bias.data.copy_(getattr(s, f'self_attn_{p}').bias.data)
            d.fc1.weight.data.copy_(s.fc1.weight.data)
            d.fc1.bias.data.copy_(s.fc1.bias.data)
            d.fc2.weight.data.copy_(s.fc2.weight.data)
            d.fc2.bias.data.copy_(s.fc2.bias.data)
        self.ln_post.weight.data.copy_(audio_encoder.ln_post.weight.data)
        self.ln_post.bias.data.copy_(audio_encoder.ln_post.bias.data)
        self.proj1.weight.data.copy_(audio_encoder.proj1.weight.data)
        self.proj1.bias.data.copy_(audio_encoder.proj1.bias.data)
        self.proj2.weight.data.copy_(audio_encoder.proj2.weight.data)
        self.proj2.bias.data.copy_(audio_encoder.proj2.bias.data)
        # PE is NOT copied — Qwen3ASREncoder generates its own Whisper-concat PE

    def forward(self, input_features, attention_mask=None):
        seq_len = input_features.size(1)
        # Per-chunk repeating PE (matching training: PE resets to 0..12 for each chunk)
        # seq_len is always N * chunk_output_len (e.g. N * 13 for chunk_size=100)
        pe_chunk = self.positional_embedding[:self.chunk_output_len]  # [chunk_out, d_model]
        n_repeats = (seq_len + self.chunk_output_len - 1) // self.chunk_output_len
        pe = pe_chunk.repeat(n_repeats, 1)[:seq_len]  # [seq_len, d_model]
        x = input_features + pe.unsqueeze(0)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x


def split_audio_encoder(audio_encoder, chunk_size=100):
    """Split AudioEncoder into (Qwen3ASRConvFrontend, Qwen3ASREncoder).

    The frontend uses fold-into-batch (matching Wasser1462 conv_frontend.onnx).
    The encoder generates its own Whisper-concat PE (matching Wasser1462 encoder.int8.onnx).

    chunk_size=100 matches Wasser1462 ONNX behavior: ceil(T/100) * fff(100) output frames.
    """
    frontend = Qwen3ASRConvFrontend(
        chunk_size=chunk_size,
        conv_hidden=audio_encoder.conv_hidden,
        d_model=audio_encoder.d_model,
    )
    frontend.load_weights_from_audio_encoder(audio_encoder)
    chunk_output_len = Qwen3ASRConvFrontend._get_feat_extract_output_lengths(chunk_size)
    encoder = Qwen3ASREncoder(
        d_model=audio_encoder.d_model,
        n_layers=audio_encoder.n_layers,
        n_heads=audio_encoder.n_heads,
        ffn_dim=audio_encoder.ffn,
        output_dim=audio_encoder.proj2.out_features,
        max_pe_len=audio_encoder.embed_positions.shape[0],
        chunk_output_len=chunk_output_len,
    )
    encoder.load_weights_from_audio_encoder(audio_encoder)
    return frontend, encoder


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

class Qwen3ASRThinker(nn.Module):
    """The ``thinker`` sub-module containing lm_head, model (embed+layers+norm), audio_tower."""

    def __init__(self, config, audio_config):
        super().__init__()
        hs = config["hidden_size"]
        n_heads = config["num_attention_heads"]
        n_kv = config.get("num_key_value_heads", n_heads)
        head_dim = config.get("head_dim", hs // n_heads)
        ffn = config["intermediate_size"]
        vocab = config["vocab_size"]

        self.lm_head = nn.Linear(hs, vocab, bias=False)
        self.model = Qwen3TextModel(config)
        self.audio_tower = AudioEncoder(**audio_config)


class Qwen3TextModel(nn.Module):
    """Text backbone: embed_tokens, N×decoder layers, final norm."""

    def __init__(self, config):
        super().__init__()
        n_layers = config.get("num_hidden_layers", 28)
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(n_layers)])
        self.norm = QwenRMSNorm(config["hidden_size"])


class Qwen3ASRWrapper(nn.Module):
    """Top-level wrapper matching the state dict's ``thinker.*`` structure.

    The mapper navigates via attribute access::
        model.thinker.lm_head → nn.Linear
        model.thinker.model.embed_tokens → nn.Embedding
        model.thinker.model.layers[0] → Qwen3DecoderLayer
        model.thinker.audio_tower → AudioEncoder
    """

    def __init__(self, config, audio_config):
        super().__init__()
        self.text_config = config
        self.thinker = Qwen3ASRThinker(config, audio_config)

    def _load_sharded_state_dict(self, model_path):
        """Load weights from sharded safetensors (indexed by model.safetensors.index.json)."""
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        # Group keys by shard file
        shard_files = {}
        for key, shard_name in weight_map.items():
            shard_files.setdefault(shard_name, []).append(key)

        # Load each shard
        sd = {}
        for shard_name, keys in shard_files.items():
            shard_path = os.path.join(model_path, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Expected shard {shard_path}")
            with safe_open(shard_path, framework="pt") as f:
                for k in keys:
                    sd[k] = f.get_tensor(k)

        return sd

    def load_weights(self, model_path):
        """Load weights from safetensors (single or sharded)."""
        single_path = os.path.join(model_path, "model.safetensors")
        index_path = os.path.join(model_path, "model.safetensors.index.json")

        if os.path.exists(single_path):
            with safe_open(single_path, framework="pt") as f:
                keys = list(f.keys())
                sd = {k: f.get_tensor(k) for k in keys}
        elif os.path.exists(index_path):
            sd = self._load_sharded_state_dict(model_path)
        else:
            raise FileNotFoundError(
                f"Expected either {single_path} or {index_path}"
            )

        # Text embeddings
        self.thinker.model.embed_tokens.weight.data.copy_(sd["thinker.model.embed_tokens.weight"])
        self.thinker.lm_head.weight.data.copy_(sd["thinker.lm_head.weight"])
        self.thinker.model.norm.weight.data.copy_(sd["thinker.model.norm.weight"])

        # Decoder layers
        n_text_layers = self.text_config.get("num_hidden_layers", 28)
        for i in range(n_text_layers):
            prefix = f"thinker.model.layers.{i}"
            self.thinker.model.layers[i].load_weights(sd, prefix)

        # Audio encoder
        self.thinker.audio_tower.load_weights(sd)


def load_qwen3_asr(model_path, dtype=torch.float32):
    """Load Qwen3-ASR from a local model directory containing config.json + safetensors.

    Supports both single ``model.safetensors`` and sharded
    ``model-0000N-of-0000M.safetensors`` via ``model.safetensors.index.json``.

    Returns a Qwen3ASRWrapper module that the MNN LlmModel pipeline can consume.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    tc = raw["thinker_config"]["text_config"]
    ac = raw["thinker_config"]["audio_config"]

    text_config = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        "intermediate_size": tc["intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "rope_theta": tc.get("rope_theta", 1000000.0),
        "max_position_embeddings": tc.get("max_position_embeddings", 65536),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "num_hidden_layers": tc.get("num_hidden_layers", 28),
    }

    audio_config = {
        "d_model": ac.get("d_model", 896),
        "n_audio_heads": ac.get("encoder_attention_heads", 14),
        "audio_ffn": ac.get("encoder_ffn_dim", 3584),
        "n_audio_layers": ac.get("num_hidden_layers", 18),
        "conv_hidden_size": ac.get("downsample_hidden_size", 480),
        "output_dim": ac.get("output_dim", text_config["hidden_size"]),
        "n_window": ac.get("n_window", 50),
        "n_window_infer": ac.get("n_window_infer", 800),
        "conv_chunksize": ac.get("conv_chunksize", 500),
    }

    model = Qwen3ASRWrapper(text_config, audio_config)
    model.load_weights(model_path)
    model = model.to(dtype=dtype)
    model.eval()
    return model
