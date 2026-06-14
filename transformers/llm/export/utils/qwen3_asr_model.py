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

    def forward(self, x):
        # Pre-LN self-attention with multi-head
        residual = x
        x = self.self_attn_layer_norm(x)
        B, L, D = x.shape

        q = self.self_attn_q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn_k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn_v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
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
        """Create sinusoidal position embeddings (Whisper-style concatenation).

        Official formula: cat([sin(scaled_time), cos(scaled_time)], dim=-1)
        Groups all sines in the first half, all cosines in the second half.
        NOT interleaved (sin,cos,sin,cos...).
        """
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal position embedding requires even dim, got {dim}")
        log_timescale_increment = torch.log(torch.tensor(10000.0)) / (dim // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(dim // 2, dtype=torch.float))
        scaled_time = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * inv_timescales.unsqueeze(0)
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
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
