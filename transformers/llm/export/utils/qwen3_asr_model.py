"""Qwen3-ASR custom model — built from safetensors weights.

Qwen3-ASR is not in the standard transformers library. This module
loads the model directly from safetensors into a torch.nn.Module
structure that the MNN llmexport.py pipeline can consume.

Architecture:
  thinker                    — top-level wrapper
  ├── lm_head               — Linear(1024, 151936, bias=False)
  ├── model.embed_tokens    — Embedding(151936, 1024)
  ├── model.layers[0..27]   — 28× Qwen3 decoder layers
  │   ├── input_layernorm          — RMSNorm(1024)
  │   ├── self_attn.q_proj         — Linear(2048, 1024, bias=False)
  │   ├── self_attn.k_proj         — Linear(1024, 1024, bias=False)
  │   ├── self_attn.v_proj         — Linear(1024, 1024, bias=False)
  │   ├── self_attn.o_proj         — Linear(1024, 2048, bias=False)
  │   ├── self_attn.q_norm         — RMSNorm(128)
  │   ├── self_attn.k_norm         — RMSNorm(128)
  │   ├── post_attention_layernorm — RMSNorm(1024)
  │   └── mlp
  │       ├── gate_proj  — Linear(3072, 1024, bias=False)
  │       ├── up_proj    — Linear(3072, 1024, bias=False)
  │       └── down_proj  — Linear(1024, 3072, bias=False)
  ├── model.norm           — RMSNorm(1024)
  └── audio_tower          — Whisper-style audio encoder
      ├── conv2d1   — Conv2d(1, 480, 3, stride=2)
      ├── conv2d2   — Conv2d(480, 480, 3, stride=2)
      ├── conv2d3   — Conv2d(480, 480, 3, stride=2)
      ├── conv_out  — Linear(7680, 896)
      ├── layers[0..17] — 18× Transformer encoder layers
      │   ├── self_attn_layer_norm  — LayerNorm(896)
      │   ├── self_attn             — QKV attention (14 heads, 64 dim)
      │   │   ├── q_proj, k_proj, v_proj, out_proj
      │   ├── fc1           — Linear(896, 3584)
      │   ├── fc2           — Linear(3584, 896)
      │   └── final_layer_norm — LayerNorm(896)
      ├── ln_post  — LayerNorm(896)
      ├── proj1    — Linear(896, 896)
      └── proj2    — Linear(896, 1024)
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
# Audio encoder (Whisper-style: 3×Conv2d + 18×Transformer + Projector)
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
    """Whisper-style audio encoder: Conv2d ×3 → Transformer ×18 → Projection."""

    def __init__(self):
        super().__init__()
        # Convolutional front-end (mel: [B, 128, T] → [B, 480, T/8, 16])
        self.conv2d1 = nn.Conv2d(1, 480, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(480, 480, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(480, 480, kernel_size=3, stride=2, padding=1)
        # Project from conv features to transformer dimension
        self.conv_out = nn.Linear(7680, 896)  # 480 * 16 = 7680
        # Positional embeddings (sinusoidal)
        self.register_buffer("embed_positions", self._create_sinusoidal_positions(3000, 896))
        # Transformer encoder layers
        self.layers = nn.ModuleList([AudioEncoderLayer() for _ in range(18)])
        # Output projection
        self.ln_post = nn.LayerNorm(896)
        self.proj1 = nn.Linear(896, 896, bias=True)
        self.proj2 = nn.Linear(896, 1024, bias=True)

    @staticmethod
    def _create_sinusoidal_positions(max_len, dim):
        """Create sinusoidal position embeddings (same as Whisper)."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_features):
        # input_features: [B, 128, T] (mel spectrogram)
        x = input_features.unsqueeze(1)  # [B, 1, 128, T]

        # Conv front-end: stride=2 each, so T → T//2 → T//4 → T//8
        x = F.gelu(self.conv2d1(x))     # [B, 480, 64, T//2]
        x = F.gelu(self.conv2d2(x))     # [B, 480, 32, T//4]
        x = F.gelu(self.conv2d3(x))     # [B, 480, 16, T//8]

        # Flatten spatial dims
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(B, W, C * H)                  # [B, T_enc, 480*16=7680]
        x = self.conv_out(x)                      # [B, T_enc, 896]

        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.embed_positions[:seq_len, :].unsqueeze(0)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # [B, T_enc, 1024]
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

        for i in range(18):
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

    def __init__(self, config):
        super().__init__()
        hs = config["hidden_size"]
        n_heads = config["num_attention_heads"]
        n_kv = config.get("num_key_value_heads", n_heads)
        head_dim = config.get("head_dim", hs // n_heads)
        ffn = config["intermediate_size"]
        vocab = config["vocab_size"]

        self.lm_head = nn.Linear(hs, vocab, bias=False)
        self.model = Qwen3TextModel(config)
        self.audio_tower = AudioEncoder()


class Qwen3TextModel(nn.Module):
    """Text backbone: embed_tokens, 28×decoder layers, final norm."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(28)])
        self.norm = QwenRMSNorm(config["hidden_size"])


class Qwen3ASRWrapper(nn.Module):
    """Top-level wrapper matching the state dict's ``thinker.*`` structure.

    The mapper navigates via attribute access::
        model.thinker.lm_head → nn.Linear
        model.thinker.model.embed_tokens → nn.Embedding
        model.thinker.model.layers[0] → Qwen3DecoderLayer
        model.thinker.audio_tower → AudioEncoder
    """

    def __init__(self, config):
        super().__init__()
        self.thinker = Qwen3ASRThinker(config)

    def load_weights(self, model_path):
        """Load weights from a safetensors file."""
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Expected {safetensors_path}")

        with safe_open(safetensors_path, framework="pt") as f:
            keys = list(f.keys())
            sd = {k: f.get_tensor(k) for k in keys}

        # Text embeddings
        self.thinker.model.embed_tokens.weight.data.copy_(sd["thinker.model.embed_tokens.weight"])
        self.thinker.lm_head.weight.data.copy_(sd["thinker.lm_head.weight"])
        self.thinker.model.norm.weight.data.copy_(sd["thinker.model.norm.weight"])

        # Decoder layers
        for i in range(28):
            prefix = f"thinker.model.layers.{i}"
            self.thinker.model.layers[i].load_weights(sd, prefix)

        # Audio encoder
        self.thinker.audio_tower.load_weights(sd)


def load_qwen3_asr(model_path, dtype=torch.float32):
    """Load Qwen3-ASR from a local model directory containing config.json + model.safetensors.

    Returns a Qwen3ASRWrapper module that the MNN LlmModel pipeline can consume.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    tc = raw["thinker_config"]["text_config"]
    config = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        "intermediate_size": tc["intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "rope_theta": tc.get("rope_theta", 1000000.0),
        "max_position_embeddings": tc.get("max_position_embeddings", 65536),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
    }

    model = Qwen3ASRWrapper(config)
    model.load_weights(model_path)
    model = model.to(dtype=dtype)
    model.eval()
    return model
