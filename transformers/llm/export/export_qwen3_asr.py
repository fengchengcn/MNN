#!/usr/bin/env python3
"""
Qwen3-ASR-0.6B → MNN Export Script

Exports the Qwen3-ASR-0.6B model to MNN format by:
1. Loading weights directly from safetensors (bypassing transformers library)
2. Building equivalent PyTorch modules
3. Exporting to ONNX
4. Converting to MNN via MNNConvert

Usage:
    python export_qwen3_asr.py --model_path /path/to/Qwen3-ASR-0.6B --dst_path ./output
"""

import os
import sys
import json
import math
import argparse
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ============================================================
# Audio Encoder
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding (non-learnable)."""
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :].to(x.dtype)


class AudioEncoderLayer(nn.Module):
    """Standard Transformer Encoder layer (pre-norm)."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn_q = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_k = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_v = nn.Linear(d_model, d_model, bias=True)
        self.self_attn_o = nn.Linear(d_model, d_model, bias=True)

        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.activation_fn = F.gelu

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        B, S, D = hidden_states.shape

        # Self-attention
        q = self.self_attn_q(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn_k(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn_v(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        attn_output = self.self_attn_o(attn_output)
        hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    """Qwen3-ASR Audio Encoder - Whisper-style."""
    def __init__(self, config=None):
        super().__init__()
        d_model = 896
        n_heads = 14
        d_ff = 3584
        num_layers = 18

        # Conv2d downsampling (3 layers, stride=2 each)
        self.conv1 = nn.Conv2d(1, 480, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(480, 480, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(480, 480, kernel_size=3, stride=2, padding=1, bias=True)

        # Projection from conv output to d_model
        self.conv_out = nn.Linear(480 * 16, d_model, bias=False)

        # Sinusoidal position embedding
        self.embed_pos = SinusoidalPositionEmbedding(d_model, max_len=3000)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            AudioEncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])

        # Post-encoder
        self.ln_post = nn.LayerNorm(d_model)

        # Projector: LayerNorm → GELU → Linear(896→896) → Linear(896→1024)
        self.proj1 = nn.Linear(d_model, d_model, bias=True)  # 896→896
        self.proj2 = nn.Linear(d_model, 1024, bias=True)     # 896→1024

    def forward(self, input_features):
        """
        Args:
            input_features: [batch, 128, time] Mel spectrogram
        Returns:
            audio_embeds: [batch, T', 1024] Encoder outputs
        """
        # Reshape for Conv2d: [B, 128, T] → [B, 1, 128, T]
        x = input_features.unsqueeze(1)

        # Conv2d downsampling
        x = F.gelu(self.conv1(x))  # [B, 480, 64, T/2]
        x = F.gelu(self.conv2(x))  # [B, 480, 32, T/4]
        x = F.gelu(self.conv3(x))  # [B, 480, 16, T/8]

        # Reshape: [B, 480, 16, T/8] → [B, T/8, 480*16]
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)

        # Linear projection to d_model
        x = self.conv_out(x)  # [B, T', 896]

        # Sinusoidal position embedding
        x = self.embed_pos(x)

        # Transformer encoder layers
        for layer in self.layers:
            x = layer(x)

        # Post LayerNorm
        x = self.ln_post(x)

        # Projector: Linear → GELU → Linear
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        return x  # [B, T', 1024]

    def load_weights(self, state_dict):
        """Load weights with 'thinker.audio_tower.' prefix."""
        mapping = {
            'conv1.weight': 'thinker.audio_tower.conv2d1.weight',
            'conv1.bias': 'thinker.audio_tower.conv2d1.bias',
            'conv2.weight': 'thinker.audio_tower.conv2d2.weight',
            'conv2.bias': 'thinker.audio_tower.conv2d2.bias',
            'conv3.weight': 'thinker.audio_tower.conv2d3.weight',
            'conv3.bias': 'thinker.audio_tower.conv2d3.bias',
            'conv_out.weight': 'thinker.audio_tower.conv_out.weight',
            'ln_post.weight': 'thinker.audio_tower.ln_post.weight',
            'ln_post.bias': 'thinker.audio_tower.ln_post.bias',
            'proj1.weight': 'thinker.audio_tower.proj1.weight',
            'proj1.bias': 'thinker.audio_tower.proj1.bias',
            'proj2.weight': 'thinker.audio_tower.proj2.weight',
            'proj2.bias': 'thinker.audio_tower.proj2.bias',
        }

        # Load top-level params
        for model_key, sd_key in mapping.items():
            if sd_key in state_dict:
                self.state_dict()[model_key].copy_(state_dict[sd_key])
                print(f"  Loaded: {model_key}")
            else:
                print(f"  WARNING: {sd_key} not found in state_dict")

        # Load transformer layers
        for i, layer in enumerate(self.layers):
            prefix = f'thinker.audio_tower.layers.{i}'
            layer_mapping = {
                'self_attn_layer_norm.weight': f'{prefix}.self_attn_layer_norm.weight',
                'self_attn_layer_norm.bias': f'{prefix}.self_attn_layer_norm.bias',
                'self_attn_q.weight': f'{prefix}.self_attn.q_proj.weight',
                'self_attn_q.bias': f'{prefix}.self_attn.q_proj.bias',
                'self_attn_k.weight': f'{prefix}.self_attn.k_proj.weight',
                'self_attn_k.bias': f'{prefix}.self_attn.k_proj.bias',
                'self_attn_v.weight': f'{prefix}.self_attn.v_proj.weight',
                'self_attn_v.bias': f'{prefix}.self_attn.v_proj.bias',
                'self_attn_o.weight': f'{prefix}.self_attn.out_proj.weight',
                'self_attn_o.bias': f'{prefix}.self_attn.out_proj.bias',
                'final_layer_norm.weight': f'{prefix}.final_layer_norm.weight',
                'final_layer_norm.bias': f'{prefix}.final_layer_norm.bias',
                'fc1.weight': f'{prefix}.fc1.weight',
                'fc1.bias': f'{prefix}.fc1.bias',
                'fc2.weight': f'{prefix}.fc2.weight',
                'fc2.bias': f'{prefix}.fc2.bias',
            }
            for model_key, sd_key in layer_mapping.items():
                if sd_key in state_dict:
                    layer.state_dict()[model_key].copy_(state_dict[sd_key])
                else:
                    print(f"  WARNING: {sd_key} not found in state_dict")
        print(f"  Loaded {len(self.layers)} transformer layers")
        print("  Audio encoder weights loaded successfully")


# ============================================================
# LLM Decoder - Standard Qwen3
# ============================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=131072, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [B, num_heads, seq_len, head_dim]
        # position_ids: [B, seq_len]
        inv_freq = self.inv_freq.to(x.dtype)
        inv_freq = inv_freq[None, :, None].float()  # [1, dim/2, 1]
        position_ids = position_ids[:, None, :].float()  # [B, 1, seq_len]
        freqs = (inv_freq @ position_ids).transpose(1, 2)  # [B, seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, seq_len, dim]
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, seq_len, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK-Norm (Qwen3 specific)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=1e-6)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=1e-6)

    def forward(self, hidden_states, attention_mask, cos, sin):
        B, S, D = hidden_states.shape

        # Project
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm (per head RMSNorm)
        q = self.q_norm(q.float()).to(q.dtype)
        k = self.k_norm(k.float()).to(k.dtype)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand K,V to match Q heads
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128, intermediate_size=3072):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Decoder(nn.Module):
    """Qwen3 decoder model with inputs_embeds support."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get('hidden_size', 1024)
        self.num_heads = config.get('num_attention_heads', 16)
        self.num_kv_heads = config.get('num_key_value_heads', 8)
        self.head_dim = config.get('head_dim', 128)
        self.intermediate_size = config.get('intermediate_size', 3072)
        self.num_layers = config.get('num_hidden_layers', 28)
        self.vocab_size = config.get('vocab_size', 151936)

        # Embedding
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        # Note: lm_head == embed_tokens (tied embeddings)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Rotary embeddings
        self.rotary = Qwen3RotaryEmbedding(self.head_dim)

        # Layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim, self.intermediate_size)
            for _ in range(self.num_layers)
        ])

        self.norm = Qwen3RMSNorm(self.hidden_size, eps=1e-6)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        B, S, D = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(S, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        # Precompute RoPE
        cos, sin = self.rotary(hidden_states, position_ids)

        if attention_mask is None:
            # Causal mask
            attention_mask = torch.full((S, S), float('-inf'), device=hidden_states.device)
            attention_mask = torch.triu(attention_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        # Run decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos, sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, state_dict):
        """Load weights with 'thinker.model.' or 'thinker.' prefix."""
        with torch.no_grad():
            # Embed tokens
            if 'thinker.model.embed_tokens.weight' in state_dict:
                self.embed_tokens.weight.copy_(state_dict['thinker.model.embed_tokens.weight'])
            print(f"  Loaded: embed_tokens.weight [{self.embed_tokens.weight.shape}]")

            # LM head (tied: same as embed_tokens)
            if 'thinker.lm_head.weight' in state_dict:
                self.lm_head.weight.copy_(state_dict['thinker.lm_head.weight'])
                print(f"  Loaded: lm_head.weight [{self.lm_head.weight.shape}]")

            # Load transformer layers
            for i in range(self.num_layers):
                prefix = f'thinker.model.layers.{i}'
                layer = self.layers[i]

                # Input layernorm (RMSNorm - weight only, no bias)
                if f'{prefix}.input_layernorm.weight' in state_dict:
                    layer.input_layernorm.weight.copy_(state_dict[f'{prefix}.input_layernorm.weight'])

                # Attention (no bias - attention_bias=False)
                if f'{prefix}.self_attn.q_proj.weight' in state_dict:
                    layer.self_attn.q_proj.weight.copy_(state_dict[f'{prefix}.self_attn.q_proj.weight'])
                    layer.self_attn.k_proj.weight.copy_(state_dict[f'{prefix}.self_attn.k_proj.weight'])
                    layer.self_attn.v_proj.weight.copy_(state_dict[f'{prefix}.self_attn.v_proj.weight'])
                    layer.self_attn.o_proj.weight.copy_(state_dict[f'{prefix}.self_attn.o_proj.weight'])

                    # QK-Norm
                    if f'{prefix}.self_attn.q_norm.weight' in state_dict:
                        layer.self_attn.q_norm.weight.copy_(state_dict[f'{prefix}.self_attn.q_norm.weight'])
                        layer.self_attn.k_norm.weight.copy_(state_dict[f'{prefix}.self_attn.k_norm.weight'])

                # Post-attention layernorm (RMSNorm - weight only, no bias)
                if f'{prefix}.post_attention_layernorm.weight' in state_dict:
                    layer.post_attention_layernorm.weight.copy_(state_dict[f'{prefix}.post_attention_layernorm.weight'])

                # MLP
                if f'{prefix}.mlp.gate_proj.weight' in state_dict:
                    layer.mlp.gate_proj.weight.copy_(state_dict[f'{prefix}.mlp.gate_proj.weight'])
                    layer.mlp.up_proj.weight.copy_(state_dict[f'{prefix}.mlp.up_proj.weight'])
                    layer.mlp.down_proj.weight.copy_(state_dict[f'{prefix}.mlp.down_proj.weight'])

            print(f"  Loaded {self.num_layers} transformer layers")
            print("  Decoder weights loaded successfully")


class Qwen3DecoderWithKVCache(nn.Module):
    """Qwen3 decoder with explicit K/V cache (all 28 layers in single forward).

    Design:
    - K/V cache is stored as a flat 5D tensor: [L, B, Hk, C, Hd] = [28,1,8,C,128]
    - Works for both prefill (C=0) and decode (C>0)
    - Single onForward call processes all 28 layers with cache management

    Inputs:
        inputs_embeds:   [B, S, D]      token embeddings
        position_ids:    [B, S]         absolute positions
        attention_mask:  [B, 1, S, C+S] causal mask
        k_cache:         [L, B, Hk, C, Hd] cached K for all layers
        v_cache:         [L, B, Hk, C, Hd] cached V for all layers
    Outputs:
        logits:  [B, S, V]
        new_k:   [L, B, Hk, C+S, Hd] updated K cache
        new_v:   [L, B, Hk, C+S, Hd] updated V cache
    """
    NUM_LAYERS = 28
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    HIDDEN = 1024

    def __init__(self, decoder):
        super().__init__()
        # Take only the decoder guts (no embed_tokens)
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.lm_head = decoder.lm_head
        self.rotary = decoder.rotary
        self.head_dim = decoder.head_dim
        self.num_heads = decoder.num_heads
        self.num_kv_heads = decoder.num_kv_heads
        self.num_key_value_groups = decoder.num_heads // decoder.num_kv_heads

    def forward(self, inputs_embeds, position_ids, attention_mask, k_cache, v_cache):
        B, S, D = inputs_embeds.shape
        past_len = k_cache.shape[3]

        # Precompute RoPE for all S positions in this step
        cos, sin = self.rotary(inputs_embeds, position_ids)

        new_keys = []
        new_values = []

        hidden_states = inputs_embeds

        for i in range(self.NUM_LAYERS):
            layer = self.layers[i]
            past_k = k_cache[i]   # [B, Hk, past_len, Hd]
            past_v = v_cache[i]   # [B, Hk, past_len, Hd]

            # === Self-attention ===
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            q = layer.self_attn.q_proj(hidden_states)
            k = layer.self_attn.k_proj(hidden_states)
            v = layer.self_attn.v_proj(hidden_states)

            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # QK-Norm
            q = layer.self_attn.q_norm(q.float()).to(q.dtype)
            k = layer.self_attn.k_norm(k.float()).to(k.dtype)

            # RoPE
            q, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

            # KV Cache: concat with past
            k_full = torch.cat([past_k, k_rope], dim=2)  # [B, Hk, past_len+S, Hd]
            v_full = torch.cat([past_v, v], dim=2)       # [B, Hk, past_len+S, Hd]
            new_keys.append(k_full)
            new_values.append(v_full)

            # GQA: expand KV heads to match Q heads
            k_attn = k_full.repeat_interleave(self.num_key_value_groups, dim=1)
            v_attn = v_full.repeat_interleave(self.num_key_value_groups, dim=1)

            # Attention
            attn = torch.matmul(q, k_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            hidden_states = torch.matmul(attn, v_attn)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
            hidden_states = layer.self_attn.o_proj(hidden_states)
            hidden_states = residual + hidden_states

            # === MLP ===
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # Final norm + lm_head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Stack all layer caches back to 5D
        new_k = torch.stack(new_keys, dim=0)  # [L, B, Hk, past_len+S, Hd]
        new_v = torch.stack(new_values, dim=0)

        return logits, new_k, new_v


# ============================================================
# Tokenizer handling
# ============================================================

def export_tokenizer(model_path, dst_path):
    """Copy tokenizer files to output directory."""
    import shutil
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                       'tokenizer.txt', 'added_tokens.json', 'special_tokens_map.json']
    for f in tokenizer_files:
        src = os.path.join(model_path, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_path, f))
            print(f"  Copied: {f}")

    # Create tokenizer.txt from vocab.json if needed
    if not os.path.exists(os.path.join(dst_path, 'tokenizer.txt')):
        vocab_path = os.path.join(model_path, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            tok_path = os.path.join(dst_path, 'tokenizer.txt')
            with open(tok_path, 'w') as f:
                for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                    f.write(f'{token}\n')
            print(f"  Created: tokenizer.txt ({len(vocab)} tokens)")


# ============================================================
# ONNX Export
# ============================================================

def export_onnx(model, onnx_path, input_tensors, input_names, output_names, dynamic_axes, model_name):
    """Export a PyTorch model to ONNX."""
    print(f"\n=== Exporting {model_name} to ONNX ===")

    # set to eval and float
    model.eval()
    model.float()

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # For torch 2.12+, use dynamic_shapes instead of dynamic_axes
    try:
        dynamic_shapes = []
        for name in input_names:
            if name in dynamic_axes:
                dim_shapes = {}
                for k, v in dynamic_axes[name].items():
                    dim_shapes[k] = torch.export.Dim(v)
                dynamic_shapes.append(dim_shapes)
            else:
                dynamic_shapes.append(None)

        torch.onnx.export(
            model,
            input_tensors,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes if any(dynamic_shapes) else None,
            opset_version=18,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"  torch.export failed ({e}), falling back to dynamo=False...")
        torch.onnx.export(
            model,
            input_tensors,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  ONNX saved to: {onnx_path}")
    return onnx_path


def convert_to_mnn(onnx_path, mnn_path, mnnconvert_binary, quant_bit=16, quant_block=0):
    """Convert ONNX to MNN format using MNNConvert."""
    print(f"\n=== Converting {os.path.basename(onnx_path)} to MNN ===")

    os.makedirs(os.path.dirname(mnn_path), exist_ok=True)

    cmd = [mnnconvert_binary, '-f', 'ONNX', '--modelFile', onnx_path, '--MNNModel', mnn_path]

    if quant_bit < 16:
        cmd.extend(['--weightQuantBits', str(quant_bit)])
        if quant_block > 0:
            cmd.extend(['--weightQuantBlock', str(quant_block)])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    print(f"  MNN model saved to: {mnn_path}")
    if os.path.exists(mnn_path):
        size_mb = os.path.getsize(mnn_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Export Qwen3-ASR-0.6B to MNN format')
    parser.add_argument('--model_path', type=str, default='/root/projects/mnn-models/Qwen3-ASR-0.6B',
                        help='Path to the Qwen3-ASR model directory')
    parser.add_argument('--dst_path', type=str, default='/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN',
                        help='Output directory for MNN models')
    parser.add_argument('--mnnconvert', type=str, default='/root/projects/MNN/build/MNNConvert',
                        help='Path to MNNConvert binary')
    parser.add_argument('--quant_bit', type=int, default=4,
                        help='Quantization bits (4, 8, or 16 for no quantization)')
    parser.add_argument('--quant_block', type=int, default=0,
                        help='Quantization block size')
    parser.add_argument('--export_audio', action='store_true', default=True,
                        help='Export audio encoder')
    parser.add_argument('--export_decoder', action='store_true', default=True,
                        help='Export LLM decoder')
    parser.add_argument('--export_kv_cache', action='store_true', default=True,
                        help='Export LLM decoder with KV cache support')
    parser.add_argument('--skip_onnx', action='store_true',
                        help='Skip ONNX export (use existing ONNX files)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.dst_path, exist_ok=True)
    onnx_dir = os.path.join(args.dst_path, 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)

    # Check MNNConvert
    if not os.path.exists(args.mnnconvert):
        print(f"WARNING: MNNConvert not found at {args.mnnconvert}")
        print("ONNX files will be created but MNN conversion will be skipped.")
        args.mnnconvert = None

    # Load state dict
    print("=" * 60)
    print("Loading safetensors weights...")
    safetensors_path = os.path.join(args.model_path, 'model.safetensors')
    if not os.path.exists(safetensors_path):
        print(f"ERROR: {safetensors_path} not found!")
        return 1

    state_dict = {}
    with safe_open(safetensors_path, framework='pt') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    print(f"  Loaded {len(state_dict)} weight tensors")
    print()

    # Read config
    config_path = os.path.join(args.model_path, 'config.json')
    with open(config_path) as f:
        full_config = json.load(f)

    # Decoder config
    decoder_config = full_config.get('thinker_config', {}).get('text_config', {})
    decoder_config['model_type'] = 'qwen3_asr'
    decoder_config['num_hidden_layers'] = decoder_config.get('num_hidden_layers', 28)
    decoder_config['vocab_size'] = decoder_config.get('vocab_size', 151936)

    print("=" * 60)
    print("Model Configuration:")
    print(f"  Audio Encoder: 3 Conv2d + 18 Transformer layers (d_model=896, n_heads=14)")
    print(f"  LLM Decoder: 28 Qwen3 layers (hidden=1024, heads=16, kv_heads=8)")
    print(f"  Total parameters: {sum(v.numel() for v in state_dict.values()) / 1e9:.2f}B")
    print(f"  Output: {args.dst_path}")
    print(f"  Quantization: {'int' + str(args.quant_bit) if args.quant_bit < 16 else 'bf16'}")
    print()

    # =========================================
    # Export Audio Encoder
    # =========================================
    if args.export_audio:
        print("=" * 60)
        print("EXPORTING AUDIO ENCODER")
        print("=" * 60)

        audio_encoder = Qwen3ASRAudioEncoder()
        audio_encoder.load_weights(state_dict)

        audio_onnx_path = os.path.join(onnx_dir, 'audio_encoder.onnx')
        audio_mnn_path = os.path.join(args.dst_path, 'audio_encoder.mnn')

        if not args.skip_onnx:
            # Export to ONNX (dynamic time dimension)
            dummy_input = torch.randn(1, 128, 200)  # ~12.8 seconds at 16kHz
            export_onnx(
                audio_encoder,
                audio_onnx_path,
                (dummy_input,),
                input_names=['input_features'],
                output_names=['audio_embeds'],
                dynamic_axes={
                    'input_features': {2: 'time'},
                    'audio_embeds': {1: 'seq_len'},
                },
                model_name='Audio Encoder'
            )

        # Convert to MNN
        if args.mnnconvert and os.path.exists(audio_onnx_path):
            convert_to_mnn(audio_onnx_path, audio_mnn_path, args.mnnconvert,
                          quant_bit=min(args.quant_bit, 8),
                          quant_block=args.quant_block)

        # Copy config for audio encoder
        audio_config_path = os.path.join(args.model_path, 'preprocessor_config.json')
        if os.path.exists(audio_config_path):
            import shutil
            shutil.copy2(audio_config_path, os.path.join(args.dst_path, 'preprocessor_config.json'))

    # =========================================
    # Export LLM Decoder
    # =========================================
    if args.export_decoder:
        print("\n" + "=" * 60)
        print("EXPORTING LLM DECODER")
        print("=" * 60)

        decoder = Qwen3Decoder(decoder_config)
        decoder.load_weights(state_dict)

        decoder_onnx_path = os.path.join(onnx_dir, 'llm.onnx')
        decoder_mnn_path = os.path.join(args.dst_path, 'llm.mnn')

        if not args.skip_onnx:
            # Export decoder with inputs_embeds support
            # The decoder takes: inputs_embeds (or input_ids), attention_mask, position_ids

            # For ONNX export we use inputs_embeds directly (as Qwen3-ASR uses prefix embeddings)
            B, S, D = 1, 32, 1024  # Static dims for export, dynamic axes for actual use
            dummy_embeds = torch.randn(B, S, D)
            dummy_attention_mask = torch.zeros(B, 1, S, S)
            dummy_position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)

            # Export full decoder (with tied embed_tokens + lm_head)
            # We use inputs_embeds so the ONNX model takes pre-computed embeddings
            class DecoderWrapper(nn.Module):
                def __init__(self, decoder):
                    super().__init__()
                    self.decoder = decoder

                def forward(self, inputs_embeds, attention_mask, position_ids):
                    return self.decoder(inputs_embeds=inputs_embeds,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids)

            wrapped_decoder = DecoderWrapper(decoder)

            export_onnx(
                wrapped_decoder,
                decoder_onnx_path,
                (dummy_embeds, dummy_attention_mask, dummy_position_ids),
                input_names=['inputs_embeds', 'attention_mask', 'position_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'inputs_embeds': {0: 'batch', 1: 'seq_len'},
                    'attention_mask': {0: 'batch', 2: 'seq_len', 3: 'seq_len'},
                    'position_ids': {0: 'batch', 1: 'seq_len'},
                    'logits': {0: 'batch', 1: 'seq_len'},
                },
                model_name='LLM Decoder'
            )

        # Convert to MNN
        if args.mnnconvert and os.path.exists(decoder_onnx_path):
            convert_to_mnn(decoder_onnx_path, decoder_mnn_path, args.mnnconvert,
                          quant_bit=args.quant_bit,
                          quant_block=args.quant_block)

            # Also export with separate weights if quantized
            if args.quant_bit < 16:
                # MNNConverter with separate weight for quantized models
                convert_to_mnn(decoder_onnx_path,
                              os.path.join(args.dst_path, 'llm.mnn.weight'),
                              args.mnnconvert,
                              quant_bit=args.quant_bit,
                              quant_block=args.quant_block)

        # Export embeddings in BF16 for MNN LLM engine
        if 'thinker.model.embed_tokens.weight' in state_dict:
            embed_weight = state_dict['thinker.model.embed_tokens.weight'].float().cpu().numpy()
            # Save as BF16 (uint16)
            import numpy as np
            embed_bf16 = np.frombuffer(embed_weight.tobytes(), dtype=np.uint32) >> 16
            embed_bf16 = embed_bf16.astype(np.uint16)
            embed_path = os.path.join(args.dst_path, 'embeddings_bf16.bin')
            with open(embed_path, 'wb') as f:
                f.write(embed_bf16.tobytes())
            print(f"\n  Saved: embeddings_bf16.bin ({embed_weight.shape})")

    # =========================================
    # Export LLM Decoder with KV Cache
    # =========================================
    if args.export_kv_cache and args.export_decoder:
        print("\n" + "=" * 60)
        print("EXPORTING LLM DECODER WITH KV CACHE")
        print("=" * 60)

        # Reuse the decoder loaded above (or load if skipped)
        if 'decoder' not in dir():
            decoder = Qwen3Decoder(decoder_config)
            decoder.load_weights(state_dict)

        kv_decoder = Qwen3DecoderWithKVCache(decoder)

        kv_onnx_path = os.path.join(onnx_dir, 'llm_kv.onnx')
        kv_mnn_path = os.path.join(args.dst_path, 'llm_kv.mnn')

        if not args.skip_onnx:
            B, S, D = 1, 16, 1024
            L, Hk, Hd = 28, 8, 128
            past_len = 1  # non-zero to avoid ONNX zero-dim edge cases

            dummy_embeds = torch.randn(B, S, D)
            dummy_pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
            dummy_mask = torch.zeros(B, 1, S, past_len + S).float()
            dummy_k_cache = torch.randn(L, B, Hk, past_len, Hd)
            dummy_v_cache = torch.randn(L, B, Hk, past_len, Hd)

            export_onnx(
                kv_decoder,
                kv_onnx_path,
                (dummy_embeds, dummy_pos, dummy_mask, dummy_k_cache, dummy_v_cache),
                input_names=['inputs_embeds', 'position_ids', 'attention_mask',
                             'k_cache', 'v_cache'],
                output_names=['logits', 'k_cache_out', 'v_cache_out'],
                dynamic_axes={
                    'inputs_embeds': {0: 'batch', 1: 'seq_len'},
                    'position_ids': {0: 'batch', 1: 'seq_len'},
                    'attention_mask': {0: 'batch', 2: 'seq_len', 3: 'total_len'},
                    'k_cache': {3: 'cache_len'},
                    'v_cache': {3: 'cache_len'},
                    'logits': {0: 'batch', 1: 'seq_len'},
                    'k_cache_out': {3: 'new_len'},
                    'v_cache_out': {3: 'new_len'},
                },
                model_name='LLM Decoder (KV Cache)'
            )

        if args.mnnconvert and os.path.exists(kv_onnx_path):
            convert_to_mnn(kv_onnx_path, kv_mnn_path, args.mnnconvert,
                          quant_bit=args.quant_bit,
                          quant_block=args.quant_block)
    print("\n" + "=" * 60)
    print("CREATING RUNTIME CONFIG")
    print("=" * 60)

    llm_config = {
        'model_type': 'qwen3_asr',
        'hidden_size': decoder_config.get('hidden_size', 1024),
        'layer_nums': decoder_config.get('num_hidden_layers', 28),
        'attention_mask': 'float',
        'attention_type': 'full',
        'is_mrope': True,
        'is_audio': True,
        'audio_type': 'qwen3_asr',
        'audio_start': 151669,
        'audio_end': 151670,
        'audio_pad': 151676,
        'attn_scale': 0.08838834764831845,  # 1/sqrt(128)
        'quant_bit': args.quant_bit,
        'vocab_size': decoder_config.get('vocab_size', 151936),
        'llm_model': 'llm.mnn',
        'llm_weight': 'llm.mnn.weight' if args.quant_bit < 16 else '',
        'embedding': 'embeddings_bf16.bin',
        'tie_embeddings': True,
    }

    config_path_out = os.path.join(args.dst_path, 'config.json')
    with open(config_path_out, 'w') as f:
        json.dump(llm_config, f, indent=2)
    print(f"  Saved: config.json")

    # Copy tokenizer
    print("\n" + "=" * 60)
    print("COPYING TOKENIZER")
    print("=" * 60)
    export_tokenizer(args.model_path, args.dst_path)

    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    for f in sorted(os.listdir(args.dst_path)):
        fp = os.path.join(args.dst_path, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  {f:30s} {size_mb:8.2f} MB")
        elif os.path.isdir(fp):
            print(f"  {f:30s} <DIR>")
    print("\nDone!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
