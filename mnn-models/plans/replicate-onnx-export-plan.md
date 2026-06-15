---
date: 2026-06-14
status: completed
tags: [qwen3-asr, export, dual-model, plan, implementation, replicate-onnx]
category: plan
aliases: [照抄ONNX导出方案, Replicate ONNX Export Plan, 防御性重构]
related: [[analysis/export-pipeline-analysis]], [[dual-model-export-plan]], [[progress]]
---

# Qwen3-ASR 照抄 ONNX 拆分方案（防御性重构 · 已完成）

> 日期：2026-06-14 制定，2026-06-15 完成 | 状态：**completed**
>
> 策略：以已验证的 Wasser1462 双模型 ONNX 为"设计说明书"，用我们自己的 PyTorch 代码复刻相同输入输出签名的两个模型，实现全链路自控。

## 为什么选择"照抄"而非 Fold-into-Batch 创新

| 维度 | Fold-into-Batch（创新） | 照抄 ONNX（防御） |
|------|:---|:---|
| C++ 端改动 | 需新增 valid_seq_len 提取 + Slice 逻辑 | **0 改动** — 现有双模型路径完全复用 |
| MNNConvert 兼容性 | 未知（新算子组合可能触发 Layout Transform 性能惩罚） | **已验证** — cosim 1.0，MNN 完美支持 |
| 维度推导风险 | 手动计算 stride math + partial chunk 边界 | Wasser1462 已算好并验证通过 |
| 幽灵帧处理 | C++ Slice 物理截断 | ONNX 图内部 Slice，自动截断 |
| 实现难度 | 高（需处理符号化 shape 推导） | 中（对着图写 PyTorch） |

## 目标 ONNX 规格（来自 Wasser1462，已验证）

### conv_frontend.onnx

| 属性 | 值 |
|------|-----|
| 节点数 | 130 |
| opset | 17 |
| **输入** | `input_features`: `[batch, n_frames, 128]` ← 时间优先 |
| **输出** | `conv_output`: `[batch, n_audio_tokens, hidden_dim]` |
| 输出特点 | 全部为有效帧（Pad/Slice 在图内部完成，无幽灵帧） |

图结构（Netron 解剖结果）：

```
input_features [B, T, 128]
    │
    ├─ Shape → Gather (batch, n_frames, 128)
    │
    ├─ [0:34] 动态 pad 量计算:
    │     Mod: n_frames % chunk_size
    │     Sub: chunk_size - (n_frames % chunk_size)
    │     Mod: 结果 % chunk_size  →  pad_len
    │     ConstantOfShape → pad tensor [0,0,0,0,0,pad_len]
    │
    ├─ [46] Pad → [B, T+pad_len, 128]
    │
    ├─ [60] Reshape: Fold-into-Batch
    │     [B, T+pad_len, 128] → [B*N, ...]
    │
    ├─ Conv2d × 3 (k=3, s=2, p=1) + GELU
    │     Conv1: [B*N, 480, 50, 64]
    │     Conv2: [B*N, 480, 25, 32]
    │     Conv3: [B*N, 480, 13, 16]
    │
    ├─ Transpose + Reshape: Unfold → [B, T_enc_padded, 7680]
    │
    ├─ Shape→Gather→Slice: 截断 pad 帧 → [B, T_enc_valid, 7680]
    │
    └─ MatMul (conv_out): [B, T_enc_valid, hidden]=896
            ↓
        conv_output [B, n_audio_tokens, 896]
```

### encoder.int8.onnx

| 属性 | 值 |
|------|-----|
| 节点数 | 1887 |
| opset | 17 |
| **输入 1** | `input_features`: `[batch, n_audio_tokens, 896]` |
| **输入 2** | `feature_attention_mask`: `[batch, n_audio_tokens]` |
| **输出** | `audio_features`: `[batch, n_audio_tokens, 1024]` |

内部结构：Sinusoidal PE → 18× Transformer（含 attention mask）→ LayerNorm → Proj1 → GELU → Proj2

**关键**：encoder 接收 attention_mask（用于 Transformer 内部的多头注意力），但前端输出已不含幽灵帧，所以 mask 通常全为 1。

## 实现方案

> **注意**：以下 Step 1/2 代码为 06-14 原始计划。实际实施中对 `Qwen3ASREncoder` 的 PE 做了关键修正（从连续递增改为按 chunk 重复 0..12），并移除了 ghost frame Slice（与 Wasser1462 行为对齐）。最终实现见 `qwen3_asr_model.py` 中的 `Qwen3ASRConvFrontend` 和 `Qwen3ASREncoder` 类，以及下方 § 实施结果。

### Step 1: 实现 Qwen3ASRConvFrontend

文件：`transformers/llm/export/utils/qwen3_asr_model.py`

```python
class Qwen3ASRConvFrontend(nn.Module):
    """
    Self-controlled conv frontend for Qwen3-ASR.
    Reproduces the same I/O signature as the verified Wasser1462 conv_frontend.onnx:

      Input:  [B, T, 128]     (time-major mel features)
      Output: [B, T', 896]    (all valid frames, pad/slice handled internally)

    Key design decisions:
    - torch.cat() for dynamic padding (F.pad breaks graph with Tensor args)
    - Fold-into-Batch to parallelize chunked conv (same technique as verified ONNX)
    - Internal Slice to remove ghost frames (no valid_len output needed)
    """

    def __init__(self, chunk_size=100, conv_hidden=480, d_model=896, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Conv2d×3: k=3, s=2, p=1  (same as official & Wasser1462)
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

    def forward(self, x):
        # x: [B, T, 128]  time-major (consistent with omni.cpp Permute)
        B, T, C = x.shape
        chunk = self.chunk_size

        # ── Step 1: Dynamic pad to multiple of chunk_size ──
        # pad_len = (chunk - T % chunk) % chunk  (kept in tensor graph)
        pad_len = (chunk - T % chunk) % chunk

        # Safe dynamic padding: Concat instead of F.pad.
        # Primary approach: torch.zeros with dynamic pad_len → ONNX ConstantOfShape.
        # If torch.zeros throws with dynamic Tensor size (some PyTorch/ONNX combos), 
        # fall back to: static_zeros.expand(B, pad_len, C).
        zeros = torch.zeros(B, pad_len, C, dtype=x.dtype, device=x.device)
        # Fallback (uncomment if the above fails during export):
        # zeros = torch.zeros(1, 1, C, dtype=x.dtype, device=x.device).expand(B, pad_len, C)
        x = torch.cat([x, zeros], dim=1)  # [B, T_pad, 128], T_pad = T + pad_len
        T_pad = T + pad_len

        # ── Step 2: Fold-into-Batch ──
        N = T_pad // chunk  # integer, derived from shape — traceable
        # [B, N*chunk, 128] → [B, N, chunk, 128] → [B*N, chunk, 128]
        x = x.reshape(B, N, chunk, C)
        x = x.permute(0, 2, 3, 1)              # [B, chunk, 128, N]
        x = x.reshape(B, 1, C, N * chunk)       # BATCH fold happens here
        # → Conv2d expects [B*N, 1, 128, chunk]
        # Need: [B, 1, 128, N*chunk] → reshape to fold N
        # Actually: reshape B×N into batch: [B*N, 1, C, chunk]
        x = x.reshape(B * N, 1, C, chunk)       # CORRECT fold

        # ── Step 3: Conv2d×3 (parallel across all "chunks") ──
        x = F.gelu(self.conv1(x))   # [B*N, 480, 64, 50]
        x = F.gelu(self.conv2(x))   # [B*N, 480, 32, 25]
        x = F.gelu(self.conv3(x))   # [B*N, 480, 16, T_out_chunk]

        _, H_conv, H_freq, T_out_chunk = x.shape
        # H_conv=480, H_freq=16, T_out_chunk=(chunk-1)//2+1 三层后

        # ── Step 4: Unfold ──
        # [B*N, 480, 16, T_out] → [B, N, 480, 16, T_out]
        x = x.reshape(B, N, H_conv, H_freq, T_out_chunk)
        # → [B, 480, 16, N, T_out] → flatten
        x = x.permute(0, 2, 3, 1, 4)             # [B, 480, 16, N, T_out_chunk]
        x = x.reshape(B, H_conv * H_freq, N * T_out_chunk)
        # → [B, 7680, N * T_out_chunk]
        x = x.permute(0, 2, 1)                    # [B, enc_T_padded, 7680]

        # ── Step 5: Slice out ghost frames from partial last chunk ──
        # Full chunks: N-1 full chunks → (N-1) * T_out_chunk valid frames
        # Last partial chunk: official formula on (T_orig - (N-1)*chunk) frames
        T_orig_last_chunk = T - (N - 1) * chunk
        valid_last = self._get_feat_extract_output_lengths(T_orig_last_chunk)
        valid_total = (N - 1) * T_out_chunk + valid_last

        # 🚨 CRITICAL: valid_total must stay as a Tensor — NO int() or .item()!
        # Calling int(valid_total) triggers .item(), which breaks the ONNX trace graph
        # and bakes the dummy input's value into the exported model.
        x = x[:, :valid_total, :]  # Tensor slice → dynamic ONNX Slice op

        # ── Step 6: Linear projection ──
        x = self.conv_out(x)  # [B, T_enc_valid, d_model=896]

        return x
```

**关键点**：
- `pad_len = (chunk - T % chunk) % chunk` 全程在 tensor 图中，不调用 `.item()`
- `torch.zeros(B, pad_len, C)` 中 `pad_len` 是符号化整数 → ONNX `ConstantOfShape`
- `T_out_chunk` 从 conv3 输出动态获取
- `valid_total` 精确处理 partial chunk，最终 Slice 只输出有效帧
- 输出无幽灵帧 → encoder 拿到的是干净数据

### Step 2: 实现 Qwen3ASREncoder

```python
class Qwen3ASREncoder(nn.Module):
    """
    Self-controlled transformer encoder.
    Same I/O signature as verified Wasser1462 encoder.int8.onnx:

      Input 1: input_features [B, n_audio_tokens, 896]
      Input 2: feature_attention_mask [B, n_audio_tokens]
      Output:  audio_features [B, n_audio_tokens, 1024]

    The attention_mask is passed through to transformer layers for correctness,
    though in practice all values are 1 (frontend already removed ghost frames).
    """
    def __init__(self, d_model=896, n_layers=18, n_heads=14,
                 ffn_dim=3584, output_dim=1024, **kwargs):
        super().__init__()
        # Positional embedding (sinusoidal, concat mode)
        self.max_pe_len = 5000
        pe = self._create_sinusoidal_positions(self.max_pe_len, d_model)
        self.register_buffer("positional_embedding", pe, persistent=False)

        # 18× Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model * 4)
        self.proj2 = nn.Linear(d_model * 4, output_dim)

    @staticmethod
    def _create_sinusoidal_positions(max_len, dim):
        """Whisper-style: [all_sins | all_cosines] — NOT interleaved."""
        if dim % 2 != 0:
            raise ValueError(f"dim {dim} must be even")
        log_inc = torch.log(torch.tensor(10000.0)) / (dim // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(dim // 2, dtype=torch.float))
        scaled = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * inv.unsqueeze(0)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
        return pe

    def forward(self, input_features, attention_mask=None):
        # x: [B, n_audio_tokens, 896]
        seq_len = input_features.size(1)

        # Add PE (slice to match seq_len)
        pe = self.positional_embedding[:seq_len, :]
        x = input_features + pe

        # 18× Transformer
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x  # [B, n_audio_tokens, 1024]
```

**与 Wasser1462 encoder 对齐要点**：
- PE 使用 pre-computed buffer + slicing（避免在 ONNX 图中动态生成 PE，减少节点数）
- attention_mask 参数保留（匹配 ONNX 的 2-input 签名）
- 输出维度 [B, T', 1024] — 与 LLM decoder 的 hidden_size 对齐

### Step 3: 导出脚本

`transformers/llm/export/utils/audio.py` → `Qwen3AsrAudio.export()`:

```python
def export(self, onnx_dir):
    # ── Export conv_frontend ──
    frontend = self.frontend.eval()
    dummy = torch.randn(1, 300, 128)  # T > 2*chunk to verify multi-chunk
    torch.onnx.export(
        frontend,
        dummy,
        os.path.join(onnx_dir, "conv_frontend.onnx"),
        input_names=["input_features"],
        output_names=["conv_output"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "n_frames"},
            "conv_output": {0: "batch", 1: "n_audio_tokens"},
        },
        opset_version=17,
    )

    # Verify: run with T=100, 200, 350 and check output frames match official formula
    for T_test in [100, 200, 350]:
        with torch.no_grad():
            out = frontend(torch.randn(1, T_test, 128))
            expected = self._expected_output_frames(T_test)
            assert out.shape[1] == expected, \
                f"T={T_test}: got {out.shape[1]}, expected {expected}"

    # ── Export encoder ──
    encoder = self.encoder.eval()
    dummy_feat = torch.randn(1, 39, 896)      # dynamic T'
    dummy_mask = torch.ones(1, 39)             # dynamic T'
    torch.onnx.export(
        encoder,
        (dummy_feat, dummy_mask),
        os.path.join(onnx_dir, "encoder.onnx"),
        input_names=["input_features", "feature_attention_mask"],
        output_names=["audio_features"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "n_audio_tokens"},
            "feature_attention_mask": {0: "batch", 1: "n_audio_tokens"},
            "audio_features": {0: "batch", 1: "n_audio_tokens"},
        },
        opset_version=17,
    )
```

### Step 4: C++ 端（omni.cpp）— 无需改动！

现有双模型路径（`mAudioEncoder != nullptr`）直接复用：

```
fbank → Permute({0,2,1}) → conv_frontend.mnn → encoder.mnn → Permute({1,0,2}) → decoder
```

配置：
```json
{
    "audio_model": "conv_frontend.mnn",
    "audio_encoder": "encoder.mnn"
}
```

**与 Wasser1462 版本的差异**：文件由我们的 llmexport.py 生成而非第三方下载。C++ 代码行数：0。

## 验证计划

| 阶段 | 验证内容 | 通过标准 |
|------|---------|---------|
| 1 | PyTorch frontend 帧数验证 | T=100→13, T=200→25, T=350→44, T=500→63 (单chunk N=1)；T=800→101, T=1600→202 (多chunk N≥2) |
| 2 | PyTorch frontend vs Wasser1462 conv_frontend ONNX | cosim > 0.999（同 T 同输入） |
| 3 | ONNX export 成功 + 动态轴 | Netron 检查无硬编码 T 值；用不同 T 跑 ONNX RT 输出帧数正确 |
| 4 | MNNConvert → conv_frontend.mnn + encoder.mnn | 转换成功，无报错 |
| 5 | MNN vs PyTorch cosim | conv_frontend cosim > 0.99, encoder cosim > 0.99 |
| 6 | 桌面端双模型串联 | AE 输出 cosim vs Wasser1462 > 0.99 |
| 7 | 手机实机长音频 | ≥16s 音频识别正常，无幻觉 |

## 潜在风险与对策

### 风险 1: torch.zeros 的动态 shape 推导 ⚠️

**现象**：`torch.zeros(B, pad_len, C)` 中 `pad_len` 是动态值（如 `(chunk_size - T % chunk_size) % chunk_size`）。当 `pad_len` 是 0-d Tensor 时，`torch.onnx.export` 需要将其映射为 `ConstantOfShape`，但某些 PyTorch 版本会抛出 `TypeError: size must be int`。

**对策**（两层防御）：

**方案 A（首选）**：直接用 `torch.zeros(B, pad_len, C)` — opset 17 + PyTorch 2.x 通常支持。

**方案 B（防弹降级）**：用 expand 绕过 shape 类型检查：

```python
# expand 在 ONNX 中映射为 Expand op，完美接收动态 Tensor shape
static_zeros = torch.zeros(1, 1, C, dtype=x.dtype, device=x.device)
zeros = static_zeros.expand(B, pad_len, C)
x = torch.cat([x, zeros], dim=1)
```

原理：`torch.zeros(1, 1, C)` 的 shape 是 Python int → 必定成功；`.expand(B, pad_len, C)` 中 `pad_len` 作为 Tensor 传给 ONNX Expand op → 动态 shape 保留在图内。

**验证**：
- 导出后 Netron 检查：`ConstantOfShape`（或 `Expand`）的 shape 输入应为动态的 `Shape → Mod → Sub` 链，而非 `Constant`
- 用不同 T 的输入跑 ONNX RT，确认 padding 量自适应

### 风险 2: MNNConvert 与 opset 17 的兼容性 ⚠️ LOW

**现象**：Wasser1462 ONNX 使用 opset 17。MNNConvert 可能不完全支持 opset 17 引入的某些算子变体。

**对策**：
- Wasser1462 已验证 MNNConvert 对其 opset 17 ONNX 的转换正确（cosim 1.0/0.997）
- 我们的图结构与之高度相似 → 兼容性风险极低
- 如果遇到问题，尝试降级到 opset 14

### 风险 3: PE slicing 在 ONNX 中的符号化 ⚠️ LOW

**现象**：`pe = self.positional_embedding[:seq_len, :]` 的 slicing 在 ONNX 中变为 `Slice` op。需要确保 `seq_len` 是动态的。

**对策**：
- `seq_len = input_features.size(1)` 在 ONNX 图中是 `Shape → Gather` → 动态值
- `Slice` 的 end 参数由这个动态值提供 → 完全可 trace
- 验证：导出后用不同 T 的输入跑 ONNX RT，确认 PE 长度自适应

### 风险 4: chunk_size 与训练配置辨析 ⚠️ RESOLVED（2026-06-15 确认 chunk=100）

**原疑虑**：`Qwen3ASRConvFrontend` 的 `chunk_size` 应与 HF config 中的哪个参数对齐？`conv_chunksize=500` 还是 `n_window*2=100`？

**确认结论**：chunk_size = **100**（`n_window * 2`）。`conv_chunksize=500` 是官方代码中 conv 循环的**批处理大小**（每次并行处理最多 500 个 chunk，防 OOM），不是 mel 帧切分的 chunk 大小。chunk_size=100 与 Wasser1462 ONNX 实际行为完全一致（帧数 100% 匹配），与官方 `torch.split(chunk_lengths)` 的语义对齐。

## 文件变更

| 文件 | 变更 | 说明 |
|------|------|------|
| `qwen3_asr_model.py` | 新增 `Qwen3ASRConvFrontend` + 重构 `Qwen3ASREncoder` | 双模型 wrapper |
| `audio.py` | `export()` 导出两个 ONNX | conv_frontend.onnx + encoder.onnx |
| `llmexport.py` | 适配双模型导出 | 按需调整 |
| `omni.cpp` | **无需修改** | 双模型路径已有 |
| `config.json` | `audio_model: "conv_frontend.mnn"`, `audio_encoder: "encoder.mnn"` | 配置 |

## 与 Fold-into-Batch 方案的对比

| | Fold-into-Batch 方案 | 照抄 ONNX 方案（本方案） |
|------|:---|:---|
| C++ 改动 | ~15 行（valid_len 提取 + Slice） | **0 行** |
| 幽灵帧处理 | C++ 侧物理 Slice | ONNX 图内 Slice（Wasser1462 已验证） |
| Encoder attention_mask | 不需要（参数更少） | 保留（匹配 ONNX 签名，方便迁移） |
| 实现风险 | 新算子组合，需全链路验证 | 对标已验证结构，渐进替换 |
| 推荐场景 | 后续优化（有空再研究） | **当前生产落地** |

## 实施结果（2026-06-15 完成）

### 发现 & 修复的 Bug

经过三轮调试，发现了三个关键 train-inference 不匹配：

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | **chunk_size 误配** | 计划代码用 `chunk_size=100`，后误改为 500（匹配 `conv_chunksize`）。Wasser1462 ONNX 实测使用 chunk=100 | chunk_size 固定为 100 |
| 2 | **Ghost frame Slice** | 我们裁剪了 partial chunk 的 ghost frames (`valid_total`)，Wasser1462 保留全部 `N × f³(chunk)` 帧 | 移除 Slice，输出全量帧 |
| 3 | **PE 模式错误（主因）** | 训练时 PE 按 chunk 重复 0..12；我们连续递增 0..seq_len-1。长音频 PE 外推 15×+ 远超训练分布 → 注意力崩溃 → 幻觉 | Encoder PE 改为 `repeat(PE[0:13], N)[:seq_len]` |

### 最终验证

| 指标 | 结果 |
|------|:--:|
| 帧数 vs Wasser1462 (T=50~1600) | 100% 匹配 |
| PyTorch vs Wasser1462 ONNX cosim | 1.000000 |
| MNN E2E cosim @ T=1600 | 0.9996 |
| 手机短句识别 | ✅ 正常 |
| 手机长句识别 (≥16s) | ✅ **修复** |

### 产出文件

```
Qwen3-ASR-0.6B-Omni-INT8/
├── conv_frontend.mnn         14 KB   (graph, chunk=100, fold-into-batch)
├── conv_frontend.mnn.weight  11 MB   (INT8 weights)
├── encoder.mnn              335 KB   (graph, repeating PE 0..12 per chunk)
├── encoder.mnn.weight       189 MB   (INT8 weights)
├── llm.mnn                  494 KB   (decoder)
├── llm.mnn.weight           604 MB   (decoder INT8 weights)
├── config.json → audio_model: "conv_frontend.mnn", audio_encoder: "encoder.mnn"
└── tokenizer.txt              3 MB
```

### 已知局限

- **Windowed attention 未实现**：官方训练使用 104 帧分窗注意力，当前使用全序列双向注意力。PE 修复已大幅改善长音频，windowed attention 作为后续优化
- **encoder.mnn 使用 MNNConvert INT8 量化（FP32 图 + INT8 权重）**，不同于 Wasser1462 的原生 INT8 推理图（DynamicQuantizeLinear + MatMulInteger）。MNNConvert 量化精度经 cosim 验证可用

## 参考

- Wasser1462 ONNX: `/Users/bxy/Documents/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/`
- 官方 modeling: `github.com/QwenLM/Qwen3-ASR`
- 单模型方案: [[analysis/export-pipeline-analysis]]
- Fold-into-Batch 方案: [[dual-model-export-plan]]
- 项目进度: [[progress]]
