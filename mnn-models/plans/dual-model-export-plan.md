---
date: 2026-06-14
status: superseded
tags: [qwen3-asr, export, dual-model, plan, implementation]
category: plan
aliases: [双模型导出方案, Dual-Model Export Plan]
related: [[analysis/export-pipeline-analysis]], [[analysis/root-cause-analysis]], [[sherpa-ae-mnn-integration]], [[progress]], [[replicate-onnx-export-plan]]
---

# Qwen3-ASR 双模型导出方案（Fold-into-Batch · 已被替代）

> 日期：2026-06-14 | 状态：**superseded**（被 [[replicate-onnx-export-plan]] 替代）
>
> 目标：用我们自己的 llmexport.py 自控导出 conv_frontend + encoder 双模型，替换第三方 Wasser1462 ONNX。
>
> **为何未被采用**：此方案需要 C++ 侧新增 `valid_seq_len` Slice 逻辑（~15 行改动），且 encoder 不保留 attention_mask 参数。最终采用的复制 ONNX 方案 C++ 侧零改动，且 encoder 签名与 Wasser1462 ONNX 完全兼容（2 输入：features + attention_mask），方便对照调试。PE 修复在两方案中均适用。

## 背景

### 为什么需要双模型

1. **官方 Qwen3-ASR forward()** 包含 Chunk/Pad/Slice/Window 逻辑，`torch.onnx.export()` 无法 trace 这些动态操作（`torch.split` 产生 list、`pad_sequence` 变长、boolean mask 索引）
2. **当前单模型 audio.mnn**（简化版，无分块）在短音频（≤10s）表现良好，但长音频（≥16s）出现语音幻觉。根因：模型训练时分块 conv（每 100 mel 帧独立 conv），推理时连续 conv → 多 chunk 时中间表示偏离训练分布
3. **Wasser1462 双模型**（conv_frontend + encoder）在手机实测中完美支持长音频，但来自第三方不可控 ONNX

### 核心洞察

**不需要 `torch.split` 产生动态 List**。用 **Reshape → Batch 维度折叠（Fold into Batch）** 替代动态切分，将时间维度 (T) 的动态分块转换为空间维度 (Batch×N) 的并行计算。全程使用 ONNX 可 trace 的标准 op。

```
原始（不可 trace）:
  torch.split → [chunk_0, chunk_1, ..., chunk_N]  ← list[Tensor]，长度动态
  pad_sequence → [N, C, max_len]                   ← 变长 → 等长
  conv → [N, H, T']                                ← 每块独立

替代（可 trace）:
  F.pad → reshape [B×N, C, 100, 1]                 ← 展平成 batch 维
  conv → [B×N, H, 16, 13]                          ← 并行计算所有"块"
  reshape → permute → [B, H*16, N*13]              ← 恢复、拼接
```

## 架构设计

### 切分点

```
FBank [1, 128, T]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FrontendWrapper → conv_frontend.mnn             │
│  (1 input, 2 outputs)                            │
│                                                   │
│  1. N = (T + 99) // 100, pad_len = N*100 - T    │
│  2. torch.cat([x, zeros(B,C,pad_len)])  ← 动态  │
│  3. reshape [B*N, 1, 128, 100] (Fold into Batch) │
│  4. Conv2d×3 → [B*N, 480, 16, T_out_chunk]      │
│  5. Unfold → [B, 7680, enc_T_padded]             │
│  6. valid_seq_len: 官方公式处理 partial chunk    │
│                                                   │
│  输出: features [1, 7680, enc_T_padded]           │
│        valid_seq_len (scalar)                    │
└─────────────────────────────────────────────────┘
    │                        │
    │  features              │  valid_seq_len
    ▼                        ▼
┌───────────────────┐
│  omni.cpp          │  _Slice(features, [0,0,0], [-1,-1,valid])
│  截断幽灵帧          │  → [1, 7680, valid_enc_T]
└───────────────────┘
    │
    │  clean features (all valid)
    ▼
┌─────────────────────────────────────────────────┐
│  EncoderWrapper → encoder.mnn                    │
│  (1 input, 1 output — NO attention mask)         │
│                                                   │
│  1. Linear(7680 → 896) + bias=False              │
│  2. Sinusoidal PE (concat 模式)                   │
│  3. 18×Transformer (full bidirectional)          │
│  4. LayerNorm → Proj1 → GELU → Proj2             │
│                                                   │
│  输出: audio_embeds [1, valid_enc_T, 1024]        │
└─────────────────────────────────────────────────┘
    │
    ▼
_Permute({1,0,2}) → Decoder (llm.mnn)
```

### 与 omni.cpp 的对接

现有双模型路径完全复用，只需更新 config.json：

```json
{
    "audio_model": "conv_frontend.mnn",
    "audio_encoder": "encoder.mnn"
}
```

C++ 侧推理流程（已实现，无需修改）：
```
fbank → Permute({0,2,1}) → conv_frontend.mnn → encoder.mnn → decoder
```

唯一可能的增强：在 encoder 输出后做 `Slice [0, valid_seq_len, :]` 截掉 partial chunk 的幽灵帧。

## 实现步骤

### Step 1: 实现 FrontendWrapper

文件：`transformers/llm/export/utils/qwen3_asr_model.py`

核心修正：
- ❌ 不用 `F.pad(x, (0, pad_len))` —— pad tuple 不接受 Tensor，会断图
- ❌ 不硬编码 `13` —— 用动态 `out_len` 和官方 `_get_feat_extract_output_lengths`
- ✅ 用 `torch.cat([x, zeros])` + `ConstantOfShape` 实现动态 pad
- ✅ 单独计算 partial last chunk 的有效帧数，输出 `valid_seq_len` 供 C++ 截断

```python
class Qwen3ASRFrontend(nn.Module):
    """
    Conv frontend with chunked conv using Fold-into-Batch trick.
    Mathematically equivalent to official Chunk → Pad → Conv → Slice.

    Key design decisions (see plan doc risk analysis):
    1. torch.cat() instead of F.pad() — F.pad only accepts Python int tuple,
       which breaks the dynamic graph. Concat with dynamically-shaped zeros
       is fully traceable (ONNX ConstantOfShape).
    2. Dynamic out_len per chunk — NOT hardcoded 13. Uses the official
       _get_feat_extract_output_lengths formula: floor((L-1)/2)+1 per layer.
    3. valid_seq_len output — partial last chunk produces fewer valid frames.
       C++ side uses _Slice to truncate, so encoder receives only clean data
       (no attention mask needed in encoder at all).
    """
    def __init__(self, conv_hidden=480, chunk_size=100, conv_layers=3, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size   # n_window * 2 = 100
        self.conv_layers = conv_layers # 3
        self.conv1 = nn.Conv2d(1, conv_hidden, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_hidden, conv_hidden, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_hidden, conv_hidden, 3, stride=2, padding=1)

    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        """Official formula: L_out = floor((L - 1) / 2) + 1 per stride-2 conv layer."""
        for _ in range(3):  # 3 conv layers with stride=2
            input_lengths = (input_lengths.int() - 1) // 2 + 1
        return input_lengths  # dynamic Tensor, not Python int

    def forward(self, x):
        # x: [B, 128, T] mel-major, T is dynamic
        B, C, T_orig = x.shape
        chunk = self.chunk_size

        # ── Step 1: compute chunk count (dynamic, stays in graph) ──
        # Use integer arithmetic: (T + chunk - 1) // chunk  (avoids torch.ceil issues)
        # Do this with Python // on the SYMBOLIC T — PyTorch JIT traces it as dynamic Div+Add
        N = (T_orig + chunk - 1) // chunk
        pad_T = N * chunk
        pad_len = pad_T - T_orig  # dynamic scalar tensor

        # ── Step 2: dynamic pad via cat (F.pad breaks graph with Tensor arg) ──
        # ConstantOfShape is the standard ONNX idiom for dynamic zeros
        zeros_shape = torch.stack([B, C, pad_len])
        zeros = torch.zeros(zeros_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], dim=-1)  # [B, C, pad_T]

        # ── Step 3: Fold into Batch ──
        # [B, C, N*chunk] → [B, C, N, chunk] → [B, N, C, chunk] → [B*N, C, chunk]
        x = x.reshape(B, C, N, chunk)          # reshape: dynamic N in graph
        x = x.permute(0, 2, 1, 3)               # [B, N, C, chunk]
        x = x.reshape(B * N, 1, C, chunk)       # fold N into batch dim
        # x: [B*N, 1, C=128, chunk=100]

        # ── Step 4: Conv2d×3 ──
        # Each stride-2 conv: L_out = floor((L-1)/2) + 1
        # Input C=128, chunk=100:
        #   conv1: C→16, T→50   conv2: C→32, T→25   conv3: C→16, T→13
        # → output spatial dims depend on C (frequency dim), not just T
        x = F.gelu(self.conv1(x))   # [B*N, H, 64, 50]
        x = F.gelu(self.conv2(x))   # [B*N, H, 32, 25]
        x = F.gelu(self.conv3(x))   # [B*N, H, 16, T_out_chunk]
        # T_out_chunk is DYNAMIC — don't hardcode! It depends on chunk_size
        # For chunk=100: (100-1)//2+1=50, (50-1)//2+1=25, (25-1)//2+1=13

        _, H_conv, H_freq, T_out_chunk = x.shape
        # H_conv = 480 (conv_hidden), H_freq = 16 (128/8), T_out_chunk = dynamic

        # ── Step 5: Unfold from Batch ──
        # [B*N, H_conv, H_freq, T_out_chunk] → [B, N, H_conv, H_freq, T_out_chunk]
        x = x.reshape(B, N, H_conv, H_freq, T_out_chunk)
        # → [B, H_conv, H_freq, N, T_out_chunk]
        x = x.permute(0, 2, 3, 1, 4)
        # → [B, H_conv * H_freq, N * T_out_chunk] = [B, 7680, enc_T_padded]
        enc_T_padded = N * T_out_chunk
        x = x.reshape(B, H_conv * H_freq, enc_T_padded)
        # x: [B, 7680, enc_T_padded]

        # ── Step 6: compute valid_seq_len ──
        # Full chunks: each produces T_out_chunk valid frames
        # Last partial chunk: _get_feat_extract_output_lengths(last_chunk_len) valid frames
        last_chunk_len = T_orig - (N - 1) * chunk  # dynamic
        valid_last = self._get_feat_extract_output_lengths(last_chunk_len)  # dynamic
        valid_seq_len = (N - 1) * T_out_chunk + valid_last  # dynamic scalar

        return x, valid_seq_len
```

**关键点**：
- `N` 和 `pad_len` 全程在 Tensor 图中计算，不调用 `.item()` 或 `int()`
- `T_out_chunk` 从 conv3 输出 shape 动态获取，不硬编码
- `valid_seq_len` 精确处理 partial last chunk，传给 C++ 做 Slice

### Step 2: 实现 EncoderWrapper

**关键修正**：Encoder 不再接收 `attention_mask`。因为 C++ 侧在喂入 Encoder 之前已经用 `_Slice` 把幽灵帧物理截断了，Encoder 拿到的全是有效数据。

```python
class Qwen3ASREncoder(nn.Module):
    """
    Pure transformer encoder. NO attention mask — valid_seq_len truncation
    is done in C++ (omni.cpp) via _Slice BEFORE feeding into this model.
    """
    def __init__(self, d_model=896, n_layers=18, output_dim=1024, **kwargs):
        super().__init__()
        self.conv_out = nn.Linear(7680, d_model, bias=False)
        self.encoder_layers = nn.ModuleList([...])  # 18× TransformerLayer
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model * 4)
        self.proj2 = nn.Linear(d_model * 4, output_dim)

    def forward(self, x):
        # x: [B, 7680, valid_enc_T] — already truncated by C++, all clean
        x = x.transpose(1, 2)           # [B, valid_enc_T, 7680]
        x = self.conv_out(x)            # [B, valid_enc_T, d_model]
        x = x + self._create_sinusoidal_positions(
            x.size(1), x.size(2)
        )
        for layer in self.encoder_layers:
            x = layer(x)                # full bidirectional, no mask needed
        x = self.ln_post(x)
        x = self.proj2(F.gelu(self.proj1(x)))
        return x                        # [B, valid_enc_T, output_dim]
```

### Step 3: 修改导出脚本

`transformers/llm/export/utils/audio.py` → `Qwen3AsrAudio.export()`:

```python
def export(self, onnx_dir):
    # Export frontend (1 input, 2 outputs)
    frontend_path = os.path.join(onnx_dir, "conv_frontend.onnx")
    dummy_input = torch.randn(1, 128, 300)  # T=300 > 200 so multi-chunk
    torch.onnx.export(
        self.frontend,
        dummy_input,
        frontend_path,
        input_names=["input_features"],
        output_names=["features", "valid_len"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time"},
            "features": {0: "batch", 2: "enc_time_padded"},
        },
        opset_version=14,
    )

    # Export encoder (1 input, 1 output — NO attention_mask!)
    encoder_path = os.path.join(onnx_dir, "encoder.onnx")
    dummy_features = torch.randn(1, 7680, 39)
    torch.onnx.export(
        self.encoder,
        dummy_features,                    # single input, not tuple
        encoder_path,
        input_names=["features"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "features": {0: "batch", 2: "valid_enc_time"},
            "audio_embeds": {0: "batch", 1: "valid_enc_time"},
        },
        opset_version=14,
    )
```

**导出验证**：
- 导出后用不同 T 的输入跑一遍 ONNX Runtime，确认输出帧数随 T 变化
- Netron 打开检查图中无硬编码的 N 或 T_out_chunk 常量

### Step 4: omni.cpp 适配

当前双模型路径骨架已存在（`conv_frontend → encoder`），需要两个改动：

**改动 1**：conv_frontend 有两个输出（features + valid_len），需要分别提取。
**改动 2**：用 valid_len 截断 features 再喂入 encoder。

```cpp
// ── qwen3_asr two-model path (self-controlled export) ──
if (audio_type == "qwen3_asr" && mAudioEncoder.get() != nullptr) {
    // Step 1: FBank [1, 128, T] → Permute({0,2,1}) → [1, T, 128]
    auto ae_input = _Permute(input_features, {0, 2, 1});

    // Step 2: conv_frontend (1 input → 2 outputs: features + valid_len)
    auto frontend_outputs = mAudioModule->onForward({ae_input});
    VARP features = frontend_outputs[0];    // [1, 7680, enc_T_padded]
    VARP valid_len = frontend_outputs[1];   // scalar (or 1D tensor)

    // Step 3: Truncate ghost frames from partial last chunk
    int valid = valid_len->readMap<int>()[0];
    int padded_T = features->getInfo()->dim[2];
    if (valid < padded_T) {
        features = _Slice(features,
            _var<int>({0, 0, 0}, {3}),
            _var<int>({-1, -1, valid}, {3}));
    }
    // features: [1, 7680, valid] — all clean data

    // Step 4: encoder (1 input — no attention mask)
    auto enc_outputs = mAudioEncoder->onForward({features});
    audio_embedding = enc_outputs[0];
    // audio_embedding: [1, valid, 1024]
}
```

**C++ 改动量**：~15 行（在现有 `if (mAudioEncoder)` 分支内替换 conv 调用逻辑）。现有双模型路径的 Permute → encoder 调用 → 结果处理框架全部复用。

## 潜在风险与对策

### 风险 1: torch.cat 的动态 shape 推导 ⚠️ MEDIUM

**现象**：`torch.zeros(zeros_shape)` 中的 `zeros_shape = torch.stack([B, C, pad_len])`，其中 `pad_len` 是动态 Tensor。在某些 PyTorch/ONNX 版本组合中，`torch.zeros` 可能无法正确推导符号化 shape。

**对策**：
- 导出前用 `torch.onnx.export(..., dynamo=True)` 或调整 opset 到 17+
- 备选方案：用 `x[:, :, :pad_T]` 的 padding 模式，或显式 `onnx::ConstantOfShape`
- 导出后 Netron 检查 `ConstantOfShape` 节点的 shape 输入是否为动态的 `Shape` + `Concat` 而非 `Constant`

### 风险 2: 幽灵帧导致 Decoder 幻觉 ⚠️ → ✅ 已解决

**原现象**：Partial chunk 的 pad 帧产生"废数据"，Decoder cross-attention 扫描到 → 幻觉。

**对策（已融入设计）**：
- FrontendWrapper 输出 `valid_seq_len`（scalar）
- C++ 侧在 encoder 输入前 `_Slice([0, valid_seq_len])` **物理截断**，而非用 attention mask 软屏蔽
- Encoder 完全不接触无效帧 → 输出 100% 干净
- 这比 attention mask 方案更可靠：mask 只能让 attention 忽略幽灵帧，但幽灵帧本身仍然被 transformer 的 FFN 处理并影响后续层的 hidden states

### 风险 3: Permute/Reshape 引入 MNN 性能开销 ⚠️ MEDIUM

**现象**：Fold-into-Batch 的 `permute` + `reshape` 在 PyTorch 中只改变 stride，但在 MNN 底层可能翻译为真实内存搬移。

**对策**：
- 导出前在关键 permute 后加 `.contiguous()` 减少碎片化 stride
- 用 Netron 检查 ONNX 图中 Transpose 节点数量
- 导出后用 MNNConvert 的图优化（`--keepInputFormat` / `--optimizeLevel`）处理
- 如果实测性能不达标，考虑在 MNN 侧加 geometry pass 消除冗余 Transform

### 风险 4: torch.ceil 的 ONNX 符号化 ⚠️ MEDIUM

**现象**：`torch.ceil(tensor)` 在某些 opset 版本或 ONNX Runtime 中映射为 `Ceil` op，但符号化 shape inference 可能不传播动态性。

**对策**：
- 使用 `torch.div(tensor, chunk_size, rounding_mode='trunc')` 等更底层 op 组合替代 ceil
- 测试 opset_version=14 和 opset_version=17 的行为差异
- 如果 ceil 符号化失败，可用 `(T + chunk_size - 1) / chunk_size`（整数除法）替代

### 风险 5: 最后一帧 conv 边界效应 ⚠️ LOW

**现象**：F.pad 填充的帧是零值，conv 在 pad 边界的行为与真实音频帧边界不同。这是 chunking 固有行为，也是训练时的行为，因此是正确的。

**对策**：确认 padding_mode='constant'（零填充），与官方 `pad_sequence` 的默认行为一致。

## 验证计划

| 阶段 | 验证内容 | 方法 |
|------|---------|------|
| 1 | PyTorch 单 chunk vs Fold-to-Batch | 同一随机输入，max diff < 1e-6 |
| 2 | PyTorch multi-chunk vs 官方 forward | 同一随机输入（>100 帧），对比输出帧数和值 |
| 3 | ONNX export 成功 + 动态轴检查 | Netron 确认无硬编码维度 |
| 4 | MNNConvert 转换 + cosim 验证 | conv_frontend cosim vs PyTorch > 0.999 |
| 5 | 端到端 ONNX RT vs MNN 双模型 | AE 输出 cosim > 0.99，first token 一致 |
| 6 | 手机实机长音频测试 | ≥16s 音频，对比 Wasser1462 结果 |

## 文件变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `qwen3_asr_model.py` | 新增 `Qwen3ASRFrontend` + 重构 `Qwen3ASREncoder` | Fold-into-Batch conv, 双模型 wrapper |
| `audio.py` | `Qwen3AsrAudio.export()` 导出两个 ONNX | conv_frontend.onnx + encoder.onnx |
| `llmexport.py` | 适配双模型导出流程 | 可能无需改动 |
| `omni.cpp` | （可选）valid_seq_len Slice | 截断幽灵帧 |
| `config.json` | `audio_model: "conv_frontend.mnn"`, `audio_encoder: "encoder.mnn"` | 启用双模型路径 |

## 参考

- 官方 modeling：`github.com/QwenLM/Qwen3-ASR` → `modeling_qwen3_asr.py`
- 当前单模型方案：[[analysis/export-pipeline-analysis]]
- Wasser1462 双模型方案：[[sherpa-ae-mnn-integration]]
- 根因分析（已更正）：[[analysis/root-cause-analysis]]
- 项目进度：[[progress]]
