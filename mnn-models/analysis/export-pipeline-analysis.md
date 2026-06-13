---
date: 2026-06-14
status: active
tags: [qwen3-asr, analysis, export, modeling, pipeline]
category: analysis
aliases: [导出链路分析, Export Pipeline Analysis]
related: [[root-cause-analysis]], [[sherpa-ae-mnn-integration]], [[progress]]
---

# Qwen3-ASR 导出链路分析

> 日期：2026-06-14 | 状态：active
>
> 发现官方 modeling 代码后，对当前导出链路和"正道"的重评估。

## 背景

2026-06-14 确认：Qwen3-ASR-0.6B 的 **modeling 代码是开源的**，在独立 GitHub 仓库 `github.com/QwenLM/Qwen3-ASR` 中，包含完整的 `modeling_qwen3_asr.py`（`Qwen3ASRAudioEncoder.forward()` ~80 行）。

但 HuggingFace / ModelScope 模型仓库（`Qwen/Qwen3-ASR-0.6B`）**只有权重 + config，没有 Python 代码**。我们做迁移时只看了模型仓库，不知道独立代码仓库的存在，因此从 safetensors 权重反推了 `qwen3_asr_model.py`，手写 `forward()` 漏掉了 Chunk/Pad/Slice/Window 四个关键步骤。

## 官方 forward() 的完整逻辑

`github.com/QwenLM/Qwen3-ASR` 中的 `Qwen3ASRAudioEncoder.forward()` 实际流程：

```
FBank [1, 128, T]
    │
    ├── Chunking:  按 n_window*2=100 帧切块
    ├── Pad:       rnn.pad_sequence() 补齐不等长 chunk
    ├── Conv2d×3:  GELU(Conv2d(GELU(Conv2d(GELU(Conv2d(x))))))  stride=2
    ├── Unpad:     padded_embed[padded_mask_after_cnn]  裁掉填充帧
    ├── Re-window: 按 n_window_infer=800 重新分组为 Transformer 窗口
    ├── 18×Transformer:  带 cu_seqlens 变长注意力
    └── Project:  ln_post → proj1 → gelu → proj2  →  [T', 1024]
```

关键参数（来自 `config.json → thinker_config.audio_config`）：

| 参数 | 值 | 作用 |
|------|-----|------|
| `n_window` | 50 | 卷积块大小：50×2=100 帧/chunk |
| `n_window_infer` | 800 | Transformer 推理窗口 |
| `conv_chunksize` | 500 | 最大并行 chunk 数（防 OOM） |
| `downsample_hidden_size` | 480 | Conv2d 通道数 |
| `d_model` | 896 | Transformer hidden dim |
| `output_dim` | 1024 | 输出维度 |
| `encoder_layers` | 18 | Transformer 层数 |

Subsampling 公式：`_get_feat_extract_output_lengths()` — 每 100 帧 → 13 输出帧，余数走 3 层 stride-2 Conv2d（≈ ceil(x/8)），等效 subsampling ~6.5–7.7×。

## 两条导出链路的对比

### 当前链路：第三方 ONNX → MNNConvert（已完成）

```
Qwen 官方权重 (HuggingFace/ModelScope)
    │
    └──→ Wasser1462 导出脚本 ──→ conv_frontend.onnx + encoder.int8.onnx
            │                              │
            │    github.com/Wasser1462/Qwen3-ASR-onnx
            │    Modelscope: zengshuishui/Qwen3-ASR-onnx
            │
            └──→ 我们的 MNNConvert ──→ conv_frontend.mnn + encoder.mnn
                                          │
                                          └──→ Omni 引擎双模型路径 ✅ 实机运行中
```

**问题**：导出链路依赖第三方（Wasser1462 的 ONNX），不可控、无版本管理、无法复现。

### 正道：llmexport.py 全链路自控（未实现）

```
Qwen 官方权重 + modeling 代码
    │
    └──→ llmexport.py (我们的 qwen3_asr_model.py)
            │  Qwen3ASRAudioEncoder.forward()
            │  含完整 Chunk → Pad → Conv×3 → Slice → Window → Transformer → Project
            │
            └──→ torch.onnx.export() ──→ audio.onnx
                                            │
                                            └──→ MNNConvert ──→ audio.mnn (单文件)
                                                                  │
                                                                  └──→ Omni 引擎单模型路径
```

**优势**：全链路自控（和 MNN 其他所有模型一致）、可复现、可调试、单文件模型。

## 是否需要修复？

| 维度 | 当前（双模型 AE） | 修复后（单模型 llmexport） |
|------|:------|:------|
| **精度** | cosim 1.0 / 0.997（INT8，已够用） | 预期 cosim 1.0（FP32 权重） |
| **导出链路** | ❌ 依赖第三方 ONNX | ✅ llmexport.py 全链路自控 |
| **模型形态** | 🟡 双文件 conv_frontend + encoder | ✅ 单文件 audio.mnn |
| **可控性** | ❌ Wasser1462 脚本不可控 | ✅ 和 MNN 其他模型一致 |
| **可复现** | ❌ 依赖第三方 ONNX 文件 | ✅ 从权重直接导出 |
| **工作量** | 0（已完成） | ~1 天（移植 forward() + 调试导出） |
| **风险** | 0 | ONNX 追踪 chunking 逻辑可能踩坑 |

**结论：技术上值得修，但不是紧急事项。**

修复的目的不是精度（当前双模型 AE 精度已经够用），而是让 Qwen3-ASR 的导出链路回到 MNN 的正轨——和 Qwen2.5-Omni、Gemma4 等其他模型一样，llmexport.py 自控全链路，不依赖任何第三方的中间 ONNX 文件。

### 修复方案概要

将 `modeling_qwen3_asr.py` 中 `Qwen3ASRAudioEncoder.forward()` 的完整逻辑移植到我们的 `qwen3_asr_model.py`：

1. **移植 `_get_feat_extract_output_lengths()`** — 输出帧数计算公式
2. **移植 Chunking 逻辑** — `n_window*2=100` 帧切块 + `pad_sequence`
3. **添加 Unpad** — `padded_embed[padded_mask_after_cnn]`
4. **移植 Re-window** — `n_window_infer` 窗口分组 + `cu_seqlens` 生成
5. **适配 ONNX 导出** — 确认动态轴配置（`dynamic_axes`）对变长 chunk/Slice 的兼容性
6. **移除双模型路径** — Omni 引擎回退到单模型 `audio.mnn` 推理

### 注意事项

- 官方 forward() 中的 `nn.utils.rnn.pad_sequence` + `torch.split` + boolean mask 索引等动态操作，需要验证 `torch.onnx.export()` 的兼容性
- Wasser1462 拆成 conv_frontend + encoder 两个 ONNX 不是偶然——这是规避部分动态操作追踪问题的策略。如果 llmexport.py 单文件导出遇到困难，保持双模型也是合理的
- 修复后需桌面端验证：新 `audio.mnn` vs ONNX AE 的 cosim、端到端 first token 一致性

## 参考

- 官方 modeling 代码：`github.com/QwenLM/Qwen3-ASR` → `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
- 本地副本：`/tmp/modeling_qwen3_asr.py`
- Wasser1462 导出脚本：`github.com/Wasser1462/Qwen3-ASR-onnx`
- 当前双模型方案：[[sherpa-ae-mnn-integration]]
- 根因分析：[[root-cause-analysis]]
- 项目进度：[[progress]]
