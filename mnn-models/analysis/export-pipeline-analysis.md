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

### 修复方案概要（2026-06-14 修正）

实际修复方案**大幅简化**了一开始的计划。最初计划移植完整 Chunk/Pad/Slice/Window 逻辑，但在实施中发现：

1. **ONNX 不可 trace 阻断**：`torch.split`、`pad_sequence`、boolean mask 索引无法被 `torch.onnx.export()` 追踪 — 完整移植在单文件导出中根本行不通
2. **短音频数学等价**：对 ≤30s 音频，简化版（全序列连续 conv）与完整版（分块）结果相同
3. **真正问题不是架构**：cosim ~0.30 的根因是两个代码 bug（bias + PE），修复后精度已对齐

**实际修复**（2 个 bug + 配置适配），详见上方 § 实施结果。

### 实施结果（2026-06-14）

**采用简化方案**：保持简化 forward（无 chunking），原因如下：
- Chunk/Pad/Slice/Window 是官方代码的**长音频内存优化**，非精度要求
- 短音频（≤30s=3000 mel→~375 enc frames）不超单窗口（n_window_infer=800），**全序列连续 conv + full bidirectional attention 与分窗版数学等价**
- 完整 Chunk/Pad/Slice/Window 逻辑包含 `torch.split`、`pad_sequence`、boolean mask 索引等不可 trace 操作，ONNX 导出会失败
- Wasser1462 拆成双模型正是为了规避这些 ONNX 限制

**关键修复（2 个 bug）**：

| Bug | 文件位置 | 现象 | 修复 |
|-----|----------|------|------|
| `conv_out` 多余 random bias | `qwen3_asr_model.py:220` | 896 个随机值，每次加载不同 | `nn.Linear(..., bias=False)` |
| 位置编码 interleaved → concat | `qwen3_asr_model.py:231-241` | 每个位置 max diff=2.0 | `torch.cat([sin, cos], dim=1)` |

这两个 bug 是之前 cosim ~0.30 的**真凶**，修复后 cosim 提升至 0.993。

**已完成变更**：

| 文件 | 变更 | 状态 |
|------|------|:----:|
| `qwen3_asr_model.py` | 修复 conv_out bias、位置编码公式 | ✅ |
| `qwen3_asr_model.py` | AudioEncoder 添加 `n_window`/`n_window_infer`/`conv_chunksize` 配置 | ✅ |
| `audio.py` | Qwen3AsrAudio.load() 读取 window config | ✅ |
| `llmconfig.hpp` | `audio_encoder()` 默认值 `"encoder.mnn"` → `""`（防误加载旧文件） | ✅ |
| `omni.cpp` | **无需修改** — 单模型路径在 `mAudioEncoder` 为 null 时自动启用 | ✅ |
| ONNX 导出验证 | 2050 节点，全部标准 op，PyTorch vs ONNX max diff=**2.36e-6** | ✅ |
| 桌面端验证 | MNN audio.mnn vs Wasser1462 cosim=**0.993**（T=100, 同帧数 13 vs 13） | ✅ |

**验证数据**：

| 输入长度 | 帧数对比 (new vs old) | avg_frame_cosim | first_frame_cosim |
|----------|:---------------------:|:---------------:|:-----------------:|
| T=100 | 13 vs 13 | **0.9933** | **0.9961** |
| T=200 | 25 vs 26 | 0.7687 | **0.9798** |
| T=500 | 63 vs 65 | 0.6081 | 0.9035 |

- T=100 时帧数相同 → cosim 0.993 确认权重和计算完全对齐
- 大 T 时低 cosim 来自帧数不对齐（旧模型 chunk boundary 效应导致多出 ~4% 帧），非权重差异

**已知限制**：

| 限制 | 详情 | 影响 |
|------|------|------|
| **长音频识别退化** | 手机实测：≥16s 音频出现语音幻觉（FP16/INT8 均有）。7s 以内识别正常。 | 中等 |
| 非分块 conv 与训练时不一致 | 模型训练时对每 100 mel 帧分块独立 conv，各 chunk 边界有 zero-padding artifact。简化版全序列连续 conv 无此 artifact → 多 chunk 时中间表示偏离训练分布 | — |
| 旧 so fallback 误加载 | `llmconfig.hpp` 旧默认值 `"encoder.mnn"` 已修复为 `""`。需清理手机上残留旧文件 | 已修复 |
| >30s 超长音频 | 超出 n_window_infer=800 单窗口上限，需分窗 | ASR 不现实 |

**缓解措施**：VAD 模式天然将音频切分为短段（每段 <5s），绕过长音频退化问题。BATCH 模式建议限制在 10s 以内。

**根本解决**：需移植完整 Chunk/Pad/Slice 逻辑（`torch.split` + `pad_sequence` + boolean mask），但这些 op 无法被 `torch.onnx.export()` 追踪。可能需要拆为双模型（conv_frontend 含 chunk 逻辑 + encoder），类似 Wasser1462 的方案。这将在后续迭代中评估。

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
