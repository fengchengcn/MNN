---
date: 2026-06-14
status: active
tags: [qwen3-asr, accuracy, analysis, fbank]
category: analysis
aliases: [精度根因分析, Root Cause Analysis]
related: [[fbank-numerical-analysis]], [[omni-parameters]], [[progress]]
---
# Qwen3-ASR 识别精度差异分析

> 日期：2026-06-14（更新）| 状态：**已修复 — 双模型 AE 集成完成**
>
> 问题：SherpaOnnx 框架的 Qwen3-ASR 识别精度很高，但 MNN 早期导出在罕见词上表现差。
>
> **解决方案**：用 MNNConvert 转换 sherpa-onnx 的 `conv_frontend.onnx` + `encoder.int8.onnx` 替换 llmexport.py 手写的 `audio.mnn`，实现与 sherpa-onnx **完全架构等价**。详见 [[sherpa-ae-mnn-integration]]。

## 最终结论（2026-06-14 更新）

经过系统实验逐一排除后，**精度差异主因排名**：

1. **🔴 AEC + NoiseSuppressor 硬件音效（影响最大，已修复）**：Android `AcousticEchoCanceler` + `NoiseSuppressor` 在 vendor DSP 上的实现在许多手机上会严重扭曲语音频谱。去掉后（与 sherpa-onnx 一致：raw MIC 直通）识别质量大幅提升。详见 [[progress]] Pitfall #12。
2. **🔴 Audio Encoder 模型架构不等价（根因，确认）**：2026-06-14 桌面对比实验确认，MNN `audio.mnn` 和 sherpa-onnx `conv_frontend.onnx` + `encoder.int8.onnx` 在相同 fbank 输入下**输出完全不同**（cosim ~0.30，远低于 llmexport ONNX AE vs MNN AE 的 0.998），且时间维度 subsampling ratio 不同（MNN 8× vs sherpa 6.5×）。两个 AE 不是等价的模型架构，这是 MNN 精度追不上 sherpa-onnx 的根因。
3. **🟢 推理精度 (FP16)**：手机端已使用 `precision: "high"` (FP32)，排除此因素。
4. **🟢 FBank 数值差异**：kaldi-native-fbank 集成后已消除。
5. **🟢 Sampling 参数**：ASR 已配置 greedy (temp=0, top_k=1)。
6. **🟢 Audio Encoder**：桌面端对比实验已排除（见 §5）。

## 已排除的假设

| 假设 | 验证结果 | 说明 |
|------|:--:|------|
| System prompt 不匹配 | ❌ 已排除 | 影响能否识别，不影响精度 |
| Tokenizer 格式 | ❌ 已排除 | Tiktoken BPE 格式完整 |
| C++ FBank Bug | ❌ 已排除 | 修复了 log floor guard（防止 NaN），窗口/Nyquist 非 bug |
| FBank 归一化 | ❌ 已排除 | Python 训练和 C++ 推理公式一致 `(x+4)/4` |
| 重复惩罚 | ❌ 已排除 | Omni 引擎 `repetition_penalty=1.0`（无惩罚） |
| FP16 推理精度 | ❌ 已排除 | 手机 config.json 已设置 `precision: "high"` (FP32) |
| Audio Encoder 导出质量 | ❌ 已排除 | 桌面对比：MNN AE → ONNX Dec 与 ONNX AE → ONNX Dec 5/5 first token 一致 |

## 桌面端对比实验（2026-06-14，§5）

**目的**：隔离音频编码器 (AE) 与解码器 (Decoder) 的误差贡献。

**方法**：
```
同一段音频 → librosa fbank (HTK mel, preemphasis=0.97)
    ├─→ ONNX audio_encoder.onnx  ──→ AE 输出 A
    └─→ MNN audio.mnn            ──→ AE 输出 B
                                           │
                              A 和 B 分别送入 ONNX llm.onnx Decoder
                                           │
                                   对比 first token
```

- ONNX 模型由 `llmexport.py` 从同一 HF 模型导出（未经 MNNConvert）
- MNN AE 通过 C++ helper 调用 `Module::load("audio.mnn")` 推理
- Decoder 统一使用 ONNX Runtime（排除 MNN Decoder 差异）
- FBank 参数对齐：HTK mel, preemphasis=0.97, hann window, `(x+4)/4` 归一化

**结果**：

| 音频 | AE cosim | AE maxdiff | Dec cosim | First token | 判定 |
|------|:--------:|:----------:|:---------:|:-----------:|:----:|
| 0121.wav | 0.998910 | 0.0051 | 0.999931 | 11528 | ✅ |
| 0123.wav | 0.999379 | 0.0043 | 0.999949 | 11528 | ✅ |
| 0125.wav | 0.999332 | 0.0045 | 0.999945 | 11528 | ✅ |
| 0127.wav | 0.999255 | 0.0050 | 0.999967 | 11528 | ✅ |
| 0129.wav | 0.998846 | 0.0058 | 0.999839 | 11528 | ✅ |

**结论**：MNN Audio Encoder 输出与 ONNX AE 几乎一致（cosim > 0.998），在相同的 ONNX Decoder 下 first token 全部一致。**AE 不是误差源，问题在 MNN Decoder 侧**。

## 桌面端 INT8 vs FP16 输出差异观察（2026-06-14）

> ⚠️ **注意**：以下为桌面端 5 样本 A/B 输出对比，**不代表真实识别精度**。5 个样本无 ground truth，输出不同不等于判断孰优孰劣。手机实测（更多样本）FP16 效果更好。

同一 5 段音频在 4 种配置下的识别文本对比：

| 配置 | 权重 | 激活 | 0121 | 0123 | 0125 | 0127 | 0129 |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| INT8+FP16 | INT8 | FP16 | 聚集了 | 聚集了 | **乱码** | 张大伟说 | 火爆 |
| INT8+FP32 | INT8 | FP32 | 聚集了 | 聚集了 | 调整市场战略 | 张大伟说 | 火爆 |
| FP16+FP16 | FP16 | FP16 | 聚集了 | **聚极了** | 调整市场战略 | **地茶** | 火爆 |
| FP16+FP32 | FP16 | FP32 | 聚集了 | **聚极了** | 调整市场战略 | **地茶** | 火爆 |

**观察**：
- INT8 和 FP16 权重在不同样本上各有输出差异，无 ground truth 无法判断孰优孰劣
- **FP16 激活影响较小**：仅 0125 一个样本 INT8+FP16 vs INT8+FP32 出现明显不同
- 手机实测（更多样本）：FP16 识别效果 > INT8。以手机实测为准。

## Sherpa-onnx AE vs MNN AE 对比（2026-06-14，🔴 根因确认）

**目的**：用 sherpa-onnx 实际使用的 ONNX 模型和 MNN audio.mnn 对比，确认 AE 是否等价。

**方法**：同一段 fbank → 分别送入 sherpa-onnx AE（conv_frontend.onnx → encoder.int8.onnx）和 MNN audio.mnn → 对比两路 AE 输出 → 分别送入同一个 ONNX Decoder 对比 first token。

**结果**：

| 音频 | Sherpa AE frames | MNN AE frames | AE cosim | AE maxdiff | First token |
|------|:----------------:|:-------------:|:--------:|:----------:|:-----------:|
| 0121 | 65 | 53 | **0.319** | 0.101 | ✅ MATCH |
| 0123 | 65 | 51 | **0.349** | 0.129 | ✅ MATCH |
| 0125 | 65 | 54 | **0.298** | 0.130 | ✅ MATCH |
| 0127 | 65 | 56 | **0.304** | 0.127 | ✅ MATCH |
| 0129 | 52 | 45 | **0.308** | 0.112 | ✅ MATCH |

**结论**：
- **Sherpa-onnx AE 和 MNN AE 完全不等价**。相同 fbank 输入，AE 输出 cosim 仅 ~0.3（对比：llmexport.py ONNX AE vs MNN AE 的 cosim 为 0.998）
- **Subsampling ratio 不同**：sherpa ~6.5×，MNN ~8×，说明 conv_frontend 结构有差异
- **同一模型权重、不同导出代码路径导致图结构不对齐**——不是两个不同的模型，而是同一份 HF checkpoint 的权重被两套导出代码以不同精度的 forward 复现，产生了不同的计算图拓扑
- 这是 MNN Qwen3-ASR 精度追不上 sherpa-onnx 的**根本原因**

**架构差异定位**（2026-06-14）：

| 图组件 | sherpa-onnx conv_frontend | MNN llmexport.py AE | 
|--------|--------------------------|---------------------|
| 预处理 | **Pad**（补零帧，扩展时间维度） | ❌ 无 |
| Conv2d×3 | stride=2, kernel=3×3, pad=1（**同一份权重**，max diff=0） | 相同 |
| 后处理 | **Slice**（裁切输出帧） | ❌ 无 |
| 维度转置 | 多次 Transpose，perm 序列不同 | unsqueeze + reshape + transpose |
| 中间投影 | 7680 → **896** | 7680 → **1024** |
| Subsampling | ~6.5× | ~8× |

**修复方向**：在 `export_qwen3_asr.py` 的 `Qwen3ASRAudioEncoder.forward()` 中，参照 conv_frontend.onnx 的图结构添加 Pad → Conv2d×3 → Slice → Transpose 链，使输出帧数和嵌入值对齐 sherpa-onnx。

### 根因解释：为什么同一份权重会产生不同的图结构？（2026-06-14）

**不是两个不同的模型，不是不同的后训练。权重完全相同（Conv2d kernel max diff = 0），差异来自两套导出框架对原始 HF 模型 forward() 的复现精度不同。**

#### sherpa-onnx：图追踪（保留原始图拓扑）

sherpa-onnx 对原始 HuggingFace 模型做 `torch.onnx.export()` 图追踪，**完整保留了原始模型 conv_frontend 中的 Pad → Conv2d×3 → Slice 操作链**。这些 Pad/Slice 在 Whisper-style 卷积前端中用于处理时间维度对齐——当输入帧数不是 2 的幂次时，`Conv2d(k=3, s=2, p=1)` 的 PyTorch 尺寸计算公式为：

$$L_{out} = \lfloor(L_{in} + 2 - 3) / 2\rfloor + 1 = \lfloor(L_{in} - 1) / 2\rfloor + 1$$

3 层叠加后的有效下采样 **不是精确的 8×**，结合 Pad/Slice 的边界处理，实际等效下采样约 **6.5×**。

#### MNN llmexport.py：手动重建 forward()（丢失 Pad/Slice）

MNN 的 llmexport 架构设计选择是**手动重建 `nn.Module` 并自定义 `forward()`**，而非对 HF 模型做黑盒 ONNX 导出。这样做的好处是可以精确控制图结构（去除 HF 框架冗余 op、控制变量名、做算子融合），但代价是如果手写 `forward()` 没有 100% 对齐原始模型的图细节，就会引入偏差。

在 `qwen3_asr_model.py:244-253` 和 `export_qwen3_asr.py:138-151` 中，手写 `forward()` 的实现为：

```
unsqueeze(1) → Conv2d×3 (k=3, s=2, p=1) → permute+reshape → conv_out
```

**缺失了原始 HF 模型 forward() 中的 Pad 和 Slice 操作**。纯 stride-2 ×3 下采样恰好为 8×（$2^3=8$），导致输出帧数与 sherpa-onnx 不一致（如 53 vs 65 帧）。

#### 影响链路

```
同一段音频 → 相同 FBank 特征
    ├─→ sherpa-onnx AE (Pad→Conv→Slice, ~6.5× subsample) → 65 frames → 编码器→Decoder ✅
    └─→ MNN AE           (纯Conv,           ~8× subsample)   → 53 frames → 编码器→Decoder ❌ 完全不同的嵌入
                                                                           余弦相似度仅 ~0.30
```

帧数差异（65 vs 53）经过 18 层 Transformer 编码器后被放大，导致 AE 输出的嵌入向量完全不同（cosim ~0.30，对比 llmexport ONNX AE vs MNN AE 的 cosim 为 0.998+），即使送入同一个 Decoder，后续 token 分布也完全偏离。

**总结**：同一份模型权重 + 不同精度的导出代码路径 → 图结构不等价 → 数值不一致。不是模型本身有差异，是导出代码对原始 forward() 的复现保真度不同。

### 深入追踪：远超 Pad/Slice 的维度差异（2026-06-14）

尝试仅通过 Pad 对齐帧数后，cosim 仍然只有 ~0.38。进一步分析发现 sherpa conv_frontend 的 ONNX 图远比预期复杂（共 130 个节点），包含：

- **动态 Shape 计算**（前 30 个节点全部是 Shape / Gather / Mod / Concat）：运行时根据实际帧数动态计算 Pad 和 Slice 参数
- **多次 Transpose + Reshape**：维度转置序列与 MNN 手写版完全不同
- **Conv2d 作用的 tensor layout 不同**：同样的权重在错误的 tensor layout 上计算 = 语义全错

**结论**：不是简单的"漏了 Pad/Slice 两行代码"。sherpa 和 MNN 的 conv_frontend 图是**不同人写出的两套独立实现**，在 Transpose perm、Reshape 目标形状、tensor 维度语义上均有差异。手写修复需要逆向 130 节点的 ONNX 图，投入巨大且不保证正确。

### 方案评估：如何获得正确的 AE？（2026-06-14）

| 方案 | 可行性 | 代价 |
|------|:------:|------|
| 反编译 ONNX 图 → 手写 PyTorch forward() | 理论可行 | 几天逆向 + 验证，不保证正确 |
| 找 HF 原始 modeling 文件 | 需要 Qwen3-ASR 官方代码 | 目前不可用 |
| **直接用 sherpa conv_frontend.onnx + encoder.int8.onnx → MNNConvert → MNN** | ✅ **已验证可行** | 模型管理变复杂，引擎对接需适配 |
| 维持现状 | 手机 FP16 已可用 | AE 架构不等价问题未解决 |

### 方案三 MNNConvert 验证结果（2026-06-14，✅ 通过）

用 MNNConvert 转换 sherpa-onnx 模型并做数值验证：

```
同一份 fbank [1, 100, 128] 输入：

  ONNX conv_frontend.onnx  ──→  [1, 13, 896]
  MNN  conv_frontend.mnn   ──→  [1, 13, 896]   余弦相似度 = 1.0000 ✅ 完全一致

  ONNX encoder.int8.onnx   ──→  [1, 13, 1024]
  MNN  encoder.mnn         ──→  [1, 13, 1024]   余弦相似度 = 0.9968 ✅ 高度一致
```

- **conv_frontend.mnn**：与 ONNX 完全等价（cosim=1.0, maxdiff=0.005），MNNConvert 忠实保留了 Pad→Conv2d×3→Slice→Transpose 的完整动态图
- **encoder.mnn**：cosim=0.997（对比手写 audio.mnn vs sherpa encoder 的 0.30），差异可能来自 INT8 量化实现和 attention_mask 类型适配
- 两个文件路径：`/Users/bxy/Documents/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/`

**下一步**：需确认 Omni 引擎如何对接 conv_frontend.mnn (1 input, 1 output) + encoder.mnn (2 inputs: features + attention_mask)，以及引擎需要改多少代码来支持分离的 AE 模型文件。

### Decoder 对比实验（2026-06-14，§6）

**方法**：ONNX AE 产出音频嵌入 → 构建 merged embeddings → 分别送入 ONNX llm.onnx 和 MNN llm.mnn (INT8/FP16) → 对比 prefill logits。

**结果**：

| 音频 | ONNX tok | MNN INT8 tok | MNN FP16 tok | Cosim (INT8) | MaxDiff |
|------|:--------:|:------------:|:------------:|:------------:|:-------:|
| 0121 | 11528 | 11528 ✅ | 11528 ✅ | 0.9772 | 21.33 |
| 0123 | 11528 | 11528 ✅ | 11528 ✅ | 0.9759 | 22.38 |
| 0125 | 11528 | 11528 ✅ | 11528 ✅ | 0.9768 | 22.03 |
| 0127 | 11528 | 11528 ✅ | 11528 ✅ | 0.9735 | 21.72 |
| 0129 | 11528 | 11528 ✅ | 11528 ✅ | 0.9694 | 21.63 |

**分析**：
- **First token**: 5/5 MATCH（argmax 一致）
- **数值**: cosim ~0.97（远低于 AE 的 0.998），maxdiff ~21（远大于 AE 的 0.005）
- **系统性缩放差异**: MNN logit 值是 ONNX 的 ~4-5×（14-28 vs 3-6），可能来自 FusedAttention 或 RMSNorm 实现差异
- **结论**: MNN decoder 与 ONNX decoder 存在显著数值差异，但 token 排序保持正确。整个 AR decode loop 中误差未累积到改变 argmax

## 已完成的修复

### C++ FBank Log Floor Guard
`tools/audio/source/audio.cpp:657` — 添加 `clamp(min=1e-10)`：
```cpp
// Before: auto log_specgram = _Log(mel_specgram) / _Log(_Scalar<float>(10.0));
// After:
auto log_specgram = _Log(_Maximum(mel_specgram, _Scalar<float>(1e-10))) / _Log(_Scalar<float>(10.0));
```
这是正确性修复（防止静音帧产生 `-inf` → `NaN`），但与罕见词精度无关。

### Sampling 参数修正（Phase 2.6）

对于 ASR 任务，将 Omni config.json 从 `mixed` 采样改为 `greedy`：

| 参数 | 修正前 | 修正后 | 理由 |
|------|--------|--------|------|
| `sampler_type` | `mixed` | **`greedy`** | ASR 是确定性任务 |
| `temperature` | 0.1 | **0.0** | 罕见 token 置信度仅 30-70%，任何噪声足以偏离 |
| `top_k` | 40 | **1** | 关闭 |
| `top_p` | 0.9 | **1.0** | 关闭 |
| `n_gram` | 8 | **0** | 关闭（可能误伤低频术语） |

## 关键代码位置参考

| 文件 | 行号 | 内容 |
|------|------|------|
| `tools/audio/source/audio.cpp` | 657 | FBank log floor guard |
| `tools/audio/source/audio.cpp` | 301-309 | `hann_window` 实现 |
| `transformers/llm/export/utils/audio.py` | 486 | Python 参考 fbank（正确的） |
| `transformers/llm/export/utils/audio.py` | 520-540 | AE ONNX 导出（1 输入，无 attention_mask） |

## 分析方法论（可复用）

对比两个推理框架的 Qwen3-ASR 实现：
1. 同音频逐层对比：fbank → AE embedding → decoder logits
2. 定位从哪一层开始出现显著差异
3. 反向追溯导出脚本差异

此方法适用于任何跨框架精度问题排查。
