---
date: 2026-06-14
status: active
tags: [qwen3-asr, accuracy, analysis, fbank]
category: analysis
aliases: [精度根因分析, Root Cause Analysis]
related: [[fbank-numerical-analysis]], [[omni-parameters]], [[progress]]
---
# Qwen3-ASR 识别精度差异分析

> 日期：2026-06-14（更新）| 状态：**主因确认 — AEC/NS 硬件音效**
>
> 问题：SherpaOnnx 框架的 Qwen3-ASR 识别精度很高，但 MNN 早期导出在罕见词上表现差。

## 最终结论（2026-06-14 更新）

经过系统实验逐一排除后，**精度差异主因排名**：

1. **🔴 AEC + NoiseSuppressor 硬件音效（影响最大）**：Android `AcousticEchoCanceler` + `NoiseSuppressor` 在 vendor DSP 上的实现在许多手机上会严重扭曲语音频谱。去掉后（与 sherpa-onnx 一致：raw MIC 直通）识别质量大幅提升。详见 [[progress]] Pitfall #12。
2. **🟡 模型导出/转换质量差异**：SherpaOnnx 使用 INT8 量化的第三方 ONNX 导出；MNN 使用 llmexport.py 导出路径。FBank 通过 `whisper_fbank_knf()` (kaldi-native-fbank) 已对齐。
3. **🟢 Sampling 参数**：ASR 必须用 greedy (temp=0, top_k=1)。
4. **🟢 FBank 数值差异**：kaldi-native-fbank 集成后已消除。

## 已排除的假设

| 假设 | 验证结果 | 说明 |
|------|:--:|------|
| System prompt 不匹配 | ❌ 已排除 | 影响能否识别，不影响精度 |
| Tokenizer 格式 | ❌ 已排除 | Tiktoken BPE 格式完整 |
| C++ FBank Bug | ❌ 已排除 | 修复了 log floor guard（防止 NaN），窗口/Nyquist 非 bug |
| FBank 归一化 | ❌ 已排除 | Python 训练和 C++ 推理公式一致 `(x+4)/4` |
| 重复惩罚 | ❌ 已排除 | Omni 引擎 `repetition_penalty=1.0`（无惩罚） |

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
