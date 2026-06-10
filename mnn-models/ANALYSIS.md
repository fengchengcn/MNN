# Qwen3-ASR 识别精度差异分析

> 问题：SherpaOnnx 框架的 Qwen3-ASR 识别精度很高，但 MNN 项目（手写推理引擎 & Omni 引擎）在 int8/fp16 多种模型下，通用问题是常见字词句子识别还行，不常见的识别很差。

## 1. 架构对比

| 维度 | SherpaOnnx（精度高） | MNN 手写引擎 + Omni 引擎（精度差） |
|------|---------------------|----------------------------------|
| **推理运行时** | ONNX Runtime | MNN |
| **模型架构** | conv_frontend + encoder + decoder（标准三件套） | audio_encoder + llm_kv_8bit（简化双件套） |
| **FBank 提取库** | kaldi-native-fbank | MNN `whisper_fbank()` |
| **Mel bins** | 80（kaldi 默认） | 128 |
| **FBank 归一化** | **逐句均值-方差归一化** `(x-μ)/(σ+1e-5)` | **固定偏置-缩放** `max(x, max-8.0)` 后 `(x+4.0)/4.0` |
| **System Prompt** | **空字符串** `""` | **"You are a helpful assistant."** |
| **Tokenizer** | HuggingFace BPE 目录 (`vocab.json` + `merges.txt` + `tokenizer_config.json`) | 单文件 `tokenizer.txt` |
| **解码策略** | Greedy（temperature=0），无重复惩罚 | Greedy + **REP_PENALTY=1.15** |
| **FBank C++ Bug** | 无 | 缺少 log floor guard（→ `-inf`），Nyquist bin 未移除，非周期性 Hanning window |

## 2. 根因概率排序

### 🥇 System Prompt 不匹配 — 概率 ~30%

**SherpaOnnx**：默认 system prompt 为**空字符串**：

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
```

**MNN**：硬编码英文 chatbot prompt：

```
<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n<|audio_start|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
```

**源码位置**：`apps/Android/MnnLlmChat/app/src/main/cpp/qwen3_asr_engine.cpp:270`

```cpp
auto sys_msg = tok->encode("You are a helpful assistant.");
```

**影响机制**：Qwen3-ASR 训练时针对 ASR 任务使用了特定 prompt 格式。注入英文通用 chatbot prompt 导致模型在**语言建模先验**（"我是什么都懂的助手"）和**音频条件转录**（"忠实转写音频"）之间产生偏差：
- 常见词：语言模型先验强，即使 prompt 不匹配也能输出正确
- **罕见词**：模型依赖音频信号更多，但 prompt 移位导致模型偏向"聊天助手行为"而非"忠实转录"，罕见词容易被替换为常见近似词

**修复**：将 system message 改为空字符串，或改为中文 ASR 指令如 `"请准确转写以下音频内容。"`。

---

### 🥈 FBank 归一化方式不同 — 概率 ~25%

| | SherpaOnnx (kaldi-native-fbank) | MNN (`whisper_fbank`) |
|---|---|---|
| 归一化方式 | **逐句**均值-方差归一化 | **固定** `(x+4.0)/4.0` |
| | `(x - mean) / (stddev + 1e-5)` | `max(x, max_val - 8.0)` 后 normalize |
| 窗口函数 | Hamming (periodic) | **Hanning (non-periodic)** |
| Log floor guard | ✅ `clamp(min=1e-10)` | ❌ **缺失** → 零能量 → `-inf` |
| Nyquist bin | 移除（`stft[..., :-1]`） | **保留**（多一个频点） |
| Mel bins | 80（kaldi 标准） | 128 |

**影响机制**：
- **逐句均值-方差归一化**：每句特征归一到零均值单位方差，补偿不同录音音量、麦克风增益差异。罕见词依赖更精细的频谱细节，没有自适应归一化时更易被淹没
- **固定偏置** `(x+4.0)/4.0`：无自适应能力，句子整体偏轻/偏响会使特征分布偏离训练分布
- **Log floor guard 缺失**：静音帧 mel bin 能量为零，`log10(0) = -inf`，导致后续运算产生 `NaN`，对长段静音后的罕见词有破坏性影响

**源码位置**：
- C++ 缺失 guard：`tools/audio/source/audio.cpp:657-659`
- Python 参考（正确）：`transformers/llm/export/utils/audio.py:486`
  ```python
  log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # ← 有 clamp
  ```
  ```cpp
  auto log_specgram = _Log(mel_spectrogram) / _Log(10.0);  // ← 无 clamp！
  ```

---

### 🥉 Tokenizer 格式差异 — 概率 ~20%

**SherpaOnnx**：完整 HuggingFace BPE tokenizer 目录，包含：
- `vocab.json`（词表，~2.6MB）
- `merges.txt`（BPE 合并规则，~1.6MB）
- `tokenizer_config.json`（配置，~12KB）
- `split_special_tokens: false`（特殊 token 不拆分）

**MNN**：单文件 `tokenizer.txt`，通过 `MNN::Transformer::Tokenizer::createTokenizer()` 自动检测格式加载。

**影响机制**：Qwen 系列使用基于 `tiktoken` 的自定义 BPE tokenizer，vocab_size=151936。`tokenizer.txt` 需正确包含整个词表和合并规则。如果转换过程有问题：
- 罕见字符的 BPE 编码规则不完整 → 罕见词被拆成错误 token 序列 → 模型看到不该出现的 token 组合 → 输出错误
- 特殊 token 映射偏差（如 `<|audio_pad|>` = 151676）→ 音频注入位置错误
- 常见词通常由几个高频 BPE token 组成，tokenizer 映射出错概率低，所以常见词 OK

**验证**：用同一文本分别在 SherpaOnnx BPE tokenizer 和 MNN tokenizer.txt 上做 encode→decode 往返，特别关注罕见字的结果差异。

---

### 4. C++ FBank 实现 Bug — 概率 ~12%

三个已确认的 Python→C++ 移植差异（详见 [#2 表格](#-fbank-归一化方式不同--概率-25)）：
1. **Log floor guard 缺失** → 零能量 → `-inf` → `NaN`
2. **Nyquist bin 未移除** → mel filterbank 输入多一个频点 → 高频 mel band 能量偏差
3. **非周期性 Hanning window** → 频谱泄漏特性与训练时不同

三个 bug 叠加，fbank 输出与训练时不完全一致。偏差经过 audio_encoder（3×Conv2d + 18×Transformer）放大后影响最终 embedding。

---

### 5. 重复惩罚 (REP_PENALTY) — 概率 ~8%

- SherpaOnnx：`temperature=0.0`，**无重复惩罚**
- MNN：`REP_PENALTY = 1.15`（`qwen3_asr_engine.h:144`）

```cpp
// argmaxPenalized: 对已生成 token 施加惩罚
if (penalized_buf[id] < 0)
    penalized_buf[id] *= penalty;   // 负值更负
else
    penalized_buf[id] /= penalty;   // 正值降低
```

1.15 不算激进，但对罕见 token：出现一次后就被惩罚，降低了后续正确重现的概率。中文 ASR 中同一个罕见字可能在同一句中出现多次。

---

### 6. 模型转换质量 — 概率 ~5%

用户试了 int8 和 fp16 多种精度模型，问题一致。如果是量化误差导致，不同精度应有明显差异。
- MNN audio_encoder 合并了原始 Qwen3-ASR 的 conv_frontend + encoder，可能引入融合误差
- 但 SherpaOnnx 也有类似导出处理，架构差异比转换质量更可能是原因

---

## 3. 建议验证优先级

| 优先级 | 验证项 | 改动量 | 预期效果 |
|--------|--------|--------|----------|
| **P0** | 将 system prompt 改为空字符串或中文 ASR 指令 | 1 行 | 可能解决 30%+ |
| **P0** | 在 `whisper_fbank` 中加 `clamp(min=1e-10)` | 1 行 | 修复 NaN 问题 |
| **P1** | 对比 tokenizer.txt 与 HuggingFace BPE tokenizer 的 encode/decode 往返 | 离线测试 | 确认 tokenizer 准确性 |
| **P1** | 改为逐句均值-方差归一化 `(x-μ)/(σ+1e-5)` | ~10 行 | 与 SherpaOnnx 对齐归一化 |
| **P2** | 去掉重复惩罚（设 `REP_PENALTY=1.0`） | 1 行 | 排除惩罚干扰 |

---

## 4. 关键代码位置参考

| 文件 | 行号 | 内容 |
|------|------|------|
| `apps/.../cpp/qwen3_asr_engine.cpp` | 270 | System prompt 硬编码 "You are a helpful assistant." |
| `apps/.../cpp/qwen3_asr_engine.cpp` | 413,705 | `whisper_fbank(wf)` 调用（全部使用默认参数） |
| `apps/.../cpp/qwen3_asr_engine.cpp` | 496,598,785,822 | 重复惩罚调用点 |
| `apps/.../cpp/qwen3_asr_engine.h` | 135-145 | 常量定义（HIDDEN, VOCAB, REP_PENALTY 等） |
| `tools/audio/source/audio.cpp` | 639-665 | `whisper_fbank` 实现 |
| `tools/audio/source/audio.cpp` | 657 | **Bug**: `_Log(mel_spectrogram)` 无 clamp |
| `tools/audio/source/audio.cpp` | 301-309 | `hann_window`（non-periodic） |
| `transformers/llm/export/utils/audio.py` | 467-491 | Python 参考 fbank 实现（正确的） |
| `transformers/llm/export/utils/audio.py` | 486 | Python: `torch.clamp(mel_spec, min=1e-10).log10()` |

---

## 5. 分析日期与方法

- **日期**：2026-06-11
- **方法**：对比 SherpaOnnx (ONNX Runtime) 与 MNN 两个推理框架的 Qwen3-ASR 实现，深度阅读所有相关源码（C++、Python、Kotlin），对比 fbank 预处理、tokenizer、prompt 构造、解码策略等维度，结合 Web 搜索 sherpa-onnx 开源实现文档进行交叉验证。
- **SherpaOnnx 参考项目**：`D:\mojing\SherpaOnnxSimulateStreamingAsr\SherpaOnnxSimulateStreamingAsr`
- **MNN 参考源码**：`apps/Android/MnnLlmChat/app/src/main/cpp/`、`tools/audio/source/`、`transformers/llm/export/`
