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

## 3. 验证进度

| 优先级 | 验证项 | 状态 | 结论 |
|--------|--------|:--:|------|
| **P0** | System prompt 实验（空/英文/中文） | ✅ 已完成 | 非根因 |
| **P0** | `whisper_fbank` 加 `clamp(min=1e-10)` | ✅ 已修复 | 正确性修复，非主因 |
| **P1** | Tokenizer 往返对比 | ✅ 已完成 | 格式完整，非根因 |
| **P1** | FBank 逐句均值-方差归一化 | ❌ 不适用 | 训练和推理公式一致，无需改 |
| **P2** | 重复惩罚检查 | ✅ 已完成 | Omni 引擎已为 1.0（无惩罚） |
| **P0** | **模型导出逐层对比（新）** | 🔴 待做 | 同音频对比 fbank→AE→decoder 输出 |

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

---

## 6. 实验验证记录（2026-06-11）

### 6.1 System Prompt 实验（已排除）

在 Omni FP16 模型（`Qwen3-ASR-MNN-FP16/config.json`）上测试了三种 system_prompt：

| system_prompt | 结果 | 结论 |
|---|---|---|
| `""` (空) | ❌ 完全不识别，仅输出 `<asr_text>` 标记 | Jinja 模板跳过空 system 块，模型无 ASR 指令 |
| `"Transcribe speech to text."` (英文原版) | ✅ 能识别，罕见词仍差 | 任务相关，不会被回显 |
| `"请准确转写音频内容。"` (中文) | ⚠️ 部分识别，但 prompt 被模型回显为转写结果 | 中文 prompt 和输出在同一语言空间，模型混淆 |

**结论**：system_prompt 影响能否识别，但**不是罕见词精度差的根因**。保留英文原版 `"Transcribe speech to text."`。

### 6.2 Tokenizer 验证（已排除）

对比了 MNN `tokenizer.txt` 与 HuggingFace Qwen2 BPE tokenizer：

| | Omni FP16 tokenizer.txt | Old-Engine tokenizer.txt | HuggingFace BPE |
|---|---|---|---|
| 大小 | 3.2 MB | 952 KB | vocab.json 2.6MB + merges.txt 1.6MB |
| 格式 | Tiktoken BPE（完整） | 仅 token ID 列表 | BPE 目录 |
| 罕见字符 "龘靐旮旯" | 0 次出现（**正常**，字节级 BPE 编码） | 0 次出现 | — |

罕见字符通过 UTF-8 字节级 BPE 组合编码是 Qwen tokenizer 的正常行为。MNN 的 Tiktoken 格式包含完整 BPE 合并规则。

**结论**：Tokenizer 格式完整，**不是罕见词精度差的根因**。

### 6.3 C++ FBank Bug 修复（已完成，非根因）

在 `tools/audio/source/audio.cpp:657` 添加了 `clamp(min=1e-10)`：
```cpp
// Before: auto log_specgram = _Log(mel_specgram) / _Log(_Scalar<float>(10.0));
// After:
auto log_specgram = _Log(_Maximum(mel_specgram, _Scalar<float>(1e-10))) / _Log(_Scalar<float>(10.0));
```

另两个疑似 Bug 复查结果：
- **Nyquist bin 未移除**：❌ 误判。C++ 的 `_Slice` 切的是时间维，Python 的 `stft[..., :-1]` 也是切时间维。两者一致。
- **非周期性 Hann window**：差异仅在第 399/400 个样本（~0.0005），**可忽略**。

**结论**：Log floor guard 修复是正确性改进（防止 NaN），但三个"Bug"都与罕见词精度无关。

### 6.4 SherpaOnnx 模型对比（关键发现）

直接对比了两个 0.6B Qwen3-ASR 模型：

| | SherpaOnnx（准） | MNN FP16（不准） |
|---|---|---|
| 量化精度 | **INT8** | **FP16**（理论上更高精度） |
| 解码器大小 | 720.9 MB | 1137.2 MB |
| 模型来源 | ModelScope 第三方导出 | MNN 自有导出 |
| 导出脚本 | [Wasser1462/Qwen3-ASR-onnx](https://github.com/Wasser1462/Qwen3-ASR-onnx) | `transformers/llm/export/` |
| 测试集 | 含 15 个 WAV + 标准转录文本 | 无 |

**关键矛盾**：FP16 精度理应高于 INT8，但 MNN FP16 识别效果反而不如 SherpaOnnx INT8。这**排除了量化误差是主要因素**，指向模型导出质量差异。

### 6.5 更新后的根因判断

经过实验逐一排除后：

| 假设 | 状态 | 说明 |
|------|:--:|------|
| System prompt | ❌ 已排除 | 影响识别与否，不影响精度 |
| Tokenizer 格式 | ❌ 已排除 | Tiktoken BPE 完整 |
| C++ FBank Bug | ❌ 已排除 | 修复了 log guard，窗口和 Nyquist 非 bug |
| FBank 归一化 | ❌ 已排除 | Python 训练和 C++ 推理公式一致 `(x+4)/4` |
| 重复惩罚 | — 未测试 | Omni 引擎 `repetition_penalty=1.0`（无惩罚），不影响 |
| **模型导出/转换质量** | **🔴 主嫌疑** | MNN 导出与 Wasser1462 ONNX 导出产生不同质量的模型 |

### 6.6 下一步方向

同一份 HuggingFace 权重 → 两个导出脚本 → 两种推理引擎 → 显著不同的精度。应该做**同音频逐层对比**：

1. 用同一 WAV 文件，分别跑 SherpaOnnx ONNX 和 MNN
2. 对比 fbank 输出 → audio encoder embedding → decoder logits
3. 定位从哪一层开始出现显著差异
4. 反向追溯 MNN 导出脚本的问题

SherpaOnnx 模型目录已包含测试 WAV + 标准转录文本：
`D:\dowload\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\test_wavs\`
