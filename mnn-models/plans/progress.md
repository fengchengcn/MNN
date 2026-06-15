---
date: 2026-06-15
status: active
tags: [qwen3-asr, progress, milestone, index]
category: plan
aliases: [项目进度, Progress, Master Index]
related: [[llmexport-migration]], [[omni-streaming]], [[root-cause-analysis]], [[android-memory]], [[omni-parameters]], [[analysis/fbank-numerical-analysis]], [[replicate-onnx-export-plan]]
---
# Qwen3-ASR → MNN 项目状态

> 最后更新：2026-06-15 | 状态：**双模型 AE 导出完成 — PE 重复模式修复，长音频幻觉已解决 ✓**

## 当前状态总览

```
Qwen3-ASR-0.6B → MNN 迁移: ✅ 完成
├── 旧引擎 (qwen3_asr_engine.cpp):   ✅ 冻结（功能完整，不再开发）
├── Omni 引擎 (omni.cpp):            ✅ 生产可用（推荐）
├── llmexport.py 导出 (WP1-WP6):     ✅ 完成
├── Android 集成 (VoiceChatPresenter): ✅ Omni 模式运行中
├── 流式推理 (Phase 2.6 VAD+扩展窗口): ✅ 实机验证通过
├── 双模型 AE (conv_frontend+encoder): ✅ 长音频幻觉已修复  ← 2026-06-15 更新
└── 单模型 audio.mnn:                 📦 已被双模型替代
```

## 模型文件

### 旧引擎导出（ONNX→MNNConvert，已冻结）

| 文件 | 大小 | 说明 |
|------|------|------|
| `audio_encoder.mnn` | 190 MB | AE 单文件（INT8） |
| `llm_kv_8bit.mnn` + `.weight` | 0.5 MB + 575 MB | Decoder（INT8） |
| `embeddings_bf16.bin` | 297 MB | 外部嵌入表（mmap） |
| `tokenizer.txt` | 3 MB | BPE tokenizer |

### Omni 引擎导出（llmexport.py，当前使用）

| 文件 | 大小 | 说明 |
|------|------|------|
| `conv_frontend.mnn` | 14 KB | **双模型 conv frontend**（chunk=100, fold-into-batch） |
| `conv_frontend.mnn.weight` | 11 MB | conv frontend 权重（INT8） |
| `encoder.mnn` | 335 KB | **双模型 encoder**（repeating PE 0..12 per chunk） |
| `encoder.mnn.weight` | 189 MB | encoder 权重（INT8） |
| `llm.mnn` + `.weight` | 494 KB + 604 MB | Decoder（INT8），29×FusedAttention |
| `config.json` | ~1 KB | `audio_model: "conv_frontend.mnn"`, `audio_encoder: "encoder.mnn"` |
| `tokenizer.txt` / `.mtok` | ~3 MB | BPE tokenizer |

> **2026-06-15 PE 修复**：encoder PE 从连续递增改为按 chunk 重复 0..12，与训练一致。修复了长音频（≥16s）幻觉问题。详见 [[replicate-onnx-export-plan#实施结果（2026-06-15-完成）]]

## 关键里程碑

| 日期 | 事件 |
|------|------|
| 06-04 | 初步调研完成，确认 MNN 框架选型 |
| 06-05 | 旧引擎 ONNX 导出成功，首次 x86 推理 |
| 06-06 | KV Cache + 8-bit 量化实现（7.5x decode 加速） |
| 06-07 | Android 端到端推理跑通（经 5 轮 OOM/乱码/SELinux 修复） |
| 06-08 | 流式解码 + FP16 优化实机验证（~20 tok/s） |
| 06-09 | llmexport.py 迁移计划制定 + 上游 merge |
| 06-09 | 系统提示词更新 + AudioEffect 泄漏修复 + RMS 统一 |
| 06-10 | Omni 引擎模型导出成功（INT8, 814MB），x86 验证通过 |
| 06-10 | Android Omni 集成完成（三模式自动检测：Omni > Old > Sherpa） |
| 06-14 | **AEC/NS 精度根因确认**：Android `AcousticEchoCanceler` + `NoiseSuppressor` 是识别精度最大杀手。移除后 MNN 精度对齐 sherpa-onnx。详见 [[analysis/fbank-numerical-analysis]] §0 |
| 06-14 | **INT8 vs FP16 桌面端输出对比**：4 配置 × 5 音频全链路测试，不同权重格式各有输出差异，无 ground truth 无法判断孰优孰劣（手机实测 FP16 更好） |
| 06-14 | **桌面对比实验：AE 隔离**：MNN AE → ONNX Dec vs ONNX AE → ONNX Dec，5/5 first token MATCH，AE 不是误差源 |
| 06-14 | **桌面对比实验：Decoder 同输入对比**：MNN llm.mnn vs ONNX llm.onnx，first token 5/5 match，但存在系统性数值缩放差异（cosim ~0.97，logit 差 ~4-5×） |
| 06-14 | **🔴 根因确认：AE 架构不等价** ~~sherpa-onnx conv_frontend+encoder vs MNN audio.mnn，cosim ~0.30，subsampling ratio 不同（6.5× vs 8×）。同一份 HF 权重（max diff=0），但 sherpa-onnx 通过图追踪保留了原模型的 Pad→Conv→Slice 链，MNN llmexport 手写 forward() 缺失 Pad/Slice → 图结构不对齐。详见 [[root-cause-analysis]]~~ **🔄 已更正（见下方 06-14 更正条目）** |
| 06-14 | **方案三验证通过**：MNNConvert 转换 sherpa conv_frontend.onnx + encoder.int8.onnx → MNN 格式成功。conv_frontend cosim=1.0，encoder cosim=0.997。MNNConvert 忠实保留了 130 节点动态图（Shape/Gather/Pad/Transpose/Conv/Slice） |
| 06-14 | **Step 3 完成：Express API 双模型串联验证** — conv_frontend.mnn + encoder.mnn 通过 Module::onForward (NCHW) 链式调用，cosim 与 Session API 完全一致（1.0 / 0.997）。原 test_mnn_models.cpp 失败是因为链接旧版 MNN 库 |
| 06-14 | **Step 4 完成：端到端 First Token 对比** — 同一 FBank → MNN AE → ONNX Decoder vs ONNX AE → ONNX Decoder。first token 完全一致（15 vs 15），top-5 交集 5/5，logit cosim=1.0。尽管长序列 (T=300) 时 AE cosim 降至 0.989，Decoder cross-attention 完全补偿 |
| 06-14 | **双模型 AE 实机部署完成** — 代码改动 ~45 行（omni.cpp/hpp + llmconfig.hpp + Qwen3AsrTestActivity.kt + config.json）。手机 logcat 确认新路径生效：fbank → Permute({0,2,1}) → conv_frontend.mnn → encoder.mnn → decoder，AE 耗时 ~1.4s。向后兼容：encoder.mnn 缺失时自动回退旧路径 |
| 06-14 | **发现官方 modeling 代码**：`github.com/QwenLM/Qwen3-ASR` 含完整 `modeling_qwen3_asr.py`（80 行 forward），手写版漏掉 Chunk/Pad/Slice/Window。当前双模型 AE 来自第三方 ONNX → MNNConvert。正道是修复 llmexport.py → 单文件 audio.mnn，全链路自控。详见 [[analysis/export-pipeline-analysis]] |
| 06-14 | **🔄 根因更正：推翻"架构不等价"** — cosim ~0.30 的真凶是两个代码 bug（`conv_out` random bias + PE interleaved 公式），不是 Chunk/Pad/Slice 缺失。修复后 cosim = 0.993（同帧数），PyTorch vs ONNX max diff = 2.36e-6。Chunk/Pad/Slice 是官方长音频内存优化，ONNX 不可 trace，对短音频（≤30s）无需。详见 [[analysis/root-cause-analysis#更正说明]] |
| 06-14 | **单模型 audio.mnn 导出完成** — llmexport.py 导出单文件 audio.mnn (337KB) + audio.mnn.weight (210MB)，含完整的 Conv2d×3 → Transformer×18 → Project。配置：`audio_model: "audio.mnn"`，无 `audio_encoder` → 单模型路径 |
| 06-14 | **llmconfig.hpp 默认值修复** — `audio_encoder()` 默认值 `"encoder.mnn"` → `""`，防止 config.json 无 `audio_encoder` 字段时误加载旧 encoder.mnn 走双模型路径导致 SIGSEGV |
| 06-14 | **桌面端验证通过** — MNN audio.mnn vs Wasser1462 cosim 0.993 (T=100)，first frame cosim 0.98+。手机同步完成，待实机 ASR 测试 |
| 06-14 | **手机实机测试** — 短音频（≤10s）识别正常；长音频（≥16s）出现语音幻觉（FP16/INT8 均复现）。**当时猜测根因**：非分块连续 conv 与训练时分块 conv 不一致。缓解：VAD 模式天然切短段 ⭕ 。**（06-15 更正：真正根因是 PE 模式错误，见下方 06-15 条目）** |
| 06-14 | **0.6B-FP16 + 1.7B-INT8 重新导出** — 应用 bias=False + PE concat 修复，旧文件清理 ~453 MB。三个模型目录全部更新为 llmexport.py 正道单模型 |
| 06-15 | **🔴 根因定位：Positional Encoding 模式错误** — 训练时 PE 按 chunk 重复 0..12；双模型 encoder 连续递增 0..seq_len-1。长音频 PE 外推 15×+ → 注意力崩溃 → 幻觉。Wasser1462 encoder ONNX 通过 Mod 算子实现重复 PE |
| 06-15 | **A/B 测试确认** — 逐层替换 Wasser1462 ONNX 文件，定位问题在 encoder PE（非 conv_frontend）。完整 Wasser1462 双模型通过长音频测试，排除 decoder/C++ 路径问题 |
| 06-15 | **PE 修复 + chunk_size=100 + 无 ghost frame Slice** — encoder.forward() 改为 `repeat(PE[0:13], N)[:seq_len]`，完美复刻训练时的按 chunk 重置 PE 模式。chunk_size 确认 100（非 500），保留 partial chunk 全量输出帧 |
| 06-15 | **✅ 长音频修复验证通过** — 手机实机测试，≥16s 长音频识别正常，幻觉消失。双模型方案正式完成。详见 [[replicate-onnx-export-plan]] |

### 12. AEC + NoiseSuppressor 是精度最大杀手（2026-06-14，🔴 P0）

- **现象**: MNN ASR 识别精度显著差于 sherpa-onnx，即使 fbank 已通过 kaldi-native-fbank 对齐
- **根因**: `Qwen3AsrTestActivity.initAudioRecord()` 开启了 `AcousticEchoCanceler` + `NoiseSuppressor`。这些 Android 硬件 DSP 音效为**语音通话**优化（窄带），用于 ASR（全频带）会导致：
  1. 高频辅音 (`/s/`, `/f/`, `/sh/`) 被噪声抑制误删 → 音素识别错误
  2. 动态范围被 AEC 压缩 → FBank formant 跟踪失效
  3. 非线性 DSP 谐波 → 引入训练数据中不存在的频谱特征
  4. 不同手机的 DSP 质量差异大 → 精度不可预测
- **修复**: 移除 AEC/NS 初始化代码，与 sherpa-onnx 的 `AudioRecorder.kt` 完全对齐：raw MIC → PCM → Float → fbank，无任何硬件音效。
- **实机验证**: BATCH 模式 + 无 AEC/NS 后识别质量大幅提升
- **教训**: ASR 的音频链路必须与训练 pipeline 完全一致。任何"增强"（降噪、回声消除、AGC）都是偏差，不是优化。sherpa-onnx 的 `AudioRecorder.kt` 没有任何预处理是正确的参考实现。
- **代码位置**: `Qwen3AsrTestActivity.kt:initAudioRecord()` — AEC/NS 于 2026-06-14 移除

## GPU 后端实测结论

在 Kirin 990 (Mali-G76) 上实测：

| 后端 | Prefill | Decode | 判断 |
|------|---------|--------|------|
| **CPU** | 175 t/s | 21 t/s | ✅ 唯一可用 |
| OpenCL | 2.9 t/s (60×慢) | 1.7 t/s (12×慢) | ❌ 不可用 |
| Vulkan | 卡死 | 卡死 | ❌ 不可用 |

**结论**: 移动端 LLM Decode 是 memory-bound 任务，GPU kernel 启动/同步开销远超实际计算。
CPU + 4 线程 + FP16 是最优配置。

## 重要踩坑汇总

### 1. Android OOM（最严重）
- **现象**: lmkd 在模型加载期间强杀进程（RSS 3.26 GB）
- **根因**: AE + Decoder 双模型同时驻留 + embedding 全量加载 622 MB
- **修复**: 延迟加载 + 串行化 + mmap embedding + `setHint` 权重 mmap
- **效果**: 峰值从 ~3.2 GB → ~800 MB

### 2. LLM_SUPPORT_AUDIO 宏缺失
- **现象**: 模型 "光速加载"（10ms），输出 "no speech detected"
- **根因**: CMakeLists.txt 未定义 `LLM_SUPPORT_AUDIO`，解码器编译为空壳
- **修复**: `target_compile_definitions` 添加宏定义

### 3. Tokenizer 乱码
- **现象**: 中文识别结果为乱码
- **根因**: `tokenizer.txt` 是 MNN SentencePiece 二进制格式，代码按纯文本逐行读取
- **修复**: 使用 `MNN::Transformer::Tokenizer::createTokenizer()` 正确解析

### 4. SELinux WAV 写入拒绝
- **现象**: `Failed to write temp WAV (errno=13 EACCES)`
- **根因**: SELinux 禁止 untrusted_app 写 `/data/local/tmp/`
- **修复**: WAV 写到 app cache 目录

### 5. 多线程并发 runDecoder() → SIGSEGV
- **现象**: 第 3 次 utterance 在 `MNN::ThreadPool::enqueueInternal` 崩溃
- **根因**: 两个 silence-detection 线程 4ms 内同时触发 `endAudio()`
- **修复**: `std::mutex` + `std::try_to_lock`

### 6. Audio encoder segfault（llmexport.py 导出）
- **现象**: Omni 加载 audio.mnn 后 response() 崩溃
- **根因**: `--transformerFuse` 融合了 AE 内部 18 层 encoder-only Transformer
- **修复**: `export_audio()` 设置 `transformer_fuse=False`

### 7. EOS-only 输出（llmexport.py 导出）
- **现象**: 模型只输出 `<|im_end|>` 后停止
- **根因**: 缺少 Jinja chat template，Omni 无法解析 role markers
- **修复**: config.json 添加 Qwen 格式 jinja template

### 8. VAD 纯分段导致准确率崩溃（Phase 2.5 教训）
- **现象**: 罕见/技术术语识别准确率从 ~95% 降至 ~30%
- **根因**: Qwen3-ASR AE 使用 Full Bidirectional Attention，分段独立推理导致上下文断裂
- **教训**: VAD 控制生命周期 + 扩展窗口保留全量上下文，两者缺一不可
- **修复**: Phase 2.6 恢复扩展窗口，VAD 仅作生命周期控制

### 11. 累积滑动窗口实现（2026-06-13，Qwen3AsrTestActivity）

**问题**: Phase 2.6 的"VAD 引导扩展窗口"在 Android 端实际上未正确实现。
`Qwen3AsrTestActivity` 中 VAD 模式将每段音频独立发给 `processSegmentSync()` 推理，
每段各自过一遍 AudioEncoder + TextDecoder，段间无任何上下文共享。

**正确实现**:
- VAD 检测语音段 → 拼入 `accumulatedSegments` 累积 buffer
- 段间插入 0.4s 零填充静音（匹配 VAD `minSilenceDuration`）
- 每新增一段，构建全部累积音频，发一次推理（累积滑动窗口）
- 结果：2s → 4s → 6s 渐进式结果，每次都有完整上下文，后续推理可自动纠正前文

**句间断句**: 通过 wall-clock 段间间隔区分句内/句间停顿
- 间隔 < 1.5s：同一句子，继续累积
- 间隔 ≥ 1.5s：句间边界 → `flushCurrentSentence()` 发最终推理，新句从零开始

**并发模型**: `Channel.UNLIMITED` + 单 consumer 协程串行处理，`sendCumulativeTask()` 调用 `trySend` 不阻塞录音线程。

**关键 Bug 及修复**:
1. **重复输出**: 句边界 flush 时发了 final 任务，但最后一个 interim 已覆盖全部累积音频。修复：用 `lastInferenceSegmentCount` 跟踪上次推理时段数，flush 时段数未增加 → 跳过 final
2. **最后一句不输出**: flush 时 drain channel 排空了包括当前句子在内的所有待处理任务，然后因 `lastInferenceSegmentCount` 已更新又跳过 final → 整句丢失。修复：去掉 channel drain
3. **句子/轮次标注错乱**: `sentenceIndex`/`cumulativeRound` 是共享变量，consumer 读取时已被后续 flush 修改。修复：构建 `SegmentTask` 时捕获 `sentIndex`/`sentRound` 到任务中

### 9. keepHistory 污染 prompt（Omni 模式）
- **现象**: 增量推理中 FINAL 结果输出幻觉文本
- **根因**: `keepHistory=true` 导致每次 `generate()` 追加 `<audio>` tag 到历史
- **修复**: `setKeepHistory(false)`

### 10. AudioEffect 泄漏
- **现象**: AEC/NS 对象创建后引用被丢弃
- **修复**: 3 个 Kotlin 文件补上显式 `release()` + `null` 清理

## 降噪架构（二层，AEC/NS 已移除）

> **2026-06-14 更新**: Layer 2 (AEC/NS) 已从 ASR 路径移除。识别精度测试证实 AEC/NS 是最大精度杀手。
> 详见 Pitfall #12 和 [[analysis/fbank-numerical-analysis]] §0。

```
Layer 1: AudioSource.MIC → 无硬件 DSP（与 VOICE_COMMUNICATION 不同，MIC 不激活 AGC/AEC/NS）
Layer 2: ~~Android AudioEffect API~~ → ❌ 已移除（AcousticEchoCanceler + NoiseSuppressor，2026-06-14）
Layer 3: Silero VAD (神经网络) → silero_vad.onnx 通过 MNN Interpreter 推理
C++ 引擎: whisper_fbank_knf() — kaldi-native-fbank，preemphasis=0.97, HTK mel
```

> Silero VAD 已取代早期 Phase 2.5 的 RMS 能量 VAD。Silero VAD 是 LSTM 模型（V4/V5），
> 通过 `silero_vad_jni.cpp` → `Vad.kt` 集成，提供比 RMS 门限更准确的语音活动检测。

## Sherpa AE 双模型集成（2026-06-14 部署，2026-06-15 llmexport.py 自控替代）

> **更新 2026-06-15**：Wasser1462 ONNX 文件（conv_frontend.onnx + encoder.int8.onnx）仅用于调试对照，
> 正式部署使用 llmexport.py 自控导出的双模型。PE 修复后自研模型与 Wasser1462 精度一致。

### 架构

```
FBank [1, 128, T] ──Permute({0,2,1})──→ [1, T, 128]
    │
    ▼
conv_frontend.mnn  (42 MB, MNNConvert)
    │  Pad → Conv2d×3 → Slice
    │  输入 [1, T, 128]  →  输出 [1, T', 896]
    ▼
encoder.mnn  (176 MB, MNNConvert)
    │  18×Transformer INT8
    │  输入 [1, T', 896] + mask [1, T']
    │  输出 [1, T', 1024]
    ▼
_Permute({1,0,2})  →  [T', 1, 1024]  →  Decoder (llm.mnn)
```

### 验证数据

| 指标 | 短序列 (T=100) | 长序列 (T=300) |
|------|:------------:|:------------:|
| conv_frontend cosim | 1.000000 | — |
| encoder cosim | 0.996771 | 0.988756 |
| Decoder first token match | — | ✅ (15 vs 15) |
| Decoder top-5 intersection | — | 5/5 |
| Logit cosim | — | 1.000000 |

### 代码改动

| 文件 | 改动 | 行数 |
|------|------|:----:|
| `llmconfig.hpp` | `audio_encoder()` 方法 | +4 |
| `omni.hpp` | `mAudioEncoder` 成员 + 析构 | +2 |
| `omni.cpp` | 双模型加载 + qwen3_asr 推理分支 | +40 |
| `Qwen3AsrTestActivity.kt` | 扫描 conv_frontend.mnn | +2 |
| `config.json` | `audio_model`→`conv_frontend.mnn`, `audio_encoder` | ~1 |

### 模型文件变更

| 状态 | 文件 | 大小 |
|------|------|------|
| ➕ 新增 | `conv_frontend.mnn` | 42 MB |
| ➕ 新增 | `encoder.mnn` | 176 MB |
| ➖ 删除 | `audio.mnn` (旧手写版) | -337 KB |
| ➖ 删除 | `audio.mnn.weight` (旧权重) | -210 MB |
| | **净变化** | **+8 MB (+3.8%)** |

### 向后兼容

- `audio_encoder` 加载失败非致命（仅 log 警告）
- 推理时检查 `mAudioEncoder.get() != nullptr`，缺失时自动回退旧单模型路径
- 旧 `audio.mnn` 保留为 `audio.mnn.backup`

## 相关文档

| 文档 | 说明 |
|------|------|
| [[sherpa-ae-mnn-integration]] | **NEW** Sherpa AE MNN 集成方案 & 验证结果 |
| [[analysis/export-pipeline-analysis]] | **NEW** 导出链路分析：双模型 AE vs llmexport 正道 |
| [[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]] | Omni 迁移计划（WP1-WP6，已完成） |
| [[Qwen3-ASR-OMNI-STREAMING-PLAN]] | Omni 流式方案（Phase 1/2/2.6 完成） |
| [[ANALYSIS]] | 识别精度差异分析（已定位到模型导出质量） |
| [[Qwen3-ASR-ANDROID-MEMORY-ANALYSIS]] | 实机内存 + GPU 后端实测数据 |
| [[Qwen3-ASR-MEMORY-ANALYSIS]] | 旧路径 vs Omni 内存对比分析 |
| [[Qwen3-ASR-OMNI-PARAMETERS]] | Omni 推理参数详解 |
