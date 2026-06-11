# Qwen3-ASR → MNN 项目状态

> 最后更新：2026-06-11 | 状态：**Omni 引擎迁移完成，旧引擎已冻结**

## 当前状态总览

```
Qwen3-ASR-0.6B → MNN 迁移: ✅ 完成
├── 旧引擎 (qwen3_asr_engine.cpp):   ✅ 冻结（功能完整，不再开发）
├── Omni 引擎 (omni.cpp):            ✅ 生产可用（推荐）
├── llmexport.py 导出 (WP1-WP6):     ✅ 完成
├── Android 集成 (VoiceChatPresenter): ✅ Omni 模式运行中
└── 流式推理 (Phase 2.6 VAD+扩展窗口): ✅ 实机验证通过
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
| `audio.mnn` + `.weight` | 350 KB + 210 MB | AE 分离权重（INT8） |
| `llm.mnn` + `.weight` | 494 KB + 604 MB / 1.1 GB | Decoder（INT8 / FP16），29×FusedAttention |
| `config.json` | ~1 KB | 含 `is_audio`, `audio_type`, `jinja` template |
| `tokenizer.txt` / `.mtok` | ~3 MB | BPE tokenizer |

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
| 06-11 | Phase 2.6 VAD 引导扩展窗口实机验证，FP16+greedy 模型发布 |
| 06-11 | 旧引擎 Qwen3AsrEngine 移除，仅保留 Omni |

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

### 9. keepHistory 污染 prompt（Omni 模式）
- **现象**: 增量推理中 FINAL 结果输出幻觉文本
- **根因**: `keepHistory=true` 导致每次 `generate()` 追加 `<audio>` tag 到历史
- **修复**: `setKeepHistory(false)`

### 10. AudioEffect 泄漏
- **现象**: AEC/NS 对象创建后引用被丢弃
- **修复**: 3 个 Kotlin 文件补上显式 `release()` + `null` 清理

## 降噪架构（三层）

```
Layer 1: 硬件 DSP (VOICE_COMMUNICATION 音频源) → 高通/MTK 自带 AEC+NS+AGC
Layer 2: Android AudioEffect API → AcousticEchoCanceler + NoiseSuppressor
Layer 3: Silero VAD (神经网络) → silero_vad.onnx 通过 MNN Interpreter 推理
C++ 引擎: whisper_fbank() — 无额外处理
```

> Silero VAD 已取代早期 Phase 2.5 的 RMS 能量 VAD。Silero VAD 是 LSTM 模型（V4/V5），
> 通过 `silero_vad_jni.cpp` → `Vad.kt` 集成，提供比 RMS 门限更准确的语音活动检测。

## 相关文档

| 文档 | 说明 |
|------|------|
| [[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]] | Omni 迁移计划（WP1-WP6，已完成） |
| [[Qwen3-ASR-OMNI-STREAMING-PLAN]] | Omni 流式方案（Phase 1/2/2.6 完成） |
| [[ANALYSIS]] | 识别精度差异分析（已定位到模型导出质量） |
| [[Qwen3-ASR-ANDROID-MEMORY-ANALYSIS]] | 实机内存 + GPU 后端实测数据 |
| [[Qwen3-ASR-MEMORY-ANALYSIS]] | 旧路径 vs Omni 内存对比分析 |
| [[Qwen3-ASR-OMNI-PARAMETERS]] | Omni 推理参数详解 |
