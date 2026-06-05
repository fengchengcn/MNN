# Qwen3-ASR-0.6B MNN 迁移：初步调研与方案设计

> 目标：将 Qwen3-ASR-0.6B 模型转换为 MNN 格式，集成到 MnnLlmChat Android 应用中，实现端侧离线流式语音识别。
>
> **第三轮更新**（2026-06-04）：已 merge upstream/master（215 commits → v3.5.0）。上游新增 Gemma4Audio/Lfm2Audio 等 audio encoder 导出模式、engine 端 audio_type 分发框架、Android AEC/NS 支持。这些可大幅复用，工作量下调至 **7-10 天**。详见 §3.2 和 §4。

---

## 1. 项目现状

### 1.1 MnnLlmChat Android 应用

- 全功能多模态 LLM 安卓应用，支持文本/图像/音频/视频
- 已有语音对话功能（Voice Chat），集成 ASR + TTS
- 当前 ASR 方案：**sherpa-mnn**（JNI 集成 `libsherpa-mnn-jni.so`）
- 支持的 ASR 模型类型：**Zipformer transducer**（encoder/decoder/joiner 三件套 + tokens.txt）
- ASR 模型通过 ModelMarket 远程下载，存储在本地目录
- 配置格式：`config.json` 指定 transducer 的 encoder/decoder/joiner 路径和 tokens

### 1.2 MNN LLM 导出工具

- `transformers/llm/export/llmexport.py`：LLM 模型导出工具
- 已支持的模型类型（部分）：qwen, qwen2, qwen3, qwen3_moe, llama, gemma2/3, deepseek-vl, internvl, minicpmv, gpt_oss 等
- 已支持的音频多模态：qwen2_audio, qwen2_5_omni（audio encoder 独立导出为 `audio.onnx`）
- 导出流程：torch → ONNX → MNN（可选 4-bit/8-bit 量化）
- 导出产物（标准 LLM）：`llm.mnn` + `llm.mnn.weight` + `embeddings_bf16.bin` + `tokenizer.txt` + `config.json`

### 1.3 MNN 推理引擎

- `transformers/llm/engine/`：LLM 推理引擎 C++ 实现
- 支持的 decoder 架构：Llama/Qwen2/Qwen3 系列（RMSNorm, GQA, SwiGLU, RoPE/MRoPE）
- **关键发现：Omni 类（`omni.cpp`）已有完整的音频多模态推理管线！**：
  - `MNN::AUDIO::whisper_fbank()` — Whisper 风格 Mel 滤波器组特征提取（128 维，16kHz）✅
  - `mAudioModule` — 加载和运行音频编码器 MNN 模型 ✅
  - `audioProcess()` — 音频 → fbank → encoder → embedding 的完整流程 ✅
  - **Embedding 注入机制** — 将音频 embedding 替换 LLM token 序列中特定 token 的 embedding ✅
  - 已有 Qwen2-Audio、Qwen2.5-Omni 的完整支持代码
- Qwen3 decoder 结构已在 llm.cpp 中完整支持（QK Norm、GQA、SwiGLU、MRoPE）
- OpenCL/Vulkan GPU 加速、量化推理（4-bit/8-bit）均已就绪

### 1.4 当前 ASR 模型的局限

sherpa-mnn 仅支持 **Zipformer transducer** 模型，特点是：
- 声学编码器 + 预测解码器 + Joiner 网络
- 使用 RNN-T loss 训练
- 模型文件：`encoder-*.mnn` + `decoder-*.mnn` + `joiner-*.mnn` + `tokens.txt`
- **不支持 Whisper 架构、不支持 LLM-based decoder**

---

## 2. Qwen3-ASR-0.6B 模型架构

### 2.1 总体结构

```
Qwen3ASRForConditionalGeneration
  └── thinker
        ├── audio_tower (Qwen3ASRAudioEncoder)  ← 音频编码器
        ├── model (Qwen3Model, 28层 Transformer)  ← LLM 解码器
        ├── embed_tokens                           ← 词嵌入
        └── lm_head                                ← 输出投影
```

推理流程：

```
音频(16kHz) → Mel特征(128维) → Audio Encoder → 投影层 → prefix embeddings(1024维)
                                                                    ↓
                文字输出 ← LLM Decoder (自回归) ← [prefix_embeddings | text_tokens]
```

**关键性质**：decoder 使用**标准 causal self-attention**（无跨注意力），encoder 输出作为 soft prompt prefix 直接拼接到文本 token embeddings 前面。

### 2.2 Audio Encoder（Whisper-style）

| 参数 | 规格 |
|------|------|
| 架构 | 3层 Conv2d 下采样 + 18层 Transformer Encoder |
| 隐藏维度 | 896 |
| 注意力头数 | 14 |
| FFN 维度 | 3584 |
| 位置编码 | 正弦位置编码（运行时计算，非可学习） |
| 下采样倍率 | 8x（3层 Conv2d, stride=2） |
| 输出帧率 | 12.5Hz（16kHz 输入） |
| 参数量 | ~180M |

**前处理**：音频 → 128维 Mel 滤波器组特征 → Conv2d 下采样

**后处理（投影层）**：LayerNorm → GELU → Linear(896→896) → Linear(896→1024)

### 2.3 LLM Decoder（Qwen3-style）

| 参数 | 规格 |
|------|------|
| 层数 | 28 |
| 隐藏维度 | 1024 |
| 注意力头数 | Q: 16, KV: 8（GQA 2:1） |
| head_dim | 128 |
| 位置编码 | MRoPE（mrope_section=[24,20,20]，纯音频下退化为标准 RoPE） |
| RoPE theta | 1,000,000（NeoX-style） |
| MLP 类型 | SwiGLU（gate_proj + up_proj + down_proj, intermediate=3072） |
| Norm | RMSNorm（eps=1e-6） |
| Q/K Norm | 每头 RMSNorm（Qwen3 特有） |
| 词表大小 | 151,936 |
| Tied Embeddings | ✓（embed_tokens == lm_head） |
| 参数量 | ~600M（纯 decoder） |

### 2.4 与已有支持模型的对比

| 特性 | Qwen3-0.6B（已支持） | Qwen3-ASR-0.6B |
|------|----------------------|-----------------|
| 架构 | Decoder-only | Encoder-Decoder |
| Decoder 结构 | Qwen3（QK Norm, GQA, SwiGLU） | **相同** |
| 词表 | 151,936 | 151,936 |
| hidden_size | 1024? | 1024 |
| 特殊之处 | 无 | 多了 audio encoder（~180M） + 投影层 |

**结论**：Decoder 部分与标准 Qwen3 结构完全一致，现有 MNN Qwen3 导出路径可以复用。

### 2.5 与已有音频多模态模型的对比

| 特性 | Qwen2-Audio（已支持） | Qwen3-ASR |
|------|----------------------|------------|
| 音频编码器 | Whisper-style | Whisper-style（更深的 18 层） |
| 编码器输出用法 | 替换 `<\|AUDIO\|>` token | 作为 prefix embeddings |
| 编码器导出 | 独立 audio.onnx | 同样可行 |
| 推理流程 | 一次编码 + LLM 生成 | 一次编码 + LLM 生成 |

**结论**：音频编码器的处理方式可以参考 Qwen2-Audio 的已有实现。

---

## 3. 方案设计

### 3.1 总体思路

```
┌─────────────────────────────────────────────────────────────────┐
│                        推理流水线                                │
│                                                                 │
│  音频流 ──→ Mel特征提取 ──→ Audio Encoder (MNN) ──→ 投影层      │
│                                                       │         │
│                                              prefix embeddings  │
│                                                       │         │
│  文本提示 ──→ Tokenizer ──→ Token IDs ──→ [prefix | tokens]     │
│                                                   │             │
│                                          LLM Decoder (MNN)      │
│                                                   │             │
│                                            自回归生成文本         │
└─────────────────────────────────────────────────────────────────┘
```

- **Audio Encoder**：导出为独立 MNN 模型（`audio_encoder.mnn`）
- **LLM Decoder**：复用已有 Qwen3 导出流程（`llm.mnn`）
- **推理引擎**：新增 encoder-decoder 串联逻辑

### 3.2 实现路径

**开发顺序：POC 先行，再建 Service。**

```
POC（0.5-1 天）            →  模型导出 + 工程化（3-4 天）  →  Android Service（2-3 天）
├─ CPU RTF 性能测试            ├─ encoder 导出完善           ├─ AIDL session+pushAudio
├─ inputs_embeds 已确认 ✅     ├─ decoder 导出（10行配置）    ├─ 独立进程 :asr
└─ decoder 架构已确认 ✅       ├─ 引擎集成                   ├─ 权限安全
                               └─ JNI 封装                  └─ 多客户端管理
```

**策略**：最大化复用 upstream v3.5.0 的新增能力。

#### 阶段零：POC 验证（~0.5-1 天）

inputs_embeds 注入已源码确认 ✅。Decoder 就是标准 Qwen3 架构（RMSNorm+GQA+QK Norm+SwiGLU），唯一差异是权重前缀从 `model.layers.*` 变为 `thinker.model.layers.*`，10 行 model_mapper 配置即可解决，无需单独验证。

POC 只验证 1 个假设：

**CPU 实时性能测试**
- 在 Ubuntu 上测 encoder + decoder 的 RTF（用 4-bit 量化）
- 如果服务器 CPU 上 RTF 已经 > 0.3，移动端基本没戏
- **通过标准**：RTF < 0.15（服务器端），预留 2-4x 移动端退化余量
- 通过后直接进入阶段一，不通过则评估 GPU 卸载方案

#### 阶段一：环境准备（~0.5 天）

- 在 Ubuntu 服务器上通过 ModelScope 下载 Qwen3-ASR-0.6B
- 安装 Python 依赖（torch, transformers, onnx, modelscope）
- 编译 MNNConvert（已有脚本）

#### 阶段二：模型导出（~1.5 天）

**Audio Encoder 导出**（~1 天）— 以 `Gemma4Audio` 为模板：

- 在 `audio.py` 中新增 `Qwen3ASRAudio` 类，继承 `Audio` 基类
  - `load()`：加载 `thinker.audio_tower`（18 层 Whisper-style encoder + 投影层）
  - `forward()`：Conv2d 下采样 → Transformer Encoder → LayerNorm → GELU → Linear → Linear
  - 注册到 `Audio.get_audio()`：`'qwen3_asr_audio_encoder': Qwen3ASRAudio`
- 导出 `audio_encoder.mnn`
- 参考：`Gemma4AudioExportModel` 的 wrapper 模式处理 export 时的 forward 签名

**LLM Decoder 导出**（~0.5 天）：

- 在 `model_mapper.py` 中新增 `regist_qwen3_asr()`，映射 `thinker.model.*` 权重路径
- 注册 `model_type = 'qwen3_asr'`，decoder 结构复用 `qwen3` 的 attention map（QK Norm、GQA、SwiGLU）
- 使用现有 `llmexport.py --export mnn` 导出 `llm.mnn`

**Tokenizer**：标准 Qwen3 tokenizer，无需额外工作。

#### 阶段三：推理引擎实现（~2 天）

在 `omni.cpp` 中新增 `"qwen3_asr"` 处理分支：

- `audio_type == "qwen3_asr"` 时：
  1. 调用 `whisper_fbank()`（已有 ✅）
  2. 运行 `mAudioModule->forward()`（已有模式 ✅）
  3. 将 encoder 输出作为 **prefix embeddings**（非替换式注入）
  4. 拼接 `[prefix_embeddings | text_token_embeddings]`
  5. 走标准 `Llm::generate()` decode（已有 ✅）
- 流式支持：音频分块 → encoder 增量推理 → decoder 增量生成
- 参考：`Omni::audioProcess()` 的 embedding 注入逻辑，改为 prefix 方式

#### 阶段四-六：编译、集成、测试

- MNN 编译：`-DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true`，Android arm64-v8a
- Android 集成：JNI 封装 Asr 类，MnnLlmChat 新增 ASR 引擎类型（与 sherpa-mnn 共存）
- 测试：AEC/NS 已由 upstream 集成到 AsrService，直接复用

### 3.3 关键技术风险（第三轮更新）

| 风险 | 等级 | 说明与缓解 |
|------|------|-----------|
| **inputs_embeds 注入能力** | 🟢 **已确认支持** | ✅ 源码验证：`Llm::forward(MNN::Express::VARP input_embeds)`（`llm.cpp:608`）直接接受外部 embeddings 张量。Talker 类已在生产中使用（`omni.cpp:2047` `forward(input_embeds)`）。Qwen3-ASR 只需 `Concat([audio_embeddings, text_token_embeddings])` 后调用此接口即可。**零风险**。 |
| Audio Encoder 与标准 Whisper 的差异 | 🟡 中 | Qwen3-ASR encoder 非标准 Whisper：3 层 Conv2d 下采样（vs 2 层）、动态注意力窗口。导出时需逐层验证与 PyTorch 参考输出的一致性 |
| Android CPU 实时性能 | 🟡 中 | 600M decoder 纯 CPU 推理的流式 RTF 需实测。官方数据 4-bit 下 RTF=0.02（非手机环境）。移动端散热/降频可能退化 2-4x。目标 RTF < 0.6 |
| Decoder 结构与 Qwen3 的兼容性 | 🟢 低 | 已验证：RMSNorm、GQA、QK Norm、SwiGLU、NeoX RoPE 均在 MNN 中支持 |
| MRoPE 在纯音频场景的推理正确性 | 🟢 低 | MNN 已支持 Qwen3 MRoPE（mrope_section=[24,20,20]），纯音频下退化为标准 RoPE |
| Android 端内存 | 🟢 低 | 4-bit 量化 decoder ~150MB，encoder ~25MB，mmap 加载 |

### 3.4 输出产物清单

转换完成后，每个模型目录应包含：

```
Qwen3-ASR-0.6B-MNN/
├── config.json              # 运行时配置文件
├── audio_encoder.mnn        # 音频编码器 MNN 模型
├── audio_encoder.mnn.json   # 编码器权重元信息
├── audio_encoder.mnn.weight # 编码器权重数据
├── llm.mnn                  # LLM Decoder MNN 模型
├── llm.mnn.json             # Decoder 权重元信息
├── llm.mnn.weight           # Decoder 权重数据
├── embeddings_bf16.bin      # 词嵌入权重
├── tokenizer.txt            # Tokenizer 词表
└── llm_config.json          # Decoder 配置
```

---

## 4. 工作量估算（第三轮更新，POC 先行）

| 阶段 | 预估工作量 | 难度 | 关键依赖/可复用基础 |
|------|-----------|------|-------------------|
| **零：POC 验证** | **0.5-1 天** | 🟡 | CPU RTF 实测（inputs_embeds ✅，decoder = 标准 Qwen3 ✅） |
| 一：环境准备 | 0.5 天 | ⭐ | modelscope 下载、Python 依赖 |
| 二：模型导出 | **1.5 天** | ⭐⭐ | `Gemma4Audio` 模板、Qwen3 decoder 已验证 |
| 三：推理引擎 | **2 天** | ⭐⭐ | `audio_type` 分发框架已就绪、Llm 基类完整 |
| 四：MNN 编译 | 0.5 天 | ⭐ | CMake 脚本现成 |
| 五：Android 集成 | **2 天** | ⭐⭐⭐ | AEC/NS 已由 upstream 实现、sherpa-mnn 参考 |
| 六：测试优化 | **1.5-2.5 天** | ⭐⭐ | benchmark 工具 |
| **总计** | **8-10 天** | | POC 精简为单项 RTF 测试 |

### 各轮估算对比

| 版本 | 工作量 | 关键变化 |
|------|--------|---------|
| 第一版（初步分析） | 13-19 天 | 误判需要重写 encoder-decoder 引擎 |
| 第二版（深入代码） | 8-12 天 | 发现 Omni 已有音频管线、whisper_fbank 已存在 |
| 第三版（merge upstream） | 7-10 天 | `Gemma4Audio` 提供导出模板、`audio_type` 分发框架就绪 |
| **第四版（源码验证 + POC 最简）** | **8-10 天** | 源码确认 inputs_embeds + decoder 架构，POC 只剩 CPU RTF 一项 |

### 节省工作量的关键复用点

- ❌ **不需要**写 fbank（`MNN::AUDIO::whisper_fbank` 已存在）
- ❌ **不需要**设计 audio encoder 导出架构（继承 `Gemma4Audio` 模式）
- ❌ **不需要**设计 engine 端 audio dispatch（在已有 `audio_type` 框架上扩展）
- ❌ **不需要**实现 AEC/NS（upstream 已加到 AsrService）
- ✅ **只需要**：POC 验证 inputs_embeds 注入 → 实现 `Qwen3ASRAudio` 导出类 → `"qwen3_asr"` engine 分支 → session+pushAudio IPC → App 端录音策略

---

## 5. 上游合并成果（v3.5.0，已 merge ✅）

### 5.1 与方案直接相关的上游新增能力

#### A. Audio Encoder 导出参考模式（`audio.py`）

上游新增 **3 种** audio encoder 类型，为 Qwen3-ASR 导出提供直接蓝图：

| 新增类 | 继承链 | 关键特征 | 对 Qwen3-ASR 的价值 |
|--------|--------|---------|-------------------|
| `FunAudioChatAudio` | Qwen2Audio → Qwen2_5OmniAudio | group pooling + continual_output_matching | 展示了继承扩展模式 |
| `Lfm2Audio` | Audio（基类） | 自定义 preprocessor + `audio_type="conformer"` | 自定义前处理的范式 |
| **`Gemma4Audio`** | Audio（基类） | `audio_tower` + `embed_audio` + `audio_type="usm"` | ⭐ **最接近 Qwen3-ASR**：encoder + projector 组合导出模式 |

**关键复用点**：Qwen3-ASR 的 `thinker.audio_tower`（encoder）+ `thinker.model.embed_tokens`（decoder embedding）结构与 Gemma4Audio 的 `audio_tower` + `embed_audio` 模式高度一致。

#### B. Engine 端 audio_type 分发框架（`omni.cpp`）

```cpp
// 已实现的 dispatch 逻辑，新增 audio_type 即可扩展
if (audio_type == "conformer")  → conformer_fbank()
else if (audio_type == "usm")   → usm_fbank()
else                             → whisper_fbank()  // Qwen3-ASR 走这里！
```

- `whisper_fbank()` 已存在且被 engine 默认使用 — **零工作量** ✅
- 新增 `"qwen3_asr"` audio_type 只需添加一个 else-if 分支
- embedding 注入机制（`mAudioEmbeddings`）已成熟

#### C. Android AsrService 增强

| 新增功能 | 用途 |
|---------|------|
| AcousticEchoCanceler（AEC） | 语音对话场景消除回声 |
| NoiseSuppressor（NS） | 硬件级降噪 |
| `setMuted()` | 静音控制，支持打断场景 |
| `onSpeechDetected` 回调 | 流式语音检测，提升交互体验 |

我们的 Qwen3-ASR 引擎接入后可直接受益于这些硬件能力。

#### D. 其他相关升级

| 特性 | 影响 |
|------|------|
| dflash speculative decoding | 可加速 decoder 推理（ASR 场景可选） |
| prompt cache（multi-turn） | 多轮对话场景复用 prefix 计算 |
| Interleaved Thinker/Talker | 双工语音对话的工程基础 |

### 5.2 方案影响总结

| 维度 | 调整 |
|------|------|
| Audio encoder 导出 | 以 `Gemma4Audio` 为模板，新建 `Qwen3ASRAudio`，减少 ~40% 编码量 |
| Mel fbank | 无需新增，直接使用 `whisper_fbank()` |
| Engine audio dispatch | 在已有框架上新增 `"qwen3_asr"` 分支，改动量 ~20 行 |
| Decoder 导出 | Qwen3 路径无变化，复用 |
| Android 音频硬件 | AEC/NS 已由 upstream 实现，无需额外工作 |
| **总工作量** | 从 8-12 天 → **7-10 天** |

---

## 6. 推理框架选型：MNN vs sherpa-onnx

> 决策背景：Qwen3-ASR 模型将作为系统级 ASR Service 部署在 Android 10 设备上，供多个 App 通过 IPC 调用。

### 6.1 Qwen3-ASR 的计算特征

Qwen3-ASR 不是一个标准 ASR 模型，其推理分为两个阶段：

```
音频(16kHz) → Audio Encoder(180M) → 投影层 → prefix embeddings
                                                   ↓
              文本输出 ← LLM Decoder(600M) ← [prefix | text_tokens]
```

| 组件 | 参数量 | 架构 | 推理耗时占比（估算） |
|------|--------|------|-------------------|
| Audio Encoder | ~180M | Whisper-style Transformer Encoder | 15-20% |
| LLM Decoder | ~600M | Qwen3（GQA + QK Norm + SwiGLU + RoPE） | **80-85%** |

**核心瓶颈在 decoder**——它占了绝大部分计算量，且需要完整的 LLM 推理栈支持（KV-cache、自回归解码、量化推理）。

### 6.2 候选框架对比

#### sherpa-onnx

| 维度 | 评估 | 说明 |
|------|------|------|
| ASR 管线 | ✅ 内置 | Whisper / Zipformer 流式推理开箱即用 |
| Qwen3 LLM Decoder | ❌ 不支持 | sherpa-onnx 没有 LLM 推理能力，600M 的 Qwen3 decoder 无法运行 |
| Audio Encoder | ✅ 支持 | 标准 Whisper encoder 可以运行 |
| ARM CPU 优化 | ⚠️ 通用 | 基于 ONNX Runtime，不如 MNN 针对移动端 ARM 的深度优化 |
| 量化支持 | ⚠️ 有限 | 支持 int8，缺乏 4-bit 等低比特量化 |
| Android Service | ✅ 有 demo | 有现成的 JNI 封装和示例 |

**关键缺陷**：sherpa-onnx 只能跑 audio encoder（~15% 工作），跑不了 LLM decoder（~85% 工作）。必须自己补全 decoder 推理 → 等同于从零实现 LLM 推理引擎。

#### MNN

| 维度 | 评估 | 说明 |
|------|------|------|
| Qwen3 LLM Decoder | ✅ 生产级 | GQA、QK Norm、SwiGLU、MRoPE、KV-cache、dflash speculative decoding |
| Audio Encoder | ⚠️ 待导出 | 需新增 `Qwen3ASRAudio` 类，但有 `Gemma4Audio` 模板可复用 |
| ASR 管线 | ⚠️ 待组装 | 无内置 ASR pipeline，需参照 `Omni` 类实现 |
| ARM CPU 优化 | ✅ 深度优化 | KleidiAI、Neon、SVE2、低比特量化 kernel（2/3/4-bit），llama.cpp 的 8.6x 预填充加速 |
| 量化 | ✅ 丰富 | 4-bit/8-bit 量化，decoder 可从 ~600MB 压缩至 ~150MB |
| 内存管理 | ✅ mmap | 大模型文件 mmap 加载，启动速度快 |
| Android Service | ⚠️ JNI 待写 | 但有 sherpa-mnn 的 `AsrService` 模式可参考 |

### 6.3 决策逻辑

```
                    ┌─────────────────────────────────────┐
                    │ Qwen3-ASR 推理中 85% 的算力在 LLM   │
                    │ Decoder，它需要完整的 LLM 推理栈     │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                                               ▼
    ┌─────────────────┐                             ┌─────────────────┐
    │   sherpa-onnx    │                             │      MNN        │
    │                  │                             │                 │
    │ ❌ 无 LLM 推理    │                             │ ✅ Qwen3 全支持  │
    │ ✅ 有 ASR 管线   │                             │ ⚠️ 需组装 ASR    │
    │                  │                             │                 │
    │ 缺的核心能力:     │                             │ 缺的都可以补:     │
    │ 必须从零实现     │                             │ 有模板可复用     │
    └─────────────────┘                             └─────────────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ 决策：选 MNN                        │
                    │                                     │
                    │ sherpa-onnx 缺失的是 Qwen3-ASR 的   │
                    │ 核心计算引擎（LLM），这是不可弥补的   │
                    │ 结构性缺陷。MNN 缺失的 ASR 管线可以  │
                    │ 通过扩展已有基础设施（Omni 类、     │
                    │ audio.py、Gemma4Audio）来补充，     │
                    │ 这些都有现成的参考模板。             │
                    └─────────────────────────────────────┘
```

### 6.4 决策结论

**选择 MNN 推理框架**。核心理由：

1. Qwen3-ASR 的 decoder 是一个完整的 600M 参数 Qwen3 LLM，这是推理的性能瓶颈。sherpa-onnx **根本不具备 LLM 推理能力**，这是结构性缺陷。
2. MNN 已有的 Qwen3 优化（ARM 低比特 kernel、KV-cache、speculative decoding）比从头在 onnx runtime 上实现 LLM 推理节省数周甚至数月的工作量。
3. MNN 缺失的 Audio Encoder 导出和 ASR 管线组装，有 `Gemma4Audio`、`Omni::audioProcess()` 等成熟模板，属于增量开发。

---

## 7. Android 系统级 ASR Service 架构

> 目标：将 Qwen3-ASR 注册为一个独立的系统 Service，供多个 App 通过 IPC 调用。

### 7.1 整体架构

```
┌───────────┐  ┌───────────┐  ┌───────────┐
│  App A    │  │  App B    │  │  App C    │    ← 多个客户端 App
│ (录音机)   │  │ (语音助手) │  │ (输入法)   │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │               │               │
      │    AIDL IPC   │               │
      └───────────────┼───────────────┘
                      ▼
         ┌─────────────────────────┐
         │   ASR Service           │   ← 独立进程 (:asr)
         │   (android:process)     │
         │                         │
         │  ┌───────────────────┐  │
         │  │  AsrService.kt    │  │   ← 继承 Service，AIDL 接口
         │  │  (已有框架，替换   │  │
         │  │   底层引擎)        │  │
         │  └────────┬──────────┘  │
         │           │              │
         │  ┌────────▼──────────┐  │
         │  │  JNI Layer        │  │   ← C++ ↔ Kotlin 桥接
         │  │  (libasr_jni.so)  │  │
         │  └────────┬──────────┘  │
         │           │              │
         │  ┌────────▼──────────┐  │
         │  │  MNN ASR Engine   │  │   ← C++ 推理引擎
         │  │  ┌──────────────┐ │  │
         │  │  │ Audio Encoder│ │  │   audio_encoder.mnn
         │  │  │   (MNN)      │ │  │
         │  │  └──────┬───────┘ │  │
         │  │         │          │  │
         │  │  ┌──────▼───────┐ │  │
         │  │  │ Qwen3 Decoder│ │  │   llm.mnn (4-bit 量化)
         │  │  │   (MNN)      │ │  │
         │  │  └──────────────┘ │  │
         │  └───────────────────┘  │
         └─────────────────────────┘
```

### 7.2 关键设计点

**1. 独立进程 + 权限安全**

```xml
<!-- AndroidManifest.xml -->
<permission
    android:name="com.alibaba.mnn.permission.ASR_SERVICE"
    android:protectionLevel="signature" />

<service
    android:name=".asr.AsrService"
    android:process=":asr"
    android:permission="com.alibaba.mnn.permission.ASR_SERVICE"
    android:exported="true">
    <intent-filter>
        <action android:name="com.alibaba.mnn.action.ASR_SERVICE" />
    </intent-filter>
</service>
```

- 独立进程（`android:process=":asr"`）：模型 ~225MB（4-bit），避免挤占各 App 主进程
- `signature` 级权限：仅允许同签名的 App 绑定，防止恶意 App 滥用录音通道
- 单一 Service 实例节省资源

**2. 录音策略：App 端录音 → PCM IPC → Service 推理**

```
┌─────────────┐    PCM (16kHz/16bit)     ┌──────────────────┐
│  App (前台)  │ ──── AIDL/shared mem ───→│  ASR Service     │
│  AudioRecord │                          │  (独立进程 :asr)  │
│  AEC/NS     │  ←── 识别结果 callback ── │  MNN Qwen3-ASR   │
└─────────────┘                          └──────────────────┘
```

**为什么不让 Service 直接录音？**
- Android 10+ 对后台进程录音有严格限制
- App 在前台，天然持有 `RECORD_AUDIO` 权限，不受后台限制
- Service 只做推理，不碰麦克风，权限模型更清晰
- AEC/NS 等硬件音频效果在 App 端处理（已有 `AsrService` 中的 AEC 实现）

**3. IPC 接口（AIDL — session + pushAudio 模型）**

```java
// IAsrService.aidl
interface IAsrService {
    AsrSession createSession(String appId, IAsrCallback callback, AsrConfig config);
    void destroySession(String sessionId);
    List<String> getSupportedLanguages();
}

// IAsrSession.aidl
interface IAsrSession {
    void pushAudio(in byte[] pcmData, int length);  // 推送 16kHz 16bit PCM
    void endAudio();                                  // 音频结束，触发最终识别
    void cancel();                                    // 取消本次识别
    void setMuted(boolean muted);                     // 静音控制（打断场景）
}

// IAsrCallback.aidl
interface IAsrCallback {
    void onPartialResult(String text);     // 流式中间结果
    void onFinalResult(String text);       // 最终识别结果
    void onSpeechDetected();               // 检测到语音活动
    void onError(int code, String msg);
}
```

- **session 模式**：每个 App 创建独立 session，支持多 App 注册（但同一时刻只有一个 active）
- **pushAudio**：App 推送 PCM 数据，Service 返回识别结果，职责分离清晰
- PCM 传输量：16kHz × 16bit × mono = 32KB/s，AIDL binder 完全够用（无需共享内存）

**4. 推理线程模型**

```
┌────────────────────────────────────────────────────┐
│                   ASR Service                      │
│                                                    │
│  Binder Thread Pool  ──→  Inference Worker (单线程) │
│       (IPC 消息)              │                     │
│                               ├─ encoder forward    │
│                               ├─ decoder generate   │
│                               └─ callback 回调       │
│                                                    │
│  模型加载：懒加载（首次调用时 load，warm 常驻）        │
└────────────────────────────────────────────────────┘
```

- **单一推理 worker**：串行化所有推理请求，避免多线程竞争 KV-cache
- **懒加载**：Service 启动时不加载模型，首次 `createSession` 时加载，避免拖慢系统启动
- **warm 常驻**：加载后不卸载，保持推理 ready 状态

**5. 多客户端管理**

```kotlin
class AsrService : Service() {
    // 注册表：sessionId → AsrSession
    private val sessions = ConcurrentHashMap<String, AsrSession>()
    
    // 当前 active session（同时只有一个在使用 ASR）
    @Volatile private var activeSessionId: String? = null
    
    fun pushAudio(sessionId: String, pcmData: ByteArray, length: Int) {
        if (sessionId != activeSessionId) {
            // 非 active session 推送音频，通知冲突
            callback.onError(ERROR_BUSY, "Another app is currently active")
            return
        }
        engine.pushAudio(pcmData, length)
    }
}
```

- 同一时刻只有一个 active session 使用 ASR（避免并发推理导致发热/OOM）
- 多 session 注册存在但不活跃，切 App 时快速切换而非重新加载模型

**6. 与现有 MnnLlmChat 的关系**

保持共存，根据 model type 自动路由：

```
                    ┌──────────────────────┐
                    │   AsrService          │
                    │   ┌────────────────┐  │
                    │   │ sherpa-mnn     │  │  ← 保留，Zipformer 模型兼容
                    │   │ (Zipformer)    │  │
                    │   └────────────────┘  │
                    │   ┌────────────────┐  │
                    │   │ MNN Qwen3-ASR  │  │  ← 新增，主力引擎
                    │   └────────────────┘  │
                    └──────────────────────┘

### 7.3 内存与性能预算（Android 10 中端设备）

| 指标 | 4-bit 量化 | 8-bit 量化 | BF16 原始 |
|------|-----------|-----------|----------|
| Decoder (llm.mnn) | ~150 MB | ~300 MB | ~1.2 GB |
| Encoder (audio_encoder.mnn) | ~25 MB | ~45 MB | ~180 MB |
| 运行时（KV-cache + 临时） | ~50 MB | ~80 MB | ~200 MB |
| **Service 进程总计** | **~225 MB** | **~425 MB** | **~1.6 GB** |

- 推荐 4-bit 量化，可运行于 3-4GB RAM 设备
- mmap 加载模型文件，Service 启动速度 < 500ms
- 首字延迟目标：< 200ms（2-4 秒音频窗口）

---

## 8. 讨论与决策点

### 已决策项

| # | 决策点 | 结论 | 依据 |
|---|--------|------|------|
| 1 | 推理框架 | **MNN** | sherpa-onnx 无 LLM 推理能力（§6） |
| 2 | 跨 App 架构 | **独立进程 :asr + AIDL IPC** | 节省内存、单实例共享（§7） |
| 3 | 录音策略 | **App 端录音 → PCM IPC** | Android 10+ 后台限制（§7.2） |
| 4 | ASR 引擎共存 | **新增 MNN 引擎，保留 sherpa-mnn** | 向后兼容 Zipformer 模型 |
| 5 | 权限安全 | **signature 级权限** | 防止恶意 App 滥用（§7.2） |
| 6 | AIDL 接口 | **session + pushAudio 模型** | 职责分离、多 App 支持（§7.2） |
| 7 | 推理线程 | **单一 inference worker 串行** | 避免 KV-cache 竞争 |
| 8 | 模型加载 | **懒加载 + warm 常驻** | 不拖慢启动 |

### 待验证项（POC 阶段决定）

1. **CPU RTF 达标性** — 目标 RTF < 0.6，不达标则考虑 GPU 卸载 encoder 部分
2. **量化策略** — 8-bit 验证正确性，4-bit 用于生产
3. **流式窗口大小** — 实测后决定 2s/4s/8s

### 已源码确认项（无需 POC）

- **inputs_embeds 注入** ✅ — `Llm::forward(VARP input_embeds)`（`llm.cpp:608`），Talker 已使用（`omni.cpp:2047`）
- **Decoder 架构** ✅ — Qwen3 标准架构（RMSNorm+GQA+QK Norm+SwiGLU），差异仅权重前缀 `thinker.model.*` → 10 行 mapper 配置

---

## 附录：参考资料

- MNN LLM 文档：`transformers/README.md`
- llmexport 使用说明：`transformers/llm/export/llmexport.py`
- 模型映射配置：`transformers/llm/export/utils/model_mapper.py`
- 音频编码器示例：`transformers/llm/export/utils/audio.py`（Qwen2Audio, Qwen2_5OmniAudio）
- MnnLlmChat README：`apps/Android/MnnLlmChat/README_CN.md`
- ASR 模型配置：`apps/Android/MnnLlmChat/app/src/main/java/com/k2fsa/sherpa/mnn/AsrModelConfig.kt`
