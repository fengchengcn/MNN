# Qwen3-ASR-0.6B MNN 迁移：初步调研与方案设计

> 目标：将 Qwen3-ASR-0.6B 模型转换为 MNN 格式，集成到 MnnLlmChat Android 应用中，实现端侧离线流式语音识别。
>
> **更新**（第二轮调研）：经深入代码审查，发现 MNN 框架已具备大部分所需基础设施，工作量预估从 13-19 天下调至 **8-12 天**。详见 §3.2 和 §4。

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

### 3.2 实现路径（推荐路径 A）

**扩展已有 MNN Omni 框架**。好消息是 MNN 已经具备大部分所需能力，核心思路是**参考 Omni 音频处理管线，新增 Qwen3-ASR 专用推理类**。

#### 阶段一：拉取上游更新 + 环境准备（~0.5 天）

- 当前 repo 落后 upstream 215 commits，需先 `git merge upstream/master`
- 在 Ubuntu 服务器上通过 ModelScope 下载 Qwen3-ASR-0.6B
- 安装 Python 依赖（torch, transformers, onnx, modelscope）
- 编译 MNNConvert 工具

#### 阶段二：模型导出（~2 天）

**Audio Encoder 导出**（~1 天）：
- 在 `audio.py` 中新增 `Qwen3ASRAudio` 类，参照 `Qwen2Audio` / `Qwen2_5OmniAudio`：
  - 加载 `thinker.audio_tower`（Whisper-style 18 层 encoder + 投影层）
  - 实现 forward：Conv2d 下采样 → Transformer Encoder → LayerNorm → 投影层
  - 导出为 `audio_encoder.onnx` → `audio_encoder.mnn`

**LLM Decoder 导出**（~0.5 天）：
- 在 `model_mapper.py` 中新增 `regist_qwen3_asr`，映射 `thinker.model` 的权重路径
- 验证 Qwen3 decoder 结构与已有 Qwen3 支持一致
- 使用现有 `llmexport.py` 导出 `llm.mnn`（Qwen3 decoder 架构已验证兼容）

**Tokenizer 导出**（~0.5 天）：
- Qwen3-ASR 使用标准 Qwen3 tokenizer（151,936 词表）

#### 阶段三：推理引擎实现（~3 天）

**新增 `Asr` 类**，继承或参考 `Omni` 类：
- 加载 `audio_encoder.mnn`（复用 Omni 的 `mAudioModule` 机制）
- 音频前处理：直接调用 `MNN::AUDIO::whisper_fbank()` ✅ 已有
- Encoder 推理：运行 audio encoder → 得到 prefix embeddings
- **Embedding 注入**：将 prefix embeddings 放在 token embeddings 前面（与 Omni 的 embedding 替换逻辑不同但理念相同）
- LLM Decode：调用已有 `Llm` 基类的自回归解码
- 流式支持：音频分块 → encoder 增量推理 → decoder 增量生成

**关键简化**：Qwen3-ASR 的 decoder 使用标准 causal self-attention（无跨注意力），embedding 注入后完全是标准 LLM decode，因此可以大量复用 `Llm` 基类的代码。

#### 阶段四：MNN 库编译（~0.5 天）

- 已有 CMake 参数：`-DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true`
- 交叉编译 Android arm64-v8a（已有脚本和 CI 流程）

#### 阶段五：Android 集成（~2-3 天）

- 实现 JNI 层：封装 Asr 类接口（init/process/recognize/reset）
- MnnLlmChat 接入：
  - 新增 ASR 引擎类型（与 sherpa-mnn 共存）
  - 模型下载配置（ModelMarket）
  - 流式识别回调

#### 阶段六：测试与优化（~2-3 天）

- 端到端流式识别测试
- 量化验证（建议 8-bit 先验证正确性）
- 性能调优（线程数、内存、首字延迟）
- **已有 OpenCL GPU 加速可用于 encoder**

### 3.3 关键技术风险（修正版）

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| Qwen3-ASR audio encoder 的结构与已有 Qwen2Audio encoder 差异 | 中 | Style 相似（都是 Whisper-based），主要差异在于层数（18 vs 原版）、投影层结构。参考 `Qwen2_5OmniAudio` 的实现模式 |
| Decoder 是否完全兼容 Qwen3 导出 | 低 | 已验证：RMSNorm、GQA、QK Norm、SwiGLU、NeoX RoPE 均在 MNN 中支持；`thinker.model.layers` 结构与 Qwen3 一致 |
| 流式识别的音频分块策略 | 中 | Qwen3-ASR 支持 1~8s 动态注意力窗口；移动端建议 2-4s，每次分块后 encoder 增量推理 |
| Decoder 的 embedding 注入机制 | 低 | Omni 已有 `mAudioEmbeddings` 向量 + embedding 替换逻辑，改为 prefix 注入即可 |
| Android 端内存（~1GB+ 原始模型） | 中 | 4-bit 量化可将 decoder 压缩至 ~250MB；encoder ~50MB（BF16）→ ~12MB（4-bit）；启用 mmap 加载 |
| MRoPE 位置编码兼容性 | 低 | MNN 已支持 Qwen3 MRoPE（mrope_section=[24,20,20]），纯音频场景退化为标准 RoPE |

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

## 4. 工作量估算（修正版）

**本轮调研的关键发现**：MNN Omni 类已有完整的音频多模态推理管线（whisper_fbank → audio encoder → embedding 注入 → LLM decode），Qwen3-ASR 的 decoder 是标准 Qwen3 结构（已完整支持）。主要新工作集中在 audio encoder 导出和流式 ASR 管线适配。

| 阶段 | 预估工作量 | 难度 | 已有基础 |
|------|-----------|------|---------|
| 一：环境准备 | 0.5 天 | ⭐ | - |
| 二：模型导出 | 2 天 | ⭐⭐ | Qwen3 decoder 导出已验证，Qwen2Audio encoder 导出可参考 |
| 三：推理引擎 | 3 天 | ⭐⭐⭐ | Omni 音频管线可直接复用，Llm 基类提供完整 decode |
| 四：MNN 编译 | 0.5 天 | ⭐ | 已有 CMake 脚本和 CI |
| 五：Android 集成 | 2-3 天 | ⭐⭐⭐ | 已有 sherpa-mnn 集成可参考，ASR 服务接口已定义 |
| 六：测试优化 | 2-3 天 | ⭐⭐ | 已有 benchmark 工具 |
| **总计** | **8-12 天** | | |

相比第一版估算（13-19 天），修正后的主要原因是：
- ❌ **不需要**从头写 encoder-decoder 推理引擎（Omni 已有）
- ❌ **不需要**实现 Whisper fbank 特征提取（`MNN::AUDIO::whisper_fbank` 已有）
- ❌ **不需要**处理 LLM decoder 逻辑（Qwen3 已支持，无跨注意力）
- ✅ **只需要**：新增 audio encoder 导出 + 实现 embedding 注入 + 流式分帧策略

---

## 5. 上游仓库对比

当前 repo 落后 upstream（alibaba/MNN）**215 个 commit**，版本从本地落后到 upstream `v3.5.0`。

### 主要新增内容

| 类别 | 代表性 commit |
|------|-------------|
| **LLM 特性** | dflash speculative decoding、LinearAttention prefix cache、hybrid model support |
| **新模型** | Gemma4、Hunyuan、Qwen3-VL DeepStack |
| **ARM 性能** | 低 bit 量化 kernel（2-bit/3-bit）、KleidiAI v1.16.0、SVE2 向量化 |
| **GPU** | OpenCL int4 GEMM 优化、Vulkan dynamic quant (W8A8) |
| **Bug 修复** | Qwen3-VL 重复输出、Gemma3 dual-RoPE 导出、KV cache mismatching |

**关于 Qwen3-ASR**：upstream 和本地均**没有**直接的 Qwen3-ASR 支持。但 upstream 的 Omni 音频管线优化（interleaved Thinker/Talker generation）和 LinearAttention 支持对 ASR 场景有利。**建议先 merge upstream 再开始开发。**

---

## 6. 讨论与决策点

以下问题需要在进入实现前确认：

1. **量化策略**：是否直接使用 4-bit 量化以降低内存？（建议：先用 8-bit 验证正确性，再切 4-bit）
2. **流式窗口大小**：Qwen3-ASR 支持 1~8 秒的动态注意力窗口。移动端建议 2-4 秒，需要根据实测延迟和精度 trade-off 决定
3. **Decoder 独立导出验证**：是否先单独验证 Qwen3-ASR 的 LLM decoder 能用现有 llmexport 正确导出和推理？（建议：先做，降低风险）
4. **Android 应用的 ASR 引擎切换**：是替换现有 sherpa-mnn 还是新增一种 ASR 引擎类型？（建议：新增，保留 sherpa-mnn 兼容性）

---

## 附录：参考资料

- MNN LLM 文档：`transformers/README.md`
- llmexport 使用说明：`transformers/llm/export/llmexport.py`
- 模型映射配置：`transformers/llm/export/utils/model_mapper.py`
- 音频编码器示例：`transformers/llm/export/utils/audio.py`（Qwen2Audio, Qwen2_5OmniAudio）
- MnnLlmChat README：`apps/Android/MnnLlmChat/README_CN.md`
- ASR 模型配置：`apps/Android/MnnLlmChat/app/src/main/java/com/k2fsa/sherpa/mnn/AsrModelConfig.kt`
