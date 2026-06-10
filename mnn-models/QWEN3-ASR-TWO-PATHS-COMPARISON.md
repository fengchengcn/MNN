# Qwen3-ASR 两条推理路径对比分析

> 创建：2026-06-10
> 模型：Qwen3-ASR-0.6B（28 层 / 1024 hidden / GQA / 8-bit 量化）
> 设备：华为 Mate 30 (Kirin 990, 8GB) / Mate 40 (Kirin 9000, 8GB)

---

## 一、一句话总结

**旧路径是开发者当司机（每一步手写），OMNI 路径是自动驾驶出租车（告诉目的地，引擎自己跑）。**

本质区别只有一个：**谁拥有推理循环**。

---

## 二、架构对比

```
旧路径（qwen3_asr_engine.cpp）：开发者手写推理循环
┌─────────────────────────────────────────────────────────┐
│ Qwen3AsrEngine（用户代码 ~850 行）                        │
│                                                         │
│  runDecoder() {                                         │
│    whisper_fbank()        ← 手写 mel 特征提取            │
│    m_ae_mod->onForward()  ← 手写 Audio Encoder 调用     │
│    embedLookup()          ← 手写 mmap bf16→f32 查表     │
│    createCausalMask()     ← 手写 causal mask 构造       │
│    createEmptyCache()     ← 手写 KV cache 初始化        │
│    while (gen_len < max) {                              │
│      m_llm_mod->onForward()  ← 手写逐 token 自回归      │
│      argmaxPenalized()       ← 手写贪心采样             │
│    }                                                    │
│  }                                                      │
└─────────────────────────────────────────────────────────┘

OMNI 路径（LlmSession / Omni 引擎）：引擎内置推理循环
┌─────────────────────────────────────────────────────────┐
│ LlmSession::generate("<audio>stream</audio>")  ← 一行   │
│   └─ Omni::prefillForGraph()                            │
│       ├─ AUDIO::whisper_fbank()    ← 引擎自动           │
│       ├─ mAudioModule->onForward() ← 引擎自动           │
│       ├─ Llm::embedding()          ← 引擎自动           │
│       ├─ Llm::forward()            ← 引擎自动           │
│       │   ├─ KVCache（自动管理，FP16 + expandChunk）    │
│       │   └─ FusedAttention（融合 kernel，GPU 可用）    │
│       └─ Sampler::sample()         ← 引擎自动           │
│           (top-k / top-p / temperature / min_p / ...)   │
└─────────────────────────────────────────────────────────┘
```

---

## 三、模型文件对比

| 文件 | 旧路径（ONNX 分解导出） | OMNI 路径（llmexport.py 融合导出） |
|------|:--|:--|
| **导出工具** | `export_qwen3_asr.py` → MNNConvert | `llmexport.py --export mnn` |
| **LLM Decoder** | `llm_kv_8bit.mnn` + `.weight` | `llm.mnn` + `.weight` |
| **Audio Encoder** | `audio_encoder.mnn` | `audio.mnn` + `.weight` |
| **Embeddings** | `embeddings_bf16.bin`（独立 mmap 文件） | 嵌入在 `llm.mnn.weight` 中（tie_embeddings） |
| **Tokenizer** | `tokenizer.txt` | `tokenizer.mtok` + `tokenizer.txt` |
| **模型配置** | 无（常量硬编码在 `.h` 中） | `llm_config.json`（结构信息） |
| **运行时配置** | 无（参数硬编码） | `config.json`（后端/采样/线程等全配置） |
| **Attention 格式** | 分解算子（MatMul + Add + Softmax 逐 op）| `OpType_Attention` 融合 Attention |
| **量化方式** | MNNConvert 8-bit | MNN 内置量化（8-bit / BF16 / 可配置） |
| **System Prompt** | 硬编码在 `buildPromptTokens()` | `config.json` → `system_prompt` 字段 |

### 模型目录示例

**旧路径**（`/data/local/tmp/mnn_models/Qwen3-ASR-0.6B/`）：
```
audio_encoder.mnn          (~190 MB, 独立文件)
llm_kv_8bit.mnn            (~500 KB, 图结构)
llm_kv_8bit.mnn.weight     (~575 MB, 权重)
embeddings_bf16.bin        (~296 MB, bf16 原始数据)
tokenizer.txt              (~3 MB)
```

**OMNI 路径**（`/data/local/tmp/mnn_models/Qwen3-ASR-MNN-INT8/`）：
```
llm.mnn                    (~506 KB, 图结构)
llm.mnn.weight             (~634 MB, 权重 + embeddings)
llm_config.json            (~455 B, 模型结构)
audio.mnn                  (~359 KB, 图结构)
audio.mnn.weight           (~220 MB, Audio Encoder 权重)
config.json                (~1 KB, 全配置)
tokenizer.mtok             (~3 MB)
tokenizer.txt              (~3 MB)
```

---

## 四、推理引擎对比

### 4.1 核心差异

| 维度 | 旧路径 | OMNI 路径 |
|------|:--|:--|
| **推理循环归属** | 用户代码（`qwen3_asr_engine.cpp`） | MNN 引擎（`omni.cpp` + `llm.cpp`） |
| **代码量** | ~850 行 C++ + ~300 行 Kotlin/JNI | ~0 行（引擎内置），仅 Kotlin 调用层 |
| **采样策略** | 仅 argmax + repetition penalty | top-k / top-p / temperature / min_p / tfs_z / typical / n-gram |
| **KV Cache** | 手写 5D VARP 管理 | 引擎内置 `KVCache` 类（FP16 + expandChunk=64） |
| **Embedding** | 手写 mmap + bf16→f32 查表 | 引擎内置 `Embedding` 层 |
| **Prompt 构建** | 手写 token 序列拼接 + 音频嵌入注入 | 引擎自动：tokenize → embedding → AUDIO_PAD 处注入 |

### 4.2 后端支持

| 后端 | 旧路径 | OMNI 路径 |
|------|:--|:--|
| **CPU (ARM NEON)** | ✅ FP16 + Power_High + 4 线程（~20 tok/s） | ✅ 同配置（预估 ~22-24 tok/s） |
| **Vulkan GPU** | ❌ 分解 Attention 阻塞融合 kernel | ✅ `VulkanAttention` 融合 kernel（预估 ~40-60 tok/s） |
| **OpenCL GPU** | ❌ 同上 | ✅ `AttentionBufExecution`（预估 ~30-50 tok/s） |
| **HiAI NPU** | ❌ 缺失 RMSNorm / RoPE / SiLU | ❌ 同上 |

> **说明**：当前 OMNI 路径模型导出时 `transformer_fuse: false`，FusedAttention 未开启，GPU 加速尚未解锁。重新导出时加 `--transformer_fuse` 即可。

### 4.3 功能模式

| 模式 | 旧路径 | OMNI 路径 |
|------|:--|:--|
| **BATCH（整段→结果）** | ✅ `runDecoder()` 同步阻塞 | ❌ 无等效模式 |
| **STREAMING（逐 token 流式）** | ✅ `startDecode()` / `decodeStep()` | ❌ 当前无逐 token 流式 |
| **VAD 分段（边说边出）** | ✅ Kotlin 层 VAD + 逐段 BATCH/STREAMING | ✅ Kotlin 层 VAD + `generate()` 逐段 |

---

## 五、性能对比

| 指标 | 旧路径 | OMNI 路径（CPU） | OMNI 路径（Vulkan GPU，预期） |
|------|:--|:--|:--|
| **吞吐量** | ~20 tok/s | ~22-24 tok/s | ~40-60 tok/s |
| **首 token 延迟** | ~3-5s（AE + prefill） | ~1-2s | ~0.5-1s |
| **AE 加载** | ~400ms（首次后复用） | 引擎自动管理 | 引擎自动管理 |
| **Decoder 加载** | ~1.8s（首次后复用） | 引擎自动管理 | 引擎自动管理 |
| **内存峰值** | ~1.3 GB（AE + Decoder 同时驻留） | ~1.3 GB | ~1.5 GB（GPU 额外 buffer） |

---

## 六、适用场景

| 场景 | 推荐路径 | 理由 |
|------|:--|------|
| **当前 Android 生产** | 旧路径 | Phase 1–3 实机验证通过，稳定 |
| **快速迭代调试** | 旧路径 | 手写循环完全可控，无黑盒 |
| **需要逐 token 流式** | 旧路径 | OMNI 路径暂无此能力 |
| **未来 GPU 加速生产** | OMNI 路径 | Vulkan FusedAttention → 2-3x 吞吐 |
| **多模态扩展** | OMNI 路径 | `omni.cpp` 原生支持视觉+音频 |
| **长期维护** | OMNI 路径 | 0 行手写推理代码，引擎统一演进 |

---

## 七、关键设计取舍

| 决策 | 旧路径 | OMNI 路径 |
|------|:--|:--|
| **灵活性** | 高（每一步可干预/调试） | 低（引擎黑盒，行为由 config 控制） |
| **维护成本** | 高（手写循环需跟随模型变化） | 低（引擎统一维护） |
| **采样质量** | 低（仅贪心） | 高（完整采样策略） |
| **GPU 加速** | 不可用 | 可用（需 fusion 导出） |
| **错误处理** | 手写（可控但繁琐） | 引擎内置（统一但难定制） |
| **模型格式** | ONNX 分解（MNNConvert） | llmexport.py 融合（MNN 原生） |

---

## 八、共存策略

当前 `Qwen3AsrTestActivity` 支持两条路径共存：

```
initEngine()
  ├─ findOldPathModel()    → 检测 audio_encoder.mnn
  │   └─ Qwen3AsrEngine    → BATCH + STREAMING 模式
  └─ findOmniModel()       → 检测 audio.mnn + config.json (is_audio=true)
      └─ LlmSession        → OMNI 模式

UI 状态矩阵：
  旧路径 ✅ + OMNI ✅  →  三模式全开（BATCH / STREAMING / OMNI）
  旧路径 ✅ + OMNI ❌  →  双模式（BATCH / STREAMING）
  旧路径 ❌ + OMNI ✅  →  单模式（OMNI）
  旧路径 ❌ + OMNI ❌  →  错误提示
```

回滚策略：
- 保留 `Qwen3AsrEngine` 全部代码不动
- 保留 `LlmSession` / Omni 路径独立运行
- 通过 `config.json` 或模式选择切换，无耦合

---

## 九、相关文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| 旧路径引擎 | `apps/Android/MnnLlmChat/app/src/main/cpp/qwen3_asr_engine.h` | C++ 头文件 |
| 旧路径引擎 | `apps/Android/MnnLlmChat/app/src/main/cpp/qwen3_asr_engine.cpp` | C++ 实现（~850 行） |
| 旧路径 JNI | `apps/Android/MnnLlmChat/app/src/main/cpp/qwen3_asr_jni.cpp` | JNI 桥接 |
| 旧路径 Kotlin | `apps/Android/MnnLlmChat/app/src/main/java/.../asr/Qwen3AsrEngine.kt` | Kotlin 封装 |
| 测试页面 | `apps/Android/MnnLlmChat/app/src/main/java/.../asr/Qwen3AsrTestActivity.kt` | 两路径统一测试页 |
| OMNI C++ 引擎 | `transformers/llm/engine/src/omni.cpp` | Omni 多模态推理 |
| OMNI LLM 引擎 | `transformers/llm/engine/src/llm.cpp` | LLM 推理 + 采样 |
| OMNI 配置 | `transformers/llm/engine/src/llmconfig.hpp` | 配置解析（`is_audio()` 等） |
| llmexport.py | `transformers/llm/export/llmexport.py` | OMNI 路径模型导出入口 |
| model_mapper | `transformers/llm/export/utils/model_mapper.py` | 模型字段映射注册 |
| OMNI 模型 | `mnn-models/Qwen3-ASR-MNN-INT8/` | 已导出的 OMNI 路径模型 |
| 流式优化计划 | `mnn-models/Qwen3-ASR-STREAMING-PLAN.md` | Phase 1–3 优化 + llmexport.py 迁移计划 |
