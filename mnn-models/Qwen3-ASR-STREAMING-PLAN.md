# Qwen3-ASR 生产级实时流式优化计划

> 创建：2026-06-08 | 最后更新：2026-06-09
> 基于：qwen3_asr_engine.cpp v7（Android 端到端推理成功，中文 OK，系统提示词已更新）
> 参考：MNN 框架源码深度分析（HiAI/OpenCL/Vulkan 后端 + audio 管线 + Executor 生命周期）
>
> **状态：Phase 1 ✅ | Phase 2 ✅ 实机验证通过 | Phase 3 ✅ 实机验证通过**
>
> **新增：第七章 — 向 llmexport.py 迁移的详细计划（2026-06-09）**

---

## 一、现状与瓶颈

### 1.1 当前推理管线（单次 utterance）

```
Kotlin: endAudio()  [阻塞等待 ~5-8s]
  └─ JNI: nativeEndAudio()
       └─ C++: runDecoder()  [同步执行全部步骤]
            ├─ Executor::newExecutor(MNN_FORWARD_CPU)        ~10ms
            ├─ 写临时 WAV 文件 (磁盘 I/O)                     ~5-50ms
            ├─ MNN::AUDIO::load() 读回 WAV                   ~5-20ms
            ├─ whisper_fbank() STFT+mel+log                   ~20-50ms
            ├─ loadAudioEncoder() 从磁盘加载 190MB 模型       ~2-4s
            ├─ warmup × 2 (shape infer + mem alloc)           ~200-600ms
            ├─ AE 推理 (real)                                 ~100-300ms
            ├─ ae_mod.reset() 释放 AE (~500MB)               ~0ms
            ├─ ensureDecoderLoaded() (首次 ~3-5s, 后续 0)    ~0-5s
            ├─ Prefill (53 tokens × 28 layers)                ~200ms
            ├─ Decode loop (每步 ~20-30ms × N tokens)         ~1-3s
            └─ return result
```

**每个 utterance 的可优化浪费：**
- WAV 文件往返：5-50ms 磁盘 I/O + int16 量化噪声
- AE 加载：2-4s（每次从磁盘读 190MB 模型文件）
- AE warmup × 2：200-600ms（shape 推断 + 内存分配）
- Executor 创建/销毁：~10ms

### 1.2 当前耗时预估（Kirin 9000 ARM, 3s 音频）

| 阶段 | 首次 | 后续 utterance |
|------|------|:---:|
| Executor 创建 | ~10ms | ~10ms |
| WAV 往返 | ~10ms | ~10ms |
| fbank | ~20ms | ~20ms |
| AE 加载 | ~2-4s | ~2-4s |
| AE warmup × 2 | ~200ms | ~200ms |
| AE 推理 | ~100ms | ~100ms |
| Decoder 加载 | ~3-5s | 0 (复用) |
| Prefill | ~200ms | ~200ms |
| Decode (~40 tokens) | ~800ms | ~800ms |
| **总计** | **~7-13s** | **~3.3-5.1s** |

优化目标：后续 utterance **< 1.5s**（首次仍可接受 ~5s）

---

## 二、优化方案

### 2.1 Phase 1：消除不必要开销（优先级最高）✅ 已完成

> **完成日期**：2026-06-08 | **实机验证**：华为 Mate 30 (Kirin 990, 8GB RAM)
> 
> **关键实测数据**：首次 utterance ~4.1s，后续 utterance ~1.3s（3.2x 加速），无 OOM，无崩溃（加 mutex 后）。

#### 1a. Audio Encoder 常驻（预计省 2-4s）

**当前问题**：每次 `runDecoder()` 调用 `loadAudioEncoder()` → 从磁盘加载 `audio_encoder.mnn` (190MB)，用完立即 `ae_mod.reset()`。

**优化方案**：
- 头文件新增 `std::shared_ptr<Module> m_ae_mod` 成员
- 新增 `ensureAudioEncoderLoaded()` 方法（首次加载 + warmup，后续直接复用）
- `runDecoder()` 使用 `m_ae_mod` 而非局部变量，不再 reset
- `release()` 中释放 `m_ae_mod`

**关键设计决策**：AE warmup（2 次 `onForward`）从 `runDecoder()` 移至 `ensureAudioEncoderLoaded()`。首次加载时执行 warmup，后续 utterance 直接跳过。

**内存代价**：+450-500MB（与 Decoder 同时驻留）
- `audio_encoder.mnn`：190MB 磁盘 → 运行时 ~500MB（含中间 buffer）
- 当前 Decoder：~800MB
- 峰值：~1.3GB（vs 当前 ~800MB 单模型）

**OOM 风险评估**：
- Mate 30 (8GB RAM)：App 可用 ~2-3GB → 安全
- Mate 40 (8GB RAM)：安全
- 4GB RAM 低端设备：需加运行时检查，低于阈值时回退到当前串行模式

**实现复杂度**：极低 — 与 `ensureDecoderLoaded()` 完全相同的模式

#### 1b. 移除 WAV 文件往返（预计省 5-50ms + 精度提升）

**当前问题**（`qwen3_asr_engine.cpp:370-429`）：
```
m_audio_buffer (float[]) 
  → int16 量化 → 写临时 WAV → unlink → MNN::AUDIO::load() → 读回 int16 → float
```

这是不必要的。`whisper_fbank()` 接受 `VARP waveform`，不需要文件。

**优化方案**：
```cpp
// 替换 ~140 行 WAV 代码:
auto wf = _Input({nsamples}, NHWC, halide_type_of<float>());
memcpy(wf->writeMap<float>(), m_audio_buffer.data(), nsamples * sizeof(float));
auto feat = MNN::AUDIO::whisper_fbank(wf);
```

**额外收益**：
- 消除 int16 → float → int16 → float 量化噪声
- 消除临时文件安全风险（SELinux 问题已在 v5 中遇到）
- 删除 ~140 行 WAV 代码

**MNN API 确认**：`MNN::AUDIO::load()` 无内存加载接口，但 `whisper_fbank()` 接受 `VARP`，不依赖文件路径。

**实现复杂度**：极低

#### 1c. AE Warmup 移至 Init（预计省 200-600ms/utterance）

**当前问题**：`runDecoder()` 中对 AE 做 2 次 warmup `onForward`。首次调用需要 shape 推断和内存分配，但后续调用同一 shape 不需要。

**优化方案**：`ensureAudioEncoderLoaded()` 首次加载时执行 2 次 warmup（用固定长度的 synthetic input），后续直接使用。由于输入 shape 固定为 `[1, 128, T]`（T 随音频长度变化），warmup 需要用典型 T 值执行一次。如果 T 变化触发了 re-compile，MNN 内部会自动处理，代价远小于每次 3 次 forward。

**重要说明**：如果不同 utterance 的 T（音频帧数）差异很大（如 1s vs 30s），MNN 在 shape 变化时会自动触发内部重编译（~50-100ms），但这仍然比完全重载模型快一个数量级。

**实现复杂度**：低

### 2.2 Phase 2：增量解码（生产级流式）

#### 目标

将单次阻塞调用 `runDecoder()` 拆分为：
```
startDecode() → prefill → 立即返回 first token
  ↓
while (isDecoding())
  decodeStep() → 单步自回归 → 返回新 token
  getPartialResult() → 实时文本
  ↓
reset()
```

#### 状态管理

需要在 C++ 对象中新增以下解码状态成员：

```cpp
// 增量解码状态（仅在解码期间有效）
bool m_decoding_active = false;
int m_decode_gen_len = 0;              // 已生成 token 数
int m_decode_S = 0;                    // 当前序列长度
int m_decode_current_token = 0;        // 最新 token
int m_decode_T = 0;                    // audio frames 数量
std::shared_ptr<Executor> m_decode_executor;  // 解码会话 executor
```

#### 关键设计决策

**1. Executor 生命周期**：每次解码创建一个 Executor，全程保持存活（startDecode → 所有 decodeStep → 结束）。不跨 utterance 复用 decoder executor，因为 MNN 的 exec 内存池可能碎片化。

**2. VARP 有效性保证**：只要 Executor 存活，其 `onForward()` 返回的 VARP（包括 KV cache）就保持有效。每个 `decodeStep()` 前用 `ExecutorScope scope(m_decode_executor)` 恢复上下文。

**3. 线程约束**：所有 decode 步骤必须在同一线程执行（JNI 线程亲和性）。当前 recording thread 天然满足。如果上层 Kotlin 想从不同线程调用 `decodeStep()`，需要加同步。

**4. Fallback 策略**：保留当前 `runDecoder()` 作为 fallback，通过 `bool` 配置开关切换新旧模式，确保可回滚。

#### 新增 API

**C++ 层**（`qwen3_asr_engine.h`）：
```cpp
bool startDecode();                    // AE + prefill, 返回是否成功
bool decodeStep(int* token_out);       // 单步解码, 返回是否继续
bool isDecoding() const;               // 解码进行中?
std::string getPartialResult() const;  // 当前部分结果
```

**JNI 层**（`qwen3_asr_jni.cpp`）：
```cpp
Java_..._nativeStartDecode(JNIEnv*, jobject) -> jboolean
Java_..._nativeDecodeStep(JNIEnv*, jobject) -> jboolean
Java_..._nativeIsDecoding(JNIEnv*, jobject) -> jboolean
Java_..._nativeGetPartialResult(JNIEnv*, jobject) -> jstring
```

**Kotlin 层**（`Qwen3AsrEngine.kt`）：
```kotlin
fun startDecode(): Boolean
fun decodeStep(): Boolean        // true = more tokens coming
fun isDecoding(): Boolean
fun getPartialResult(): String   // non-blocking
```

**Kotlin Flow**（`VoiceChatPresenter.kt` 新流程）：
```kotlin
// 替代当前阻塞 endAudio()
engine.startDecode()
while (engine.isDecoding()) {
    if (engine.decodeStep()) {
        val text = engine.getPartialResult()
        // post partial text to UI (main thread)
    }
}
val finalText = engine.getResultText()
engine.reset()
```

#### 内存管理

Prefill 后立即可释放的资源：
- `merged` embeddings tensor → 释放 ~10-20MB
- `tokens` vector → 释放 ~1KB
- `audio_emb` → 已写入 merged，可释放 ~1-5MB

decode loop 期间必需保持：
- `m_k_cache` / `m_v_cache`（~140MB for 30s audio）
- `m_decode_executor`
- Model weights (~1.3GB with AE resident)

#### 预期延迟改进

| 场景 | 当前 | 优化后 |
|------|------|--------|
| 首 token 延迟 | ~5-8s | ~1-2s (prefill 完即返回) |
| 后续每 token | 用户不可见（批量返回） | ~20-30ms 逐字流式 |
| 用户体感 | "等很久然后一下子出来" | "边说边出，实时流式" |

### 2.3 Phase 3：CPU 路径微调（低风险快速收益）

#### 线程数调整

当前 `num_threads=2`，在 Kirin 9000 (4×A77 + 4×A55) 上偏保守。建议：
- 默认 `num_threads=4`（绑定大核）
- 通过 `BackendConfig::Power_High` 提升 CPU 频率

#### FP16 精度尝试

MNN CPU 后端通过 ARM v8.2 NEON FP16 支持 `Precision_Low`：
```cpp
bc.precision = MNN::BackendConfig::Precision_Low;  // 尝试 FP16
```
- Kirin 990/9000 的 Cortex-A76/A77 均支持 ARM v8.2 FP16
- 理论吞吐翻倍
- 需验证 ASR 精度是否可接受（先跑一次对比 token 序列）

#### Executor 复用

当前每个 utterance 创建新 executor（`runDecoder:359`）。改为 init 时创建并复用：
```cpp
// init() 中:
m_executor = Executor::newExecutor(MNN_FORWARD_CPU, bc, m_num_threads);
```
收益：每 utterance 省 ~10ms。风险低，但需确认 AE 和 Decoder 两个 Module 可共享 executor。

---

## 三、NPU/GPU 加速可行性结论（2026-06-09 更正）

### 分析过程

检查了 MNN 以下源码（实测验证，非推测）：
- `source/backend/hiai/execution/` — HiAI NPU 后端（96 个文件，含 53 个 op 实现）
- `source/backend/opencl/execution/buffer/` — OpenCL GPU buffer 后端
- `source/backend/vulkan/buffer/execution/` — Vulkan GPU 后端
- `include/MNN/MNNForwardType.h` — 后端类型定义
- `include/MNN/Interpreter.hpp` — HintMode / setHint 机制

### HiAI NPU：完全不可行 ✅（原结论成立）

| NPU 缺失的必需 Op | 在 Qwen3-ASR 中的用途 | 验证 |
|---|---|---|
| **RMSNorm** | Qwen3 全模型使用（非标准 LayerNorm），每层 2 次 | `NPULayerNorm` 仅实现标准 LayerNorm，`useRMSNorm()` 未使用 |
| **RoPE** (Rotary Position Embedding) | 每层 Attention 的核心位置编码 | 代码库中无任何 RoPE 相关实现 |
| **SiLU** (Swish) | FFN 激活函数 | `NPUActivation` 仅支持 ReLU/ReLU6/Tanh/Sigmoid |
| **5D Tensor** | KV Cache `[28, 1, 8, seq_len, 128]` | NPU 不支持 5D |
| **Dynamic Shape** | Decode 每步 seq_len 变化 | 需静态编译 |
| **Causal Mask** | 每一步不同的下三角 mask | 无动态 mask 支持 |

MNN HiAI 后端（2019 年开发）专门针对 CNN 模型（Conv2D/Pooling/ReLU），不支持现代 Transformer/LLM 架构。

### OpenCL GPU：需要更正（原结论不准确）

**原结论说「OpenCL 不可行」是基于推测，但经源码验证，当前状况比原分析乐观，但仍有重要限制。**

#### OpenCL 后端实际支持的 Transformer Op

| Op | MNN 实现 | 源码位置 | 状态 |
|----|---------|---------|:--:|
| `OpType_Attention` (FusedAttention) | `AttentionBufExecution` | `opencl/execution/buffer/AttentionBufExecution.cpp:1854` | ✅ 已注册为 TRANSFORMER |
| `OpType_FmhaV2` (Self-Attention) | `SelfAttentionBufExecution` | `opencl/execution/buffer/SelfAttentionBufExecution.cpp:583` | ✅ 已注册为 TRANSFORMER |
| `OpType_LinearAttention` | `LinearAttentionBufExecution` | `opencl/execution/buffer/LinearAttentionBufExecution.cpp` | ✅ 含 SiLU 支持 |
| `OpType_LayerNorm` (含 RMSNorm) | `LayerNormBufExecution` | `opencl/execution/buffer/LayerNormBufExecution.cpp:26` | ✅ RMSNorm 判断写在第 165 行 |
| `SiLU` activation | GroupNorm 中的 bSwish | `opencl/execution/buffer/GroupNormBufExecution.cpp:21` | ✅ |
| KV Cache 管理 | `KVCacheCLManager` | `opencl/execution/buffer/AttentionBufExecution.hpp:20-64` | ✅ 完整 prefill/decode 路径 |

#### 真正的瓶颈：当前模型格式，而非 MNN 能力

**根本问题不在 MNN 后端，而在当前 ONNX 导出的模型格式使用了分解算子（MatMul + Add + Softmax 逐 op），而非融合的 `OpType_Attention`。**

```
当前路径（ONNX 分解）：
  模型中的 Attention → 导出为:
    MatMul(Q, K^T) → Scale → Add(causal_mask) → Softmax → MatMul(V)
    → 每个 op 在 CPU/GPU 间独立调度 → 大量同步开销

推荐路径（llmexport.py 融合）：
  模型中的 Attention → FusedAttentionOp → OpType_Attention
    → 单一 MNN kernel，OpenCL/Vulkan 加速
```

使用当前 ONNX 分解模型跑 OpenCL 的后果：
- 每个 MatMul 结果从 GPU 回读 CPU
- 再传到下一个 op 的 GPU kernel
- 28 层 × 4 次 MatMul/层 = **112 次 CPU↔GPU 数据传输**
- 数据传输时间远超计算节省

#### Mali GPU 驱动风险（真实存在）

在 Mate 40 (Kirin 9000, Mali-G78 MP24) 上：
1. **OpenCL 驱动质量**：华为对 Mali OpenCL 驱动做了改进，相比其他 Mali 设备更稳定，但 MNN 团队主要测试环境是 Adreno
2. **Vulkan 驱动质量**：Vulkan 在 Mali-G78 上的支持明显好于 OpenCL（MNN LLM 引擎主推 Vulkan）
3. **FP16 支持**：Mali-G78 原生支持 FP16，Kirin 9000 可以受益

### Vulkan GPU：比 OpenCL 更可行的 GPU 路径

MNN 的 Vulkan 后端有完整的 Transformer 支持：

| 能力 | 支持情况 | 条件 |
|------|:--|------|
| `VulkanAttention` (FusedAttention) | ✅ 完整 prefill/decode | 需 `MNN_SUPPORT_TRANSFORMER_FUSE` |
| KV Cache 管理 (`KVCache`) | ✅ 支持 FP16 + expandChunk=64 | `VulkanAttention.hpp:30-40` |
| RMSNorm | ✅ (通过 LayerNorm with RMSNorm flag) | — |
| SiLU | ✅ (unary ops 路径) | — |

VulkanAttention 的优势：
- **FP16 支持**：`mUseFP16` 标志 + KVCache FP16 存储
- **Decode sub-group 优化**：小 batch decode 场景下的专用 kernel
- **MNN LLM 引擎主力后端**：`llm.cpp` 使用 Vulkan 作为 Android GPU 路径

### 更新后的结论

| 后端 | 可行性 | 需要的条件 | 预期提升 |
|------|:--|------|------|
| **HiAI NPU** | ❌ 不可行 | MNN HiAI 后端架构需大改 | — |
| **CPU ARM NEON + FP16** | ✅ 当前路径 | 已实现（Phase 3） | —（基线：~20 tok/s） |
| **OpenCL GPU** | ⚠️ 技术上可行 | **必须先迁移到 llmexport.py 融合 Attn** | Prefill 2-3x, Decode 1.5-2x |
| **Vulkan GPU** | ✅ 推荐 GPU 路径 | **必须先迁移到 llmexport.py 融合 Attn** | Prefill 2-4x, Decode 1.5-3x |

**核心结论更新**：GPU 加速在 MNN 框架下**不是不可行**——OpenCL 和 Vulkan 后端均已实现 `OpType_Attention` 融合 kernel。但当前 ONNX 分解路径**阻塞了 GPU 加速**，因为分解算子格式无法利用融合 Attention kernel。迁移到 llmexport.py 是解锁 GPU 加速的前置条件。

> **短中期策略**：CPU ARM NEON FP16 多线程（当前 Phase 3，已验证 ~20 tok/s）是见效最快的路径。GPU 加速排到 llmexport.py 迁移之后。

---

## 四、实施计划

### Step 1：AE 常驻 + WAV 移除 + Warmup 移动 ✅ 已完成

> **实施日期**：2026-06-08 | **实际耗时**：~2h | **成功率**：100%（实机验证通过）

**修改文件**（实际）：
| 文件 | 改动 |
|------|------|
| `qwen3_asr_engine.h` | +11 行：`#include <mutex>`、`ensureAudioEncoderLoaded()` 声明、`m_ae_mod`、`m_ae_loaded`、`m_decode_mutex` |
| `qwen3_asr_engine.cpp` | +49 / -57 行：AE 常驻 + WAV 移除 + warmup 移位 + `std::try_to_lock` 并发保护 + 移除 `<fstream>` |

**额外修复**：发现并修复多线程并发 `runDecoder()` 导致 SIGSEGV 的已有 bug（两个 silence-detection 线程同时触发 `endAudio()`），通过 `std::mutex` + `std::try_to_lock` 解决。

**实机验证结果**（Mate 30, Kirin 990, 8GB）：

```
首次 utterance:
  17:22:11.276  Loading audio encoder (first use, will keep resident)...
  17:22:11.682  Audio encoder loaded                                    (406ms)
  17:22:12.291  Audio encoder warmup complete (2× synthetic forward)   (609ms)
  17:22:12.291  Audio encoder resident
  17:22:14.568  LLM decoder loaded                                    (~2.3s)
  17:22:15.312  Generated 6 tokens
  → 总耗时 ~4.1s

第二次 utterance:
  17:22:20.531  Audio encoder already loaded, reusing  ✅ 不再加载！
  17:22:21.123  Decoder already loaded, reusing         ✅ Decoder 复用
  17:22:21.788  Generated 4 tokens
  → 总耗时 ~1.3s  🚀 3.2x 加速！

全日志搜索：无 "Audio encoder released" ✅
全日志搜索：无 SELinux EACCES 错误      ✅ (WAV 已移除)
全日志搜索：无 OOM / lmkd kill           ✅
```

**验证清单**：

- [x] 编译通过（`assembleDebug`）
- [x] 首次 utterance：`warmup complete` + `resident` 日志出现
- [x] 第二次 utterance：`already loaded, reusing` 日志出现
- [x] 无 `Audio encoder released` 旧版日志
- [x] 内存 < 1.5GB RSS（无 OOM）
- [x] 并发保护：`std::try_to_lock` 防止重复 `runDecoder()` 调用
- [x] ASR 识别结果正确（中文正常）

### Step 2：增量解码（~1.5-2d，成功率 ~80%）✅ 已完成并实机验证

> **实施日期**：2026-06-08 | **实际耗时**：~1h（代码实现）+ 实机验证 | **成功率**：100%

**修改文件**（实际）：
| 文件 | 改动 |
|------|------|
| `qwen3_asr_engine.h` | +17 行：`startDecode/decodeStep/isDecoding/getPartialResult` 声明 + 5 个 decode state 成员 |
| `qwen3_asr_engine.cpp` | +221 行：`startDecode()`(136 行) + `decodeStep()`(74 行) + `isDecoding/getPartialResult` + `reset()` 扩展 |
| `qwen3_asr_jni.cpp` | +65 行：`nativeStartDecode/nativeDecodeStep/nativeIsDecoding/nativeGetPartialResult` |
| `Qwen3AsrEngine.kt` | +48 行：4 个 streaming API 方法 + 4 JNI declarations |
| `VoiceChatPresenter.kt` | +42 行：streaming decode loop + `ASR_DECODING` state + `updateAsrPartialText` interface |

**实机验证结果**（Mate 30, Kirin 990, 8GB）：
- 3 轮 utterance 全部成功，无崩溃，识别结果完全正确
- 每 token ~46-56ms 逐字流式输出，UI 实时更新
- EOS/IM_END 正常终止循环
- AE/Decoder 复用正常（第 2+ 次跳过加载）

**验证清单**：
- [x] `startDecode()` 返回 true 后 `isDecoding()` 为 true
- [x] `decodeStep()` 逐步返回 token，VARP 在跨 JNI 调用间保持有效
- [x] EOS/IM_END token 时流式循环正常退出
- [x] `runDecoder()` fallback 路径代码保留
- [x] 多轮 utterance (3 轮) 无内存泄漏或崩溃
- [x] `ExecutorScope` 重新进入时 KV cache 正确持久化
- [x] 流式 UI 实时显示部分文本

### Step 3：CPU 微调（~1h，成功率 >90%）✅ 已完成并实机验证

> **实施日期**：2026-06-08 | **实际耗时**：~30min（代码实现）+ 实机验证 | **成功率**：100%

**修改文件**（实际）：
| 文件 | 改动 |
|------|------|
| `qwen3_asr_engine.h` | +10 行：`num_threads=4` 默认值、`m_executor` 持久化 executor、`m_decode_t0` 性能计时、`<chrono>` |
| `qwen3_asr_engine.cpp` | +71 / -15 行：构造函数 `m_num_threads(4)`、`init()` 持久 executor (FP16+Power_High)、`runDecoder()` 复用 executor、`startDecode()` FP16+Power_High、`decodeStep()` FP16+Power_High、prefill/decode 性能计时、Perf 日志 |
| `Qwen3AsrEngine.kt` | +2 / -2 行：`numThreads` 默认值 2→4 |

**实机验证结果**（Mate 30, Kirin 990, 8GB）：

| 指标 | 第 1 次 (8.3s 音频) | 第 2 次 (5s 音频) | 第 3 次 (7.3s 音频) |
|------|:--:|:--:|:--:|
| **AE 加载** | 390ms | 跳过 ✅ | 跳过 ✅ |
| **AE warmup** | 613ms | 跳过 ✅ | 跳过 ✅ |
| **AE 推理** | 1062ms | ~657ms | 897ms |
| **Decoder 加载** | 1774ms | 跳过 ✅ | 跳过 ✅ |
| **Prefill** | 741ms (S=126) | 495ms (S=85) | 663ms (S=114) |
| **Decode** | 592ms (11 tok) | 325ms (7 tok) | 344ms (7 tok) |
| **吞吐** | **18.6 tok/s** | **21.5 tok/s** | **20.3 tok/s** |
| **端到端延迟** | **~5.7s** | **~1.5s** | **~2.0s** |
| **识别结果** | "你好，北京。今天天气怎么样？" ✅ | "明天星期几？" ✅ | "今天星期几？" ✅ |

**FP16 精度结论**：3 轮中文识别完全正确，无同音错字、无乱码。ARM v8.2 FP16 对 Qwen3-ASR 精度无明显影响。

**日志确认 Phase 3 生效**：
```
Phase 3: Persistent executor created (threads=4, precision=FP16, power=High)
startDecode: ... executor=per-utterance(FP16+Power_High)
Perf [streaming]: decode=592ms, 11 tokens (18.6 tok/s, 53.8 ms/tok)
```

**验证清单**：
- [x] FP16 精度：3 轮中文识别完全正确，无精度损失
- [x] 性能：后续 utterance ~1.5-2.0s（vs 优化前 3.3-5.1s 预估），加速 2-3x
- [x] 内存：3 轮无 OOM、无 lmkd kill
- [x] 稳定性：3 轮无崩溃
- [x] `runDecoder()` executor 复用路径未被覆盖（streaming 是主路径），代码已就位

---

## 五、关键技术决策记录

### D1：为什么 AE 常驻而不继续串行化？

串行化（当前方案）追求最低内存峰值，适合低端设备。但在 Mate 30/40 (8GB) 上，500MB 额外内存可接受。且 AE 常驻带来的**首次 utterance 之后的延迟改善是 2-4s**，对用户体验提升巨大。低端设备可通过运行时内存检测回退到串行化。

### D2：为什么不移除 WAV 再保留 WAV 作为 fallback？

WAV 往返是纯浪费（序列化到文件再读回），没有正确性收益。移除后代码更简洁、更快、更精确。不保留。

### D3：为什么增量解码的 Executor 不复用？

一次解码会话内（prefill + all decode steps）复用一个 executor 是必须的（KV cache 依赖它）。但跨 utterance 的 executor 复用会导致内存碎片累积（MNN 内部内存池不会自动回收碎片）。每次新建 executor 的代价 ~10ms，可接受。

### D4：为什么不迁移到 llmexport.py 路径？

参考 `Qwen3-ASR-MNN-PROGRESS.md` 第二节："暂不迁移的理由"：
- 当前路径已实现 KV Cache (7.5x) + 8-bit 量化 + Android 端到端
- 迁移需修改 `llmexport.py` 的 `inputs_embeds` 假设，架构调整大
- 当前优化优先保证 Android 生产可用性

### D5：为什么不在 Kotlin 层做异步解码？

Kotlin 层的协程只能解决 UI 不阻塞问题，不能解决底层推理性能问题。真正的延迟瓶颈在 C++ 的同步阻塞式 `runDecoder()`。必须从 C++ 层拆解。

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 | 状态 |
|------|:---:|------|------|:--:|
| AE+Decoder 双模型 OOM | 低 | 崩溃 | 运行时内存检测 + 自动回退串行 | ✅ 实机无 OOM |
| **多线程并发 `runDecoder()`** | **已触发** | **SIGSEGV** | **`std::mutex` + `try_to_lock`（已修复）** | ✅ 已修复 |
| VARP 在跨 JNI 调用时失效 | 中 | 崩溃/错误输出 | 确保 Executor 存活；fallback 到 runDecoder | ✅ 实机 3 轮正常 |
| 增量解码 token 质量下降 | 低 | 识别准确度降低 | fallback 对比测试 | ✅ 识别准确无误 |
| Mali GPU OpenCL 崩溃 | 高 | N/A | **不使用 GPU，纯 CPU 路径** | N/A |
| FP16 精度损失过大 | 中 | 识别变差 | 先跑对比测试，不通过则回退 FP32 | ✅ 3/3 完全正确 |

### 6.1 并发 Bug 详情（2026-06-08 发现并修复）

**现象**：第三次 utterance 时 SIGSEGV，堆栈在 `MNN::ThreadPool::enqueueInternal`。

**根因**：Kotlin 层 silence-detection 的两个不同 chunk（Thread 10448 和 10450）在 4ms 内各自检测到 endpoint，同时从不同线程调用 `nativeEndAudio()` → `runDecoder()`。引擎的无锁状态（executor、KV cache、`m_ae_mod`）被两个线程同时读写，MNN 内部线程池状态损坏。

**修复**（`qwen3_asr_engine.h:9`、`qwen3_asr_engine.h:104`、`qwen3_asr_engine.cpp:395`）：
```cpp
// header:
#include <mutex>
std::mutex m_decode_mutex;

// runDecoder() 入口：
std::unique_lock<std::mutex> lock(m_decode_mutex, std::try_to_lock);
if (!lock.owns_lock()) {
    LOGW("runDecoder skipped: another decode is already in progress");
    return "";
}
```
策略：非阻塞 `try_to_lock`。如果已有解码在进行，重复调用直接返回空结果，不阻塞不崩溃。

---

## 七、向 llmexport.py 迁移计划（2026-06-09）

### 7.0 迁移目标与收益评估

#### 为什么需要迁移

当前 ONNX 导出路径（`export_qwen3_asr.py` → MNNConvert）存在两个结构性问题：

1. **分解算子阻塞 GPU 加速**：当前模型中的 Attention 被分解为 MatMul + Add + Softmax 逐 op，无法利用 MNN OpenCL/Vulkan 后端的 `OpType_Attention` 融合 Attention kernel。即使 GPU 后端已实现该 kernel，模型格式不支持。

2. **未集成进 MNN Omni 引擎**：当前手写 decode 循环无法使用引擎内置的采样策略（top-k/top-p/temperature）、prefix caching、以及 `llm.cpp` 的多模态推理管线。

#### 收益预估

| 维度 | 当前路径（ONNX 分解） | 迁移后（llmexport.py 融合） | 提升 |
|------|------|------|:--:|
| **CPU 推理** | ~20 tok/s (Phase 3 已验证) | ~22-24 tok/s（内存布局优化） | +10-20% |
| **GPU (Vulkan) 推理** | 不可用（分解算子阻塞） | 预期 ~40-60 tok/s | 2-3x |
| **GPU (OpenCL) 推理** | 不可用（分解算子阻塞） | 预期 ~30-50 tok/s | 1.5-2.5x |
| **采样策略** | 仅 argmax | top-k/top-p/temperature/min_p/... | 质量提升 |
| **Prefill 延迟** | ~500-700ms | ~200-400ms (FusedAttention + GPU) | 2-3x |
| **权重体积** | 575 MB (MNNConvert 8-bit) | ~300-400 MB (BF16 + MNN 内置量化) | -30-50% |
| **代码维护** | C++ 手写 decode loop (~300 行) | 引擎内置（0 行） | 大幅简化 |

#### 前置条件（已满足）

基于 2026-06-09 源码验证，以下关键基础设施**已经存在**：

| 基础设施 | 位置 | 状态 |
|----------|------|:--:|
| MNN OpenCL `OpType_Attention` fused kernel | `source/backend/opencl/execution/buffer/AttentionBufExecution.cpp:1854` | ✅ 已注册为 TRANSFORMER |
| MNN Vulkan `VulkanAttention` fused kernel | `source/backend/vulkan/buffer/execution/VulkanAttention.cpp` | ✅ 含 KV cache + FP16 |
| MNN OpenCL `RMSNorm` (LayerNorm with RMSNorm flag) | `source/backend/opencl/execution/buffer/LayerNormBufExecution.cpp:26,165` | ✅ |
| llmexport.py `FusedAttention` custom ONNX op | `transformers/llm/export/utils/custom_op.py:36-68` | ✅ |
| llmexport.py `FakeLinear` 权重卸载 | `transformers/llm/export/utils/custom_op.py:5-34` | ✅ |
| llm.cpp `is_audio()` → 自动检测 `inputs_embeds` | `transformers/llm/engine/src/llm.cpp:316-317` | ✅ |
| omni.cpp `qwen3_asr` audio_type 分支（whisper_fbank） | `transformers/llm/engine/src/omni.cpp:876-878` | ✅ |
| omni.cpp 音频嵌入注入（mAudioEmbeddings） | `transformers/llm/engine/src/omni.cpp:1176-1185` | ✅ |
| llmconfig.hpp `is_audio()` / `audio_type()` | `transformers/llm/engine/src/llmconfig.hpp:260-266` | ✅ |

> **核心洞察**：MNN 框架的 Omni 引擎已经为 `qwen3_asr` 开了分支，fbank 处理已实现。缺失的是 Python 侧的模型注册、导出适配，以及 C++ 侧的文本 embedding 注入逻辑。工程量比原预期小得多。

---

### 7.1 工作包分解

```
WP1: 模型注册 (model_mapper.py)          ~0.5 day
  └─→ WP2: 模型加载 (model.py)           ~0.5 day
       └─→ WP3: 模型适配 (transformers.py)  ~0.5 day
            └─→ WP4: 导出适配 (llmexport.py) ~0.5 day
                 └─→ WP5: C++ 引擎集成 (omni.cpp) ~1 day
                      └─→ WP6: 端到端验证            ~1 day
─────────────────────────────────────────────────────────
                        总计:                       ~4-5 days
```

---

### 7.2 WP1: 模型注册 — model_mapper.py

**修改文件**：`transformers/llm/export/utils/model_mapper.py`

**任务**：在 `ModelMapper.__init__()` 中调用新方法 `regist_qwen3asr()`，注册 Qwen3-ASR 的模型字段映射。

**实现**：

```python
def regist_qwen3asr(self):
    # Qwen3-ASR 模型结构：HuggingFace 中 audio_encoder + text_decoder 在同一个 model 下
    # ModelScope 路径: Qwen/Qwen3-ASR-0.6B
    # config.json 包含: text_config + audio_config
    qwen3asr_config = {
        'hidden_size':    'text_config.hidden_size',
        'head_dim':       'text_config.head_dim',
        'num_attention_heads':   'text_config.num_attention_heads',
        'num_hidden_layers':     'text_config.num_hidden_layers',
        'num_key_value_heads':   'text_config.num_key_value_heads',
        'rope_theta':     'text_config.rope_theta',
        'rope_scaling':   'text_config.rope_scaling',
        'max_position_embeddings': 'text_config.max_position_embeddings',
        'attention_type': 'text_config.attention_type',  # qwen3 用 GQA
        # Audio-specific configs (写入 llm_config.json 供 C++ 读取)
        'audio_type':     'qwen3_asr',
        'is_audio':       True,
        'audio_pad':      151676,   # <|audio_pad|>
        'audio_start':    151669,   # <|audio_start|>
        'audio_end':      151670,   # <|audio_end|>
    }
    qwen3asr_model = {
        'lm':     'lm_head',               # LM head (token prediction)
        'embed':  'model.embed_tokens',    # Text embedding layer
        'blocks': 'model.layers',          # 28 × Decoder layers
        'final_layernorm': 'model.norm',   # Final RMSNorm
        'audio':  'audio_encoder',         # Audio encoder (3×Conv2d + 18×Transformer)
    }
    qwen3asr_attention = {
        'q_proj': 'self_attn.q_proj',
        'k_proj': 'self_attn.k_proj',
        'v_proj': 'self_attn.v_proj',
        'o_proj': 'self_attn.o_proj',
        'q_norm': 'self_attn.attention.q_norm',  # Qwen3 特有: Q/K-Norm
        'k_norm': 'self_attn.attention.k_norm',
    }
    qwen3asr_map = {
        'config':    qwen3asr_config,
        'model':     qwen3asr_model,
        'decoder':   self.default_decoder,
        'attention': qwen3asr_attention,   # 使用 Qwen3 的 Q/K-Norm 模式
    }
    self.regist('qwen3_asr', qwen3asr_map)
```

**关键决策**：
- `attention_type` 不同：Qwen3-ASR 使用 GQA（28 heads Q, 4 heads KV），需要在 attention 中正确处理 `num_key_value_groups = 28//4 = 7`
- `audio` 字段映射到 `audio_encoder`（不是 `audio_tower`）→ 后续 `is_audio()` 时加载 `audio_model()`
- `is_audio: True` 写入 llm_config → C++ 引擎自动走 `inputs_embeds` 路径

**验证标准**：
- [ ] `LlmModel.from_pretrained('Qwen/Qwen3-ASR-0.6B')` 不报错
- [ ] `model.audio` 正确指向 audio_encoder
- [ ] model.blocks 包含 28 层 decoder
- [ ] model.embed 正确引用 `model.embed_tokens`

---

### 7.3 WP2: 模型加载 — model.py + llmexport.py

**修改文件**：
1. `transformers/llm/export/utils/model.py` — `get_model_class()`
2. `transformers/llm/export/llmexport.py` — `load_model()`

**任务**：Qwen3-ASR 的 HuggingFace model class 不存在于标准 transformers 库中（属于自定义模型）。需要类似 `lfm2_audio` 的处理方式——从 ModelScope/本地路径加载。

**实现（model.py:42-57）**：

```python
MODEL_CLASS_MAPPING = {
    # ... 现有条目保持不变 ...
    'qwen3_asr': None,   # Sentinel: 自定义加载路径，不使用标准 HF class
}
```

**实现（llmexport.py:54-130 load_model 扩展）**：

```python
# 在 load_model() 中增加 qwen3_asr 加载分支
if self.model_type == 'qwen3_asr':
    # Qwen3-ASR 不在 transformers 注册表中，使用 trust_remote_code
    from transformers import AutoModelForCausalLM
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto'
    )
    # 或者从 ModelScope 加载:
    # from modelscope import snapshot_download
    # model_dir = snapshot_download('Qwen/Qwen3-ASR-0.6B')
    # original_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
```

**验证标准**：
- [ ] 无需修改 transformers 源码即可加载模型
- [ ] `config.model_type = 'qwen3_asr'`
- [ ] `config.text_config.hidden_size = 1024`
- [ ] `config.text_config.num_hidden_layers = 28`
- [ ] audio encoder 权重正确加载到 `model.audio`

---

### 7.4 WP3: 模型适配 — transformers.py

**修改文件**：`transformers/llm/export/utils/transformers.py`

**任务**：确保 Attention 类正确处理 Qwen3-ASR 的特有结构。

**Qwen3-ASR Attention 特殊之处**：

```
Qwen3 Decoder Layer:
├── input_layernorm (RMSNorm)
├── self_attn
│   ├── q_proj, k_proj, v_proj  (GQA: 28 Q-heads, 4 KV-heads)
│   ├── q_norm, k_norm          (Q/K LayerNorm — Qwen3 特有)
│   ├── RoPE (通过 Rotary 类)
│   ├── FusedAttention           (llmexport.py 融合路径)
│   └── o_proj
├── post_attention_layernorm (RMSNorm)
└── mlp
    ├── gate_proj, up_proj       (SwiGLU)
    └── down_proj
```

**当前 Attention 类已支持的特性**（无需修改）：
- ✅ GQA (`num_key_value_groups`)
- ✅ `q_norm`/`k_norm` 在 RoPE 前应用（`qk_norm_after_rope = False`）
- ✅ `FusedAttention` 融合路径
- ✅ `FakeLinear` 权重卸载

**可能需要调整的点**：

1. **Q/K-Norm 实现**：确认 `q_norm`/`k_norm` 是 `RMSNorm` 而非标准 `LayerNorm`。在 Qwen3-ASR 的 config 中，`text_config.qk_norm` 应设为 `'rms_norm'`。

2. **Audio Embedding 输入**：Decoder 接受 `inputs_embeds` (float32) 而非 `input_ids` (int32)。`Embedding.forward()` 在当前代码中返回 `inputs_embeds.view(-1, 1, hidden_size)`，这兼容两种输入格式。

3. **Position IDs 计算**：Audio frames 需要正确的 position IDs。在 ONNX 导出时，position_ids 作为外部输入传入，由 C++ 端计算。

**验证标准**：
- [ ] `model.blocks[i].self_attn.export_fused_attn = True` 能正常执行
- [ ] FusedAttention 的 `forward()` 在 ONNX trace 中正确生成 `LlmExporter::FusedAttention` op
- [ ] Q/K-Norm 在 RoPE 之前正确应用

---

### 7.5 WP4: 导出适配 — llmexport.py

**修改文件**：`transformers/llm/export/llmexport.py`

**任务**：适配 ONNX 导出流程和 config 生成。

#### 7.5.1 dynamic_axes 适配（第 75-79 行）

当前代码硬编码 `input_ids` 作为第一个输入名。对 audio 模型，实际输入是 `inputs_embeds`：

```python
# 当前:
self.model_dynamic_axes = {
    "input_ids" : { 0: "seq_len" },
    "attention_mask" : { 2: "seq_len", 3: "seq_len" },
    "position_ids" : { 1: "seq_len" },
}

# 修改为（qwen3_asr 模式）:
if self.model_type == 'qwen3_asr':
    self.model_dynamic_axes = {
        "inputs_embeds" : { 0: "seq_len" },
        "attention_mask" : { 2: "seq_len", 3: "seq_len" },
        "position_ids" : { 1: "seq_len" },
    }
```

#### 7.5.2 ONNX 导出适配（第 473-533 行 `export_onnx()`）

Qwen3-ASR 的 decoder 输入是 embedding 而非 token IDs。当前代码在第 488 行已经做了 `input_ids = model.embedding(input_ids)`，对于 qwen3_asr，可以直接复用这个模式，但需要将 input name 改为 `inputs_embeds`：

```python
# 标准 LLM: input_ids (int32) → embedding → decoder
# Qwen3-ASR: 外部已嵌入 → 直接传 inputs_embeds (float32)
#   对于 ONNX 导出，两者等价 — 因为 FakeLinear 替代了实际权重，
#   导出时看到的都是 float32 tensor
```

**实际上当前代码已经兼容**：`input_ids = model.embedding(input_ids)` 行 488 将 int token 转为 float embedding，之后的 ONNX 图接受 float32 输入。只需要确认 C++ 端 `inputNames = {}` (auto-detect) 能正确识别 `inputs_embeds` 而非 `input_ids`。

#### 7.5.3 config.json 生成（第 304-366 行 `export_config()`）

对 qwen3_asr，需在 config 中添加音频相关配置：

```python
if self.model_type == 'qwen3_asr':
    config.update({
        'is_audio': True,
        'audio_type': 'qwen3_asr',
        'audio_model': 'audio_encoder.mnn',
        'audio_pad': 151676,
        'audio_start': 151669,
        'audio_end': 151670,
        'system_prompt': 'You are a helpful assistant.',
    })
```

#### 7.5.4 音频编码器导出（新增）

需要在 llmexport.py 中增加 `export_audio()` 方法，将 HF 模型的 audio_encoder 导出为独立的 MNN 模型：

```python
def export_audio(self):
    if self.audio is None:
        return
    # Audio encoder 导出到 ONNX
    audio_onnx = self.audio.export(self.onnx_path)
    # 转换为 MNN
    if self.mnn_converter:
        MNNConverter(self, None).export(audio_onnx, 
                                         quant_bit=0,        # AE 不量化（精度关键）
                                         transformer_fuse=True)  # AE 内部的 Attention 可融合
```

**验证标准**：
- [ ] ONNX 导出成功，模型接受 `inputs_embeds` 作为第一输入
- [ ] ONNX 图中包含 `LlmExporter::FusedAttention` 自定义 op（非分解 MatMul）
- [ ] MNN 转换成功，`llm.mnn` + `audio_encoder.mnn` 均生成
- [ ] `llm_config.json` 正确包含 `is_audio: true` 和 `audio_type: qwen3_asr`
- [ ] `config.json` 正确包含 `audio_pad/audio_start/audio_end` token IDs

---

### 7.6 WP5: C++ 引擎集成 — omni.cpp

**修改文件**：
1. `transformers/llm/engine/src/omni.cpp` — 完善 qwen3_asr 分支
2. `transformers/llm/engine/src/llmconfig.hpp` — 无需修改（已通用）

**任务**：C++ 侧已有 `qwen3_asr` 音频处理分支和 `mAudioEmbeddings` 注入逻辑。需要补充/完善的是文本 prompt token 嵌入注入。

#### 7.6.1 当前已实现（无需修改）✅

```
omni.cpp:876-878  → whisper_fbank()  (audio_type == "qwen3_asr")
omni.cpp:1176     → AUDIO_PAD 位置注入 mAudioEmbeddings
omni.cpp:152      → mAudioModule 加载 (audio_encoder.mnn)
```

#### 7.6.2 需要实现的部分

**A. Token 序列构建**：在 `Omni::prefillForGraph()` （或等效的 embedding 构建函数）中，需要构建含 system prompt + audio token 的正确 token 序列，然后调用 `Llm::embedding()` 生成文本部分的 embeddings。

当前手写代码的等效逻辑（`qwen3_asr_engine.cpp:253-285`）需要迁移到 omni.cpp：

```cpp
// 构建 token 序列（伪代码）
std::vector<int> tokens;
// 1. System prefix (从 tokenizer encode system_prompt)
tokens.insert(end, prefix_tokens);  // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n...
// 2. Audio tokens
tokens.push_back(mAudioStart);       // 151669
tokens.insert(end, T, mAudioPad);   // 151676 × T
tokens.push_back(mAudioEnd);         // 151670
tokens.push_back(mImEnd);            // 151645
// 3. Assistant prefix
tokens.push_back(mImStart);          // 151644
tokens.push_back(assistant_id);      // 77091
```

然后将文本部分的 token 通过 `Llm::embedding()` 生成 embeddings，再将 AUDIO_PAD 位置替换为 audio encoder 输出。

**B. 关键问题**：当前 omni.cpp 的 `mAudioEmbeddings` 注入逻辑在 L1176 行，通过遍历 token 序列遇到 `mAudioPad` 时替换。需要确认这个逻辑在融合到 `Llm::forward()` 之前正确。

**C. Executor/Module 加载顺序**：Qwen3-ASR 的 Audio Encoder 和 LLM Decoder 使用不同的 MNN Module。omni.cpp 已经支持这个模式（`mAudioModule` + `mModule`），无需额外修改。

#### 7.6.3 system prompt 处理

当前 C++ 引擎中 system prompt 通过 `llm_config.json` 的 `system_prompt` 字段传入。omni.cpp 在 `Omni::load()` 或 `Llm::load()` 中通过 tokenizer 编码。但 Qwen3-ASR 的 system prompt 是音频 pipeline 的一部分，需要在 audio embedding 注入前应用到文本 embedding 构建中。

建议方案：在 `llm_config.json` 中设置 `"system_prompt": "You are a helpful assistant."`，让 LLM 引擎在 audio embedding 注入时自动应用。

**验证标准**：
- [ ] Audio encoder 加载成功（`mAudioModule`）
- [ ] `whisper_fbank()` 正确生成 mel spectrogram
- [ ] Audio encoder 推理输出正确的 embedding shape `[T_audio_frames, 1, 1024]`
- [ ] Token 序列正确构建（含 system prompt + audio tokens）
- [ ] Text embeddings 正确生成并注入到 audio pad 位置
- [ ] Merged embeddings 传入 Decoder prefill → first token 生成正确
- [ ] Decode loop 使用引擎内置采样策略

---

### 7.7 WP6: 端到端验证

**任务**：在 x86 服务器上先验证，再到 Android Mate 40 实机验证。

#### 7.7.1 x86 服务器验证

```bash
# Step 1: 导出 Qwen3-ASR 模型（llmexport.py 路径）
cd transformers/llm/export
python llmexport.py \
    --path /path/to/Qwen3-ASR-0.6B \
    --export mnn \
    --dst_path ./Qwen3-ASR-MNN \
    --quant_bit 8 \
    --transformer_fuse

# Step 2: 验证导出产物
ls ./Qwen3-ASR-MNN/
# 期望:
#   llm.mnn + llm.mnn.weight          (LLM Decoder with FusedAttention)
#   audio_encoder.mnn                  (Audio Encoder)
#   embeddings_bf16.bin               (Text embeddings)
#   tokenizer.txt / tokenizer.mtok    (Tokenizer)
#   llm_config.json                   (含 is_audio:true, audio_type:qwen3_asr)
#   config.json                        (含 system_prompt, audio_pad/start/end)

# Step 3: 运行 LLM demo 验证
cd build
./llm_demo ./Qwen3-ASR-MNN/config.json test.wav
# 期望: 正确输出语音转录文本
```

#### 7.7.2 Android 实机验证（Mate 40, Kirin 9000）

```bash
# Step 1: 推送模型到手机
adb push Qwen3-ASR-MNN /data/local/tmp/mnn_models/Qwen3-ASR-MNN/

# Step 2: 配置 config.json 使用 Vulkan GPU（可选）
# 修改 config.json: "backend_type": "vulkan"

# Step 3: 编译 APK
cd apps/Android/MnnLlmChat
./gradlew assembleDebug

# Step 4: 安装并测试
adb install app/build/outputs/apk/googleplay/debug/app-googleplay-debug.apk
# 运行 Qwen3AsrTestActivity → 验证 ASR 功能

# Step 5: 性能对比
# CPU (Phase 3 baseline): ~20 tok/s
# Vulkan GPU (目标):        ~40-60 tok/s
# CPU FusedAttention:        ~22-24 tok/s
```

#### 验证清单

| 验证项 | 环境 | 标准 |
|--------|:--|------|
| 导出成功，无报错 | x86 | 6 个文件生成 |
| llm_demo 端到端推理 | x86 | 正确转写中文 |
| Audio encoder 输出一致 | x86 | cosim > 0.999 vs 当前路径 |
| Decoder 首 token 一致 | x86 | token ID 匹配当前路径 |
| FusedAttention op 存在 | x86 | MNN 模型不含分解 MatMul |
| APK 编译成功 | Android | assembleDebug 无报错 |
| CPU 模式正常 | Mate 40 | 中文识别正确，~22 tok/s |
| Vulkan GPU 模式正常 | Mate 40 | 中文识别正确，~40-60 tok/s |
| OpenCL GPU 模式正常 | Mate 40 | 中文识别正确（可接受偶尔 fallback）|
| 多轮 utterance 无崩溃 | Mate 40 | 5 轮以上稳定 |
| 内存 < 1.5GB RSS | Mate 40 | 无 lmkd kill |

---

### 7.8 风险与注意事项

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|------|------|
| Qwen3-ASR 的 HF 格式不兼容 llmexport.py 加载器 | 中 | 阻塞 WP2 | 参考 lfm2_audio 的自定义加载模式；备选：从当前 `export_qwen3_asr.py` 中提取模型加载逻辑 |
| FusedAttention 在 Qwen3-ASR 上精度异常 | 低 | 识别质量下降 | 对比 FusedAttention vs 分解 MatMul 的 cosine similarity |
| Audio encoder MNN 转换失败 | 低 | 阻塞 WP4 | Audio encoder 导出已在 `export_qwen3_asr.py` 中验证通过，可复用 |
| Mali Vulkan 驱动在处理 28 层模型时崩溃 | 中 | GPU 不可用 | 回退到 CPU 路径；Phase 3 CPU 已充分优化 |
| `inputs_embeds` auto-detect 失败 | 低 | Decoder 加载失败 | 显式指定 inputNames（已有 fallback 机制） |

### 7.9 回滚策略

迁移采用**双轨并行**策略：
1. 保留当前 `qwen3_asr_engine.cpp` 全部代码不动
2. 新路径在 `omni.cpp` 中独立实现
3. Android 端通过 config flag 切换：
   - `use_omni_engine=false` → 走当前手写路径（已有功能，已验证）
   - `use_omni_engine=true` → 走 llmexport.py 融合路径（新功能）

这样可以安全地验证新路径，出问题立即可回退。
