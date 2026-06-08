# Qwen3-ASR 生产级实时流式优化计划

> 创建：2026-06-08 | 最后更新：2026-06-08
> 基于：qwen3_asr_engine.cpp v5（Android 端到端推理成功，中文 OK）
> 参考：MNN 框架源码深度分析（HiAI/OpenCL/Vulkan 后端 + audio 管线 + Executor 生命周期）
>
> **状态：Phase 1 ✅ 已完成并实机验证 | Phase 2 ✅ 代码完成待实机验证 | Phase 3 待实施**

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

## 三、NPU/GPU 加速可行性结论

### 分析过程

检查了 MNN 以下源码：
- `source/backend/hiai/` — HiAI NPU 后端（53 个 op 实现文件）
- `source/backend/opencl/` — OpenCL GPU 后端
- `source/backend/vulkan/` — Vulkan GPU 后端
- `include/MNN/MNNForwardType.h` — 后端类型定义
- `include/MNN/Interpreter.hpp` — HintMode / setHint 机制

### HiAI NPU：完全不可行

| NPU 缺失的必需 Op | 在 Qwen3-ASR 中的用途 |
|---|---|
| **RMSNorm** | Qwen3 全模型使用（非标准 LayerNorm），每层 2 次 |
| **RoPE** (Rotary Position Embedding) | 每层 Attention 的核心位置编码 |
| **SiLU** (Swish) | FFN 激活函数 |
| **5D Tensor** | KV Cache `[28, 1, 8, seq_len, 128]` |
| **Dynamic Shape** | Decode 每步 seq_len 变化 |
| **Causal Mask** | 每一步不同的下三角 mask |

MNN HiAI 后端（2019 年开发）专门针对 CNN 模型（Conv2D/Pooling/ReLU），不支持现代 Transformer/LLM 架构。

### OpenCL GPU：高风险不可行

OpenCL 后端 op 覆盖比 NPU 宽，但关键问题在 Kirin 平台：
1. **Mali GPU OpenCL 驱动质量差**：已知内存泄漏、崩溃、精度 bug
2. **动态 shape 支持有限**：KV Cache 的 seq_len 动态变化大概率触发 CPU fallback
3. **MNN OpenCL 后端主要针对 Qualcomm Adreno**，Kirin Mali 未经充分测试

### 结论

**NPU/GPU 不可行**，非硬件性能问题，而是 MNN 后端对 Transformer/LLM 专用算子支持的缺失。最优路径是 **CPU + ARM NEON (SDOT/FP16) + 多线程**，这也是 MNN LLM 引擎（`llm.cpp`）使用的路径。

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

### Step 2：增量解码（~1.5-2d，成功率 ~80%）✅ 代码完成

> **实施日期**：2026-06-08 | **实际耗时**：~1h（代码实现）| **待实机验证**

**修改文件**（实际）：
| 文件 | 改动 |
|------|------|
| `qwen3_asr_engine.h` | +17 行：`startDecode/decodeStep/isDecoding/getPartialResult` 声明 + 5 个 decode state 成员 |
| `qwen3_asr_engine.cpp` | +221 行：`startDecode()`(136 行) + `decodeStep()`(74 行) + `isDecoding/getPartialResult` + `reset()` 扩展 |
| `qwen3_asr_jni.cpp` | +65 行：`nativeStartDecode/nativeDecodeStep/nativeIsDecoding/nativeGetPartialResult` |
| `Qwen3AsrEngine.kt` | +48 行：4 个 streaming API 方法 + 4 JNI declarations |
| `VoiceChatPresenter.kt` | +42 行：streaming decode loop + `ASR_DECODING` state + `updateAsrPartialText` interface |

**架构设计**：
- **Executor 生命周期**：`startDecode()` 创建 → 存储为 `m_decode_executor` → 所有 `decodeStep()` 通过 `ExecutorScope` 重新进入上下文 → 解码完成时释放
- **状态管理**：`m_decoding_active` 防止无效调用；`m_decode_S`/`m_decode_gen_len`/`m_decode_current_token` 保持解码进度
- **线程安全**：`startDecode()` 复用 `m_decode_mutex` + `std::try_to_lock`（与 `runDecoder()` 相同）
- **Fallback**：`runDecoder()` 保持不变，`endAudio()` 路径完整保留
- **流式 UI**：解码循环每步 (~20-30ms) 后，发现文本变化即 `lifecycleScope.launch` 推送到主线程

**待验证**：
- `startDecode()` 返回 true 后 `isDecoding()` 为 true
- `decodeStep()` 逐步返回 token，VARP 在跨调用间保持有效
- EOS/IM_END token 时流式循环正常退出
- 保留 `runDecoder()` fallback 路径功能正常
- 多轮 utterance 无内存泄漏或崩溃
- `ExecutorScope` 重新进入时 KV cache 正确持久化

**已验证**：
- 代码语法级别 review 通过（C++ header/source consistency, JNI 签名匹配, Kotlin 类型安全）
- `VoiceChatView.updateAsrPartialText` 使用 default empty implementation，不破坏现有实现

### Step 3：CPU 微调（~1h，成功率 >90%）

- 线程数调整（2→4）+ 性能测试
- FP16 精度验证（对比 token 序列）
- Executor 复用

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
| VARP 在跨 JNI 调用时失效 | 中 | 崩溃/错误输出 | 确保 Executor 存活；fallback 到 runDecoder | Phase 2 |
| 增量解码 token 质量下降 | 低 | 识别准确度降低 | fallback 对比测试 | Phase 2 |
| Mali GPU OpenCL 崩溃 | 高 | N/A | **不使用 GPU，纯 CPU 路径** | N/A |
| FP16 精度损失过大 | 中 | 识别变差 | 先跑对比测试，不通过则回退 FP32 | Phase 3 |

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
