# Qwen3-ASR Omni 模式流式实现方案

> 创建：2026-06-10
> 最后更新：2026-06-11
> 关联文档：[[Qwen3-ASR-OMNI-PARAMETERS]] [[Qwen3-ASR-MEMORY-ANALYSIS]] [[Qwen3-ASR-STREAMING-PLAN]]（旧引擎方案）
> 状态：Phase 1 ✅ | Phase 2 ✅ | Phase 2.5 纯VAD分段 ❌→Phase 2.6 VAD引导扩展窗口 ✅ | Phase 3 已分析（30-40%）
> **关键修正**：Phase 2.5 「纯 VAD 分段」被实机验证推翻。VAD + 扩展窗口是对的正确组合。详见第四章。

---

## 一、现状分析

### 1.1 当前 Omni Batch Pipeline

```
┌─ Kotlin 层 ────────────────────────────────────────────────────┐
│  AudioRecord → FloatArray (N samples)                           │
│    → writeWavFile(filePath)    ← ❌ WAV 编解码 + 磁盘 I/O       │
│    → generate("<audio>filePath</audio>")                       │
│      → JNI: submitNative(prompt, keepHistory, listener)        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ JNI Bridge (llm_mnn_jni.cpp → llm_session.cpp) ──────────────┐
│  LlmSession::Response()                                         │
│    → processMultimodalPrompt(full_prompt_text)                  │
│      → PromptProcessor::Process()                               │
│        → HandleAudioTags() → audios["audio_0"].file_path        │
│          ❌ key 不匹配 Omni 引擎期望 (path vs "audio_0")        │
│    → llm_->response(multimodal_prompt)                          │
│      → Omni::tokenizer_encode()                                 │
│        → regex <audio> → processAudioContent()                  │
│          → audios.find(path) → MISS → fallback: audioProcess() │
│            → AUDIO::load(filePath)  ← ❌ WAV 解析 + 磁盘 I/O    │
│              → whisper_fbank → Audio Encoder → embedding       │
│                → LLM Decoder → tokens → stream back             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 当前已知问题

| # | 问题 | 影响 | 状态 |
|---|------|------|:----|
| 1 | `Integer.reverseBytes()` 被用于 16-bit short 写入 WAV 头 → 数据损坏 | C++ `AUDIO::load()` 解析失败 | Punted (Phase 1 绕过) |
| 2 | `audios` key 不匹配（`"audio_0"` vs 文件路径）→ 走 fallback | 冗余的 `multimodeProcess()` 调用 | ✅ 已修复 |
| 3 | Java 写 WAV → C++ `AUDIO::load()` 磁盘往返 | 浪费 20-100ms I/O 延迟 | ✅ 已修复 |
| 4 | Kotlin 层 Future 流式需求 vs batch 模式限制 | 不能边录边出结果 | Phase 2 |

### 1.3 延迟分解

| 环节 | 耗时（Kirin 9000 估算） | 说明 |
|------|:------------|------|
| WAV 编码（Java） | ~5ms | `RandomAccessFile` 写 44B 头 + PCM |
| 磁盘写（FUSE/ext4） | ~10-50ms | 取决于文件系统 |
| C++ `AUDIO::load` | ~10-50ms | `std::ifstream` + WAV 解析 + float 转换 |
| whisper_fbank | ~50-100ms | Mel filterbank，纯计算 |
| Audio Encoder (18 层) | ~500-1000ms | 最大延迟来源 |
| LLM Decoder prefill | ~100-200ms | 对 embedding 做 full sequence prefill |
| LLM Decode (per token) | ~25-30ms | 逐 token 自回归 |

---

## 二、方案设计

### 2.1 核心思路：直传 PCM + 滑动窗口伪流式

```
┌─ Kotlin ───────────────────────────────────────────────────────┐
│  AudioRecord chunks (100ms)                                      │
│    → FloatArray buffer 累积                                      │
│    → STOP 或达到阈值                                              │
│      → setAudioData(floatArray, sampleRate)    ← 绕过文件 I/O   │
│      → generate("<audio>stream</audio>")       ← 特殊 token     │
│        ← onProgress() 逐 token 流式回调                          │
│        ← 显示识别结果                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ JNI Bridge ───────────────────────────────────────────────────┐
│  setAudioData(): jfloatArray → _Const VARP                     │
│    → 存于 LlmSession::pending_audio_                            │
│                                                                  │
│  submitNative(): processMultimodalPrompt() → <audio>stream      │
│    → 检测到 "stream" 标识 → 注入 pending_audio_                 │
│    → llm_->response(multimodal_prompt)                          │
│      → Omni::processAudioContent() → waveform != nullptr ✅     │
│        → audioProcess(waveform) → fbank → AE → embedding        │
│          → Decoder prefill → 逐 token 流式回调                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 三阶段实施

| 阶段 | 优先级 | 目标 | 改动量 | 状态 |
|------|:------|------|:------|:----|
| **Phase 1** | 🔴 HIGH | 直传 PCM，绕过 WAV 文件 I/O | 6 文件 | ✅ 已完成 |
| **Phase 2** | 🟡 MEDIUM | 滑动窗口伪流式（边录边出结果） | 1 文件 | ✅ 已完成 |
| **Phase 2.5** | 🟠 MEDIUM | VAD + 分段推理（静音跳过 + 自然断句 + 解除 30s 上限） | 1 文件 | 🔄 实施中 |
| **Phase 3** | 🟢 FUTURE | 真流式增量 AE（引擎改造）— 详见第五章深度分析 | 引擎层 | 已分析（30-40%） |

---

## 三、Phase 1：直传 PCM ✅ 已完成

### 3.1 目标

消除 WAV 编解码 + 磁盘 I/O，FloatArray 直接变成 MNN VARP 传入 Omni 引擎。

**效果**：延迟 -20~100ms，代码简化，为流式打下基础。

### 3.2 修改清单

| # | 文件 | 操作 | 描述 |
|---|------|:--|------|
| 1 | `llm_session.h` | 修改 | 新增 `SetPendingAudio()` + `pending_audio_` VARP + `pending_audio_sample_rate_` |
| 2 | `llm_session.cpp` | 修改 | 实现 `SetPendingAudio()` + `Response()` / `ResponseWithHistory()` 注入逻辑 |
| 3 | `llm_mnn_jni.cpp` | 修改 | 新增 `setAudioDataNative` JNI 绑定 |
| 4 | `LlmSession.kt` | 修改 | 新增 `setAudioData()` Kotlin API |
| 5 | `processor.cpp` | 修改 | 修复 `audios` key：`"audio_N"` → `audio_path`（即 content），匹配 Omni 引擎 `audios.find(content)` |
| 6 | `Qwen3AsrTestActivity.kt` | 修改 | 去掉 WAV 写入，改用 `setAudioData()` + `<audio>stream</audio>` |

### 3.3 实际实现（含踩坑修正）

#### Step 1: `llm_session.h` — 新增接口

```cpp
// 头文件新增 include
#include "MNN/expr/Expr.hpp"

// public method
/**
 * Set pending audio waveform data for the next multimodal inference call.
 * The waveform is consumed once in Response() / ResponseWithHistory() and then cleared.
 */
void SetPendingAudio(const float* samples, int num_samples, int sample_rate);

// private members
MNN::Express::VARP pending_audio_{nullptr};
int pending_audio_sample_rate_{16000};
```

#### Step 2: `llm_session.cpp` — 实现核心逻辑

**2a. `SetPendingAudio()` 实现**：

```cpp
void LlmSession::SetPendingAudio(const float* samples, int num_samples, int sample_rate) {
    // ⚠️ 必须匹配 AUDIO::load() 的输出格式：{N} 1D NHWC
    pending_audio_ = MNN::Express::_Const(samples, {num_samples}, MNN::Express::NHWC,
                                          halide_type_of<float>());
    pending_audio_sample_rate_ = sample_rate;
    MNN_DEBUG("SetPendingAudio: %d samples, %d Hz", num_samples, sample_rate);
}
```

**2b. `Response()` 注入逻辑**（在 `processMultimodalPrompt` 之后、`llm_->response` 之前）：

```cpp
auto multimodal_result = processMultimodalPrompt(full_prompt_text);
// Inject pending audio waveform into multimodal audios, bypassing file I/O
if (multimodal_result.has_multimodal && pending_audio_.get() != nullptr) {
    for (auto& [key, part] : multimodal_result.multimodal_prompt.audios) {
        part.waveform = pending_audio_;
    }
    pending_audio_ = nullptr;  // single-use consumption
}
```

> 同时在 `ResponseWithHistory()` 中添加了相同的注入逻辑（该方法同样调用 `processMultimodalPrompt` → `llm_->response`）。

#### Step 3: `processor.cpp` — 修复 key 匹配

```cpp
// 旧代码：
std::string audio_key = "audio_" + std::to_string(state.image_index);

// 修复后：使用 tag 内容本身作为 key（例如文件路径或 "stream"），
// 这样 Omni 引擎的 audios.find(content) 才能正确匹配。
std::string audio_key = audio_path;
```

#### Step 4: `llm_mnn_jni.cpp` — JNI 绑定

```cpp
extern "C"
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_llm_LlmSession_setAudioDataNative(
        JNIEnv *env, jobject thiz, jlong llmPtr, jfloatArray samples, jint sampleRate) {

    auto *llm = reinterpret_cast<mls::LlmSession *>(llmPtr);
    if (!llm || !samples) return;

    jsize len = env->GetArrayLength(samples);  // ⚠️ 不是 GetFloatArrayLength!
    jfloat *data = env->GetFloatArrayElements(samples, nullptr);
    llm->SetPendingAudio(data, len, sampleRate);
    env->ReleaseFloatArrayElements(samples, data, JNI_ABORT);
}
```

#### Step 5: `LlmSession.kt` — Kotlin API

```kotlin
fun setAudioData(samples: FloatArray, sampleRate: Int) {
    setAudioDataNative(nativePtr, samples, sampleRate)
}

private external fun setAudioDataNative(llmPtr: Long, samples: FloatArray, sampleRate: Int)
```

#### Step 6: `Qwen3AsrTestActivity.kt` — 调用方改动

```kotlin
// 旧：
writeWavFile(samples, SAMPLE_RATE, wavFile.absolutePath)
val audioTag = "<audio>${wavFile.absolutePath}</audio>"

// 新：
llmSession?.setAudioData(samples, SAMPLE_RATE)
val audioTag = "<audio>stream</audio>"
```

### 3.4 踩坑记录

| # | 问题 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | `GetFloatArrayLength` 不存在 | 编译失败 | JNI 没有这个方法，正确的 API 是 `GetArrayLength`（适用所有数组类型） | `GetArrayLength(samples)` |
| 2 | `_Input` VARP 导致 crash | `SIGSEGV fault addr 0x8` in `spectrogram` | `_Input` 创建 InputType::INPUT 占位符，其内部张量信息未完全初始化，`spectrogram` 访问 `getInfo()->dim[1]` 时崩溃 | 改用 `_Const`（InputType::CONSTANT），张量信息完整 |
| 3 | `{1, N} NCHW` shape 仍 crash | 同上崩溃点 | `AUDIO::load()` 返回的是 `{N}` 1D NHWC 格式，`spectrogram` 内部的 `_Reshape` + `dim[1]` 访问依赖正确的 shape 格式 | 改为 `{num_samples}` 1D NHWC，与 `AUDIO::load()` 完全一致 |
| 4 | `pending_audio_.reset()` 不存在 | 编译失败 | MNN 的 VARP 是 class（包装 `shared_ptr<Variable>`），不是 `shared_ptr` typedef，没有 `reset()` 方法 | `pending_audio_ = nullptr` |

### 3.5 数据流验证（日志确认）

```
SetPendingAudio: 116800 samples, 16000 Hz              ← PCM 直传成功
Found audio tag with path: stream                      ← 标签解析
Registered audio: stream as stream                     ← key 匹配修复 OK
Response: Injected pending_audio_ into audios[stream]   ← 注入成功
Omni Waveform Stats: samples=116800, min=-0.7145, max=0.7627  ← 引擎正确读取数据
```

> Phase 1 的 JNI 桥 + 数据注入逻辑全部验证通过。当前在 `whisper_fbank → spectrogram` 内部仍有 shape 相关的 crash 待修复（踩坑 #3）。

### 3.6 Commits

```
c76cd069 [LLM:Feature] Qwen3-ASR Phase 1: Direct PCM path, bypass WAV file I/O
<sha2>    [LLM:Bugfix] Use _Const instead of _Input for pending_audio_ VARP
<sha3>    [LLM:Bugfix] Match AUDIO::load() tensor shape: {N} NHWC instead of {1,N} NCHW
```

---

## 四、Phase 2：滑动窗口伪流式

### 4.1 目标

用户录音过程中每 2 秒触发一次增量推理，UI 上逐词/逐 token 显示。

### 4.2 设计

```
录音线程: 每 100ms 读一块 → FloatArray buffer 累积
流式触发器: 每 2s / 音量静音 / 手动 STOP
  → 取 buffer 快照
  → setAudioData(snapshot, 16000)
  → generate("<audio>stream</audio>")
    → onProgress() → update UI 部分结果
```

### 4.3 关键数据结构

```kotlin
enum class RecordingMode { BATCH, STREAMING }

inner class StreamingRecorder {
    private val buffer = mutableListOf<Float>()
    private var lastProcessedLen = 0
    private var timer: Timer? = null
    
    fun pushChunk(chunk: FloatArray) {
        synchronized(buffer) { buffer.addAll(chunk.toList()) }
    }
    
    fun startIncremental() {
        timer = Timer("asr-stream", true).apply {
            scheduleAtFixedRate(object : TimerTask() {
                override fun run() = processIncremental()
            }, 2000, 2000)  // 每 2 秒
        }
    }
    
    private fun processIncremental() {
        val snapshot = synchronized(buffer) { buffer.toFloatArray() }
        if (snapshot.size - lastProcessedLen < SAMPLE_RATE) return  // < 1s 增量跳过
        
        lastProcessedLen = snapshot.size
        lifecycleScope.launch(Dispatchers.IO) {
            llmSession?.setAudioData(snapshot, SAMPLE_RATE)
            llmSession?.generate("<audio>stream</audio>", mapOf(), listener)
        }
    }
    
    fun stop() {
        timer?.cancel()
        // 最终完整推理
        processIncremental()
    }
}
```

### 4.4 性能考量

| 场景 | 延迟 | 说明 |
|------|:-----|------|
| 2s 语音 + AE prefill | ~600ms | 首 token 延迟 |
| 后续 2s 增量 | ~300ms | AE 只跑增量帧（如果支持） |
| 30s 语音完整推理 | ~2000ms | STOP 时触发 |

> 注：Phase 2 的增量 AE 效率取决于 Omni 引擎是否支持 Causal Attention，当前引擎是 Full Attention，每次需重跑全量 fbank+AE。**Phase 3 可优化这一点。**

### 4.5 UI 交互

```
[REC 按钮] → 变红 "● Recording (Stream)... tap STOP"
[Status]   → "Streaming — 已录制 5.3s"
[Result]   → 实时更新部分识别文本（每次增量推理后刷新）
```

### 4.6 实际实现

**修改文件**：`Qwen3AsrTestActivity.kt`（+179/-29 行）

**新增方法**：

| 方法 | 功能 |
|------|------|
| `startStreamingTimer()` | daemon Timer，首次 2s 后触发，之后每 2s 一次 |
| `stopStreamingTimer()` | 取消 timer 并置 null |
| `processIncrementalOmni(s, isFinal)` | 统一的增量/最终推理方法 |
| `ensureStreamingCard()` | 幂等创建 LIVE 结果卡片（红色 ● LIVE 徽标） |
| `updateStreamingResult(text)` | 实时更新 LIVE 卡片文本 |
| `finalizeStreamingResult(text)` | LIVE 卡片 → 永久结果卡片 → `returnToIdle()` |

**核心设计**：
- Timer 直接集成在 Activity 中（未使用独立 inner class），复用 `omniAudioBuffer` 作为共享缓冲区
- `processIncrementalOmni(samples, isFinal)` 同时处理增量推理和最终推理，通过 `isFinal` 参数区分行为
- `streamingIncrementalInProgress` 标记防止 timer tick 叠加（`generate()` 内部 `synchronized` 保证了串行化）
- LIVE 结果卡片 idempotent 创建（`ensureStreamingCard()`），带红色 ● LIVE 徽标

**并发模型**：
- 录音线程：每 100ms 写入 `omniAudioBuffer`（`synchronized` 保护）
- Timer 线程（daemon）：每 2s 读取快照 → 启动 IO coroutine → `streamingIncrementalInProgress = true`
- IO coroutine：调用 `llmSession.generate()`（`synchronized(this)` 排队），finally 块重置 flag
- UI 回调：`runOnUiThread`（`onProgress` 是 Java 回调，不能用 `withContext`）

**运行时流程**：
```
[REC] → 录音线程每 100ms 收集 PCM → omniAudioBuffer
      → Timer 每 2s 取快照 → setAudioData → generate("<audio>stream</audio>")
        → onProgress 逐 token 更新 LIVE 卡片
      → [STOP] → stopStreamingTimer → 取完整 buffer → processIncrementalOmni(s, true)
        → LIVE 卡片替换为永久结果卡片 → returnToIdle()
```

**与计划差异**：
- 未引入 `RecordingMode` 枚举 — Omni 模式始终启用流式，无需子模式切换
- 未使用 `inner class StreamingRecorder` — 直接集成到 Activity 中更简洁
- `writeWavFile` / `writeShortLE` 方法保留（未删除，调试时可能用到）
- `onProgress` 中使用 `runOnUiThread` 而非 `withContext`（普通回调不能调用 suspend 函数）

### 4.7 Commits

```
438b072f [LLM:Feature] Qwen3-ASR Phase 2: Sliding-window pseudo-streaming for Omni mode
11d11b14 [LLM:Bugfix] Fix Phase 2 build: use runOnUiThread instead of withContext in onProgress callback
d4740636 [LLM:Bugfix] Disable keepHistory in Omni ASR mode to prevent prompt contamination
```

### 4.8 踩坑记录

| # | 问题 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | `withContext(Dispatchers.Main)` 编译失败 | `Suspension functions can only be called within coroutine body` | `GenerateProgressListener.onProgress()` 是 Java 回调接口，不是 suspend 函数 | 改用 `runOnUiThread {}` |
| 2 | Prompt 污染导致识别结果完全错误 | FINAL 输出幻觉文本（`我今天去贵阳。今年是国庆。`），与真实语音无关 | `keepHistory=true` 导致每次 `generate()` 把 `<audio>stream</audio>` 追加到对话历史。`pending_audio_` 单次消费后，旧轮次 tag 无对应 waveform，模型被污染历史混淆 | `llmSession?.setKeepHistory(false)` |

**坑 #2 详细分析**：

修复前的 prompt（14 次 generate 后）：
```
You are a helpful assistant.<audio>stream</audio>  ← 第 1 轮
You are a helpful assistant.<audio>stream</audio>  ← 第 2 轮残余
...（重复 14 次）
```

修复后的 prompt（每次 generate）：
```
You are a helpful assistant.<audio>stream</audio>  ← 仅 1 次，独立推理
```

### 4.9 实机测试数据（Kirin 9000 / TAS-AL00）

**18 秒录音，完整句子 "安装成功。现在注意观察日志中的关键变化。明天星期几？"**：

| 时刻 | 音频量 | AE 耗时 | Decode 耗时 | 增量结果 |
|------|:------|:------|:------|------|
| 2.0s | 1.8s | 172ms | 390ms | `(空)` — 前 2s 为静音 |
| 4.0s | 3.8s | 237ms | 470ms | `(空)` — 仍为静音 |
| 6.0s | 5.8s | 403ms | 735ms | `I'm not sure.` — 仅 1s 有效语音 |
| 8.0s | 7.8s | 402ms | 1.0s | `安装成功。现在注意观察。` — 开始正确 |
| 10.0s | 9.8s | 503ms | 1.3s | `很成功，现在推广是这个观点。` — 后半句幻觉 |
| 12.0s | 11.8s | 594ms | 1.4s | `安装成功。现在，主控板是启动的关键点。` |
| 16.0s | 15.8s | 884ms | 2.0s | `按照中午，现在处于观察日志中的关键点，明天星期几？` |
| **FINAL** | **17.4s** | **861ms** | **2.7s** | **`安装成功。现在注意观察日志中的关键变化。明天星期几？`** ✅ |

**关键观察**：
- **Full-Attention AE 的"自动纠错"效应**：每次推理重跑全量 fbank+AE，音频越长，AE 注意力窗口看到的上下文越完整。早期增量因音频不足产生幻觉（「主控板」「推广」），最终全量推理自然修正
- **AE 成本线性增长**：embedding length 从 23 → 218（接近 O(N)），prefill 从 0.25s → 0.98s
- **decode 速度稳定**：30-35 t/s，不受音频长度影响
- **识别准确率**：`keepHistory` 修复后，最终结果几乎完美（仅 "变化" 在早期增量中被识别为 "关键点"，全量后修正）

---

## 四-B、Phase 2.5：VAD 优化 → 误入歧途 ❌

> 实施日期：2026-06-10
> 目标：无需改动 C++ 引擎，在 Kotlin 层加入 VAD + 分段推理，解决 Phase 2 的核心缺陷
> 首次方案：纯 VAD 分段 + 单次推理（去掉了增量推理）
> **实机验证推翻**：纯分段导致上下文断裂，罕见/技术术语识别准确率大幅下降。正确方案是 Phase 2.6 的 VAD 引导扩展窗口。
> 状态：已被 Phase 2.6 取代 ❌

### 4B.1 Phase 2 缺陷回顾与 VAD 初始设想

```
Phase 2 缺陷                      VAD 解决方案              效果
──────────────────────────────────────────────────────────────────
前 6s 静音浪费 3 次推理          语音活动检测，静音跳过      省 ~800ms 累计 AE
早期短音频幻觉 ("I'm not sure")  最小语音长度 500ms         保证推理时有足够上下文
推理超过 timer 间隔 (12s+)       停顿自然分段 (每段 5-12s)  单段计算量可控
30s 硬上限 (_Slice 保护)         分段后每段独立，永不触及   长录音安全
全量 fbank 重复计算              增量 fbank 缓存 mel        省 60% fbank 时间 (后续)
```

### 4B.2 VAD 状态机设计

```
                    energy > SPEECH_THRESHOLD (持续 500ms)
     ┌──────────┐  ───────────────────────────────────>  ┌──────────┐
     │          │                                         │          │
     │ SILENCE  │                                         │ SPEAKING │
     │          │  <───────────────────────────────────   │          │
     └──────────┘    energy < SILENCE_THRESHOLD (持续800ms) └──────────┘
          │                                                     │
          │  丢弃音频 (保留1s环形缓冲)                            │  累积到 segment_buffer
          │                                                     │  每2s → 增量推理 (LIVE卡片)
          │                                                     │  检测停顿 → FINAL推理
          └─────────────────────────────────────────────────────┘
```

### 4B.3 VAD 参数

| 参数 | 值 | 说明 |
|------|:------|------|
| `SPEECH_RMS_THRESHOLD` | 0.005 | 语音能量阈值（float PCM, ±1.0 归一化后） |
| `SILENCE_RMS_THRESHOLD` | 0.003 | 静音判定阈值 |
| `MIN_SPEECH_FRAMES` | 5 (500ms) | 最短语音长度，过滤短促噪音 |
| `MAX_SILENCE_FRAMES` | 8 (800ms) | 停顿多长判定为段结束 |
| `INCREMENTAL_INTERVAL` | 2000ms | 段内增量推理间隔 |
| `RING_BUFFER_SIZE` | 1s (SAMPLE_RATE) | 静音期保留的音频上下文 |

### 4B.4 分段推理流程

```
[REC] → VAD = SILENCE, 环形缓冲保留最近1s
  │
  ├─ 语音检测 (energy > 0.005 × 500ms)
  │   → VAD = SPEAKING
  │   → segmentBuffer = [环形缓冲尾] + 新chunks
  │   → segmentCount++
  │   → 显示 LIVE 卡片
  │   → 启动增量 timer (2s)
  │
  ├─ 说话中 (每 100ms 一个 chunk)
  │   → 追加到 segmentBuffer
  │   → 每 2s: 取快照 → 增量推理 → 更新 LIVE 卡片
  │   → energy 恢复: silenceFrames = 0
  │
  ├─ 停顿 800ms → VAD = SILENCE
  │   → 取消增量 timer
  │   → 最终推理 (FINAL)
  │   → LIVE 卡片 → 永久结果卡片
  │   → segmentBuffer 清空
  │   → 准备下一段
  │   → (如有新语音 → 回到语音检测)
  │
  └─ [STOP] 按钮
      → 当前段 FINAL 推理
      → clean up, returnToIdle
      → 不自动重启
```

### 4B.5 实施过程（此方案已被推翻，仅供历史参考）⚠️

> **2026-06-11 修正**：下面记录的「纯 VAD 分段」方案在实机测试中暴露出严重精度问题。见新增章节四-C 的正确方案。

**第一阶段：VAD + 增量推理（杂交方案）**

按原计划在 `Qwen3AsrTestActivity.kt` 中实施：VAD 状态机 + 2s 增量 timer + FINAL 推理。改动 ~120 行。

**实机测试发现严重问题**：

```
17:05:20.388  Thread 19563: 增量推理 START (48000 samples) → prompt 干净
17:05:21.118  Thread 19514: FINAL 推理 START (60800 samples) → ⚠️ 增量还在跑！
17:05:21.467  Thread 19563: 增量返回结果
17:05:21.468  Thread 19514: prompt 已污染 ❌
  →  "You are a helpful assistant.<audio>stream</audio>" × 2
```

增量推理和 FINAL 推理同时运行时，两个 Bug 同时触发：
1. **Prompt 污染复活**：`keepHistory=false` 在 `generate()` 结束才清理 history。FINAL 在增量完成前构建 prompt，包含了增量的残留
2. **`pending_audio_` 竞态覆盖**：增量用 48000 samples 推理中，FINAL 的 `setAudioData(60800)` 覆盖了 `pending_audio_`

**根因分析**：Phase 2 的增量推理（盲推滑动窗口）和 Phase 2.5 的 VAD（精确语音边界检测）**设计上互相冲突**：

```
Phase 2（无 VAD）：不知道什么时候说完 → 每 2s 盲推增量 → 滑动窗口是唯一选择
Phase 2.5（有 VAD）：精确检测"一句话结束" → 一次性扔给 ASR → 增量推理多余且有害
```

> ⚠️ **2026-06-11 修正**：上面的「根因分析」结论被实机数据推翻。见下方。
>
> **被推翻的结论**：
>> Phase 2（无 VAD）：不知道什么时候说完 → 每 2s 盲推增量 → 滑动窗口是唯一选择
>> Phase 2.5（有 VAD）：精确检测"一句话结束" → 一次性扔给 ASR → 增量推理多余且有害
>
> **为什么错了**：增量推理（扩展窗口）除了解决「何时结束」外，还有一个独立且关键的价值——给 Audio Encoder 提供累积的完整音频上下文。Qwen3-ASR 的 AE 使用 Full Bidirectional Attention，18 层 Transformer 中每一帧 attend 到所有帧。分段独立推理导致段间声学上下文断裂，对罕见/技术术语的识别准确率崩溃。
>
> **实机证据**（Kirin 9000 / TAS-AL00，FP16 模型，greedy sampling）：
> - 朗读：`有 VAD 精确检测"一句话结束" 一次性扔给 ASR 增量推理多余且有害`
> - 分段 #1 FINAL: `U V D，精确检测，一句话结束`（VAD→UVD 勉强可辨认，其他词基本正确）
> - 分段 #2 FINAL: `有威力，精确检测，一句话结束，一次性扔个点`（前半句幻觉）
> - 分段 #3 FINAL: `增量记忆多于现有`（增量推理→增量记忆，多余且有害→多于现有）
> - **累计准确率 ~30%，远低于 Phase 2 FINAL 的 ~95%**（针对领域术语场景）
>
> **正确方案**：Phase 2.6 VAD 引导扩展窗口（见新增章节四-C）。

**第二阶段：简化为纯 VAD 分段 + 单次推理**（已废弃 ❌）

删除所有 Phase 2 增量推理代码（-217 行），只保留 VAD 状态机 + `processSegment()` 单次推理：

```
麦克风 100ms chunk → 能量 RMS VAD
  SILENCE → 环形缓冲保留 1s 上下文
  SPEECH 检测 (500ms) → enterSpeakingState()
    收集音频到 omniAudioBuffer
    静音检测 (800ms) → endCurrentSegment()
      → processSegment(snapshot)  ← 唯一推理路径
        → setAudioData + generate → addResultCard
```

**删除清单**：

| 删除 | 原因 |
|------|------|
| `Timer` / `TimerTask` | 不再需要滑动窗口触发 |
| `startStreamingTimer()` / `stopStreamingTimer()` | 同上 |
| `triggerIncrementalInference()` | 增量推理逻辑 |
| `processIncrementalOmni(samples, isFinal)` | 复杂的增量/FINAL 二合一方法 |
| `ensureStreamingCard()` / `updateStreamingResult()` / `finalizeStreamingResult()` | LIVE 卡片 UI |
| `streamingIncrementalInProgress` / `deferredFinalSnapshot` | 并发防护（不再需要） |

**最终文件** 1241 行 → 1024 行（-217 行）。

### 4B.6 简化后的数据流

```
┌─ Kotlin ───────────────────────────────────────────────────────────┐
│  AudioRecord chunks (100ms)                                          │
│    → energy RMS VAD (float PCM)                                      │
│                                                                       │
│  VadState.SILENCE:                                                    │
│    → omniPreSpeechRing (1s ring buffer, pre-speech context)          │
│    → speech frames++ → enterSpeakingState()                          │
│                                                                       │
│  VadState.SPEAKING:                                                   │
│    → omniAudioBuffer 累积                                             │
│    → silence frames++ → endCurrentSegment()                          │
│      → snapshot = omniAudioBuffer.toFloatArray()                     │
│      → processSegment(snapshot)                                      │
│        → lifecycleScope.launch(IO) {                                 │
│            llmSession?.setAudioData(samples, 16000)                  │
│            llmSession?.generate("<audio>stream</audio>")             │
│            addResultCard(response)                                   │
│          }                                                            │
│                                                                       │
│  [STOP] button:                                                      │
│    → isRecording = false → loop exit                                 │
│    → if SPEAKING: processSegment(remaining_snapshot) → returnToIdle │
│    → if SILENCE: returnToIdle()                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4B.7 已知限制与待修复

**跨段并发 Bug**（Code Review 发现）：

`processSegment()` 启动协程后立即返回。如果 segment #1 的推理仍在运行（Qwen3 ASR 耗时 1-2s），且 VAD 检测到 segment #2 结束（最快 ~1.3s），则两个协程并发调用 `setAudioData()` + `generate()` 到同一个 `LlmSession`：

```
T1: endCurrentSegment() for seg #1 → processSegment(snapshot1) → 协程 A 启动
T2: 录音继续（SILENCE → SPEAKING → SILENCE）
T3: endCurrentSegment() for seg #2 → processSegment(snapshot2) → 协程 B 启动  
T4: 协程 A 仍在运行 → B 覆盖 pending_audio_ + 污染 history
```

**修复**：添加 `@Volatile var segmentInferenceInProgress = false` 标志，`endCurrentSegment()` 在调用 `processSegment()` 前检查，`processSegment()` 在 finally 块中重置。详见 Code Review 输出。

### 4B.8 预期效果（vs Phase 2）

| 指标 | Phase 2 | Phase 2.5 (+VAD) |
|------|:------|:------|
| 推理次数 | 8 | **5**（跳过 3 次静音） |
| 累计 AE 计算 | ~4.0s | **~3.2s** |
| 首次有效结果 | 8.0s 时 | **6.0s 时**（语音段首次达到 2s） |
| 中间幻觉 | 有（2-6s 的 "I'm not sure" 等） | **消除**（仅在 500ms+ 语音后推理） |
| 30s 限制 | 存在 | **解除**（自然分段 < 15s/段） |
| 多段拼接 | 不支持 | **支持** |
| 12s+ 卡顿 | 有 | **减轻**（段 ≤ 12s，单次推理 < 2s） |

### 4B.9 后续扩展（Phase 2.5+）：增量 fbank 缓存

当前 VAD 分段后，段内仍每次重跑全量 fbank。可通过缓存 mel 特征进一步优化：

```cpp
// omni.cpp 伪代码 — 增量 fbank 缓存
struct FbankCache {
    VARP accumulated_mel;          // [1, 128, T_total]
    std::vector<float> overlap;    // STFT overlap (n_fft tail)
    int total_samples;
};

VARP audioProcessIncremental(VARP new_waveform, FbankCache& cache) {
    auto new_frames = stft_incremental(new_waveform, cache.overlap);
    cache.accumulated_mel = _Concat({cache.accumulated_mel, new_frames}, 2);
    cache.overlap = tail_of(new_waveform, n_fft);
    return cache.accumulated_mel;  // 传给 AE (仍全量 attention)
}
```

| 收益 | 当前 (Phase 2.5) | +增量 fbank |
|------|:------|:------|
| fbank 累计 | ~0.8s (18s 录音) | **~0.5s** |
| 实现复杂度 | - | ⭐⭐ (1-2 天, C++) |

---

## 四-C、Phase 2.6：VAD 引导扩展窗口 ✅ 当前方案

> 实施日期：2026-06-11
> 目标：纠正 Phase 2.5 的错误简化，恢复扩展窗口的累积上下文优势，同时保留 VAD 的生命周期控制
> 状态：✅ 已实施并实机验证

### 4C.1 核心设计思想

**VAD 控制录音生命周期（替代盲推 timer + 手动 STOP），扩展窗口保证推理质量。**

Phase 2 的扩展窗口本质上不是「不知道何时结束的 workaround」——它是一个独立且不可替代的精度保障机制。每次推理将 `[0..current]` 的全部累积音频发送给模型，AE 的 Full Bidirectional Attention 获得完整声学上下文。

Phase 2.5 的错误在于：VAD 解决了「何时结束」，但错误地扔掉了扩展窗口。正确的做法是用 VAD 替换盲推 timer，保留扩展窗口。

### 4C.2 数据流

```
┌─ VAD 引导的扩展窗口 ──────────────────────────────────────────┐
│                                                                  │
│  VAD = SILENCE (等待语音):                                        │
│    → 环形缓冲保留 1s 上下文                                       │
│    → 检测到语音 (500ms) → enterSpeakingState()                   │
│                                                                  │
│  VAD = SPEAKING (收集中):                                         │
│    → 从环形缓冲 + 新 chunk 开始累积到 omniAudioBuffer              │
│    → 1.5s 后首次增量：取全量快照 → 发送到 Channel                  │
│    → 每 3s 后续增量：取全量快照 → 发送到 Channel                   │
│      ↑ 扩展窗口：每次发送 [0..current] 的全部音频                  │
│    → 检查 max_speech_duration (8s 安全网)                         │
│                                                                  │
│  VAD 检测到段结束 (600ms 静音) 或 max duration:                    │
│    → 取消增量 timer                                               │
│    → 全量快照 → FINAL 推理 → 永久结果卡片 ✅                       │
│    → 清空 buffer，VAD = SILENCE，准备下一段                        │
│                                                                  │
│  [STOP] 按钮:                                                     │
│    → 取消增量 timer                                               │
│    → 当前段 FINAL 推理 → returnToIdle                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4C.3 VAD 参数

| 参数 | 值 | 说明 |
|------|:------|------|
| `OMNI_SPEECH_RMS` | 0.005 | 语音检测阈值（float PCM） |
| `OMNI_SILENCE_RMS` | 0.003 | 静音判定阈值 |
| `OMNI_MIN_SPEECH_FRAMES` | 5 (500ms) | 最短语音长度 |
| `OMNI_MAX_SILENCE_FRAMES` | **6 (600ms)** | 停顿阈值（Phase 2.5 为 8→800ms） |
| `OMNI_INCREMENTAL_FIRST_MS` | **1500ms** | 首次增量延迟（语音开始后） |
| `OMNI_INCREMENTAL_INTERVAL_MS` | **3000ms** | 后续增量间隔 |
| `OMNI_MAX_SPEECH_DURATION_MS` | **8000ms** | 最大语音时长安全网 |

### 4C.4 并发模型：Channel 串行化

增量和 FINAL 推理共享同一个 `Channel<SegmentTask>`，单 consumer 协程串行处理：

```
segmentChannel ──────────────────────────────────────────
  │                    │                    │
  │  SegmentTask(snap1,│  SegmentTask(snap2,│  SegmentTask(snap3,
  │    isFinal=false)  │    isFinal=false)  │    isFinal=true)
  │                    │                    │
  └─ consumer 串行处理 ──────────────────────────────
       processSegmentSync(task)
         → 增量: updateStreamingResult (LIVE 卡片)
         → FINAL: finalizeStreamingResult (永久卡片)
```

**并发安全保障**：
- `scheduleNextIncremental()` / `triggerIncrementalInference()` 在 timer 线程
- `endCurrentSegment()` 在 recording 线程，取消 timer 后发送 FINAL
- consumer 在 IO 协程，**所有 `setAudioData()` + `generate()` 调用严格串行**
- `endCurrentSegment()` guard 改为 `omniVadState != VadState.SPEAKING`，防止同一帧内静音+max-duration 双重触发

### 4C.5 config.json 采样参数修正

Phase 2 发现 INT8 模型对技术术语/罕见词识别极差。根因是采样参数引入随机性：

| 参数 | 修正前 | 修正后 | 理由 |
|------|:--|:--|:--|
| `sampler_type` | `"mixed"` | **`"greedy"`** | ASR 是确定性任务，非对话生成 |
| `temperature` | 0.1 | **0.0** | 罕见 token 置信度仅 ~30-70%，任何非零 temperature 引入的噪声足以偏离正确路径 |
| `top_k` | 40 | **1** | 关闭 top-k 过滤 |
| `top_p` | 0.9 | **1.0** | 关闭 nucleus sampling |
| `min_p` | 0.05 | **0.0** | 关闭最小概率过滤 |
| `n_gram` | 8 | **0** | 关闭 n-gram 去重（可能误伤低频术语） |
| `repetition_penalty` | 1.05 | **1.0** | ASR 不需要重复惩罚 |

**效果**：greedy argmax 确保同段音频每次输出一致，消除随机性对低频 token 的伤害。

### 4C.6 FP16 模型支持

`findOmniModel()` 增强为评分制：FP16 > INT8 > 其他。

```kotlin
val score = when {
    name.contains("FP16") -> 2
    name.contains("INT8") -> 1
    else -> 0
}
```

FP16 模型 (`Qwen3-ASR-MNN-FP16`) 特征：
- `tokenizer_file: "tokenizer.txt"`（非 `.mtok`）
- `quant_bit: 16`（tie_embeddings）
- llm.mnn.weight ~1.1GB（INT8 ~600MB）
- 低频 token embedding 精度损失小

### 4C.7 与 Phase 2 / Phase 2.5 的对比

| 维度 | Phase 2 (盲推) | Phase 2.5 (纯VAD) | Phase 2.6 (VAD+扩展窗口) |
|------|:---|:---|:---|
| 语音检测 | 无（盲推 timer） | RMS VAD ✅ | RMS VAD ✅ |
| 端点检测 | 手动 STOP | VAD 静音 ✅ | VAD 静音 600ms ✅ |
| 推理窗口 | 扩展窗口 ✅ | 分段独立 ❌ | 扩展窗口 ✅ |
| 静音时推理 | 浪费 3 次 | 无浪费 ✅ | 无浪费 ✅ |
| 首次有效结果 | ~6-8s | 段结束即出 | **1.5s**（首次增量） |
| 中间幻觉 | 严重（短音频） | 无增量 | 早期增量可能有（LIVE 卡片显示，FINAL 修正） |
| 最终准确率 | 高（FINAL 全量） | 低（上下文断裂）❌ | **高（FINAL 全量）✅** |
| 罕见/术语识别 | 好（全量上下文） | 差（分段断裂）❌ | **好（全量上下文）✅** |
| 并发安全 | 增量+FINAL 竞态 | 单次推理 ✅ | Channel 串行 ✅ |
| 内存占用 | 全量音频快照 | 小段快照 | 全量音频快照（每段 ≤8s） |
| AE 计算量 | O(N²) 累计 | 低 | 中等（2-3 次增量+FINAL） |

### 4C.8 已知限制

1. **仍为 Full-Attention AE 全量重算**：每次增量重跑全量 fbank+AE，音频越长开销越大。<= 8s 段内可接受（Kirin 9000: ~400-800ms AE + ~1-2s decoder）
2. **RMS VAD 噪音鲁棒性有限**：安静环境表现好，嘈杂环境可能出现误触发/漏检。可升级为 Silero VAD
3. **无跨段上下文**：段间独立推理，上一段内容不影响下一段。与 Phase 2.5 相同
4. **FloatArray 快照拷贝**：8s 录音 ~512KB/次，3-5 次增量+FINAL 总计 ~2MB。测试 App 可接受
5. **Timer 每次重建**：`scheduleNextIncremental()` 每次 cancel→new Timer。生产环境建议 `ScheduledExecutorService`

---

> **分析日期**：2026-06-10
> **分析基于**：MNN 引擎源码 (`omni.cpp`, `llm.cpp`, `audio.cpp`, `kvmeta.hpp`) + Qwen3-ASR 模型导出代码 (`qwen3_asr_model.py`, `audio.py`)
> **结论**：整体成功概率 **30-40%**，建议分 3 个子阶段渐进实施

### 5.1 目标

Audio Encoder 支持流式增量输入，LLM Decoder 在 AE 未完成时就启动 prefill，实现**边说边出**的用户体验。

### 5.2 当前引擎数据流（完整链路）

```
┌─ Kotlin ───────────────────────────────────────────────────────────┐
│  FloatArray (full PCM) → setAudioData(samples, sr)                 │
│    → generate("<audio>stream</audio>")                             │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ JNI Bridge ───────────────────────────────────────────────────────┐
│  LlmSession::Response() → processMultimodalPrompt()                │
│    → audios["stream"].waveform = pending_audio_                    │
│    → llm_->response(multimodal_prompt)                             │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ Omni Engine (omni.cpp) ───────────────────────────────────────────┐
│  tokenizer_encode() → <audio> regex → processAudioContent()        │
│    → Omni::audioProcess(waveform)           ← ★ 每次全量重跑       │
│      ├── whisper_fbank(waveform)            ← 全量 STFT            │
│      │   → mel_spectrogram (STFT + mel filterbank)                 │
│      │   → _Log / _Maximum / normalize                             │
│      │   → output: [1, 128, T]                                     │
│      │                                                              │
│      ├── mAudioModule->forward(mel)          ← ★ FULL attention     │
│      │   → Qwen3-ASR 只有 1 个输入（无 attention_mask）            │
│      │   → 18 层 Transformer 全部 bidirectional self-attention     │
│      │   → 3 层 Conv2d stride=2 → 8× 时域下采样                    │
│      │   → output: [T', 1, 1024]                                   │
│      │                                                              │
│      ├── mAudioEmbeddings.push_back(audio_embedding)               │
│      └── addPositionIds(embed_len)                                  │
│                                                                     │
│  Omni::embedding(input_ids)                                         │
│    → _Concat([txt_emb, audio_emb, txt_emb, ...])                   │
│    → mAudioEmbeddings.clear()                                       │
│                                                                     │
│  Omni::response()                                                   │
│    → generate_init()                          ← 重置 gen_seq_len   │
│    → generate(input_ids)                                            │
│      ├── forwardVec(embeds)                   ← 全量 prefill        │
│      │   └── forwardRaw() → mMeta->add = blockSize                 │
│      │       └── selectModule->onForward() → KV cache 自动扩展     │
│      └── Decode loop (逐 token, gen_seq_len 递增)                  │
└────────────────────────────────────────────────────────────────────┘
```

### 5.3 Qwen3-ASR Audio Encoder 架构（关键发现）

从 `qwen3_asr_model.py` 和 `audio.py` 源码确认：

```
Whisper-style Audio Encoder:
┌─────────────────────────────────────────────┐
│  Input: mel spectrogram [1, 128, T]         │
│                                              │
│  Conv2d(1→480, k=3, stride=2)  ─┐           │
│  Conv2d(480→480, k=3, stride=2) ─┤ 8× 下采样 │
│  Conv2d(480→480, k=3, stride=2) ─┘           │
│                                              │
│  Linear(7680→896)  ← 通道变换                │
│  + Learned Positional Embedding              │
│                                              │
│  18× Transformer Encoder Layer:              │
│    ├── LayerNorm                             │
│    ├── MultiHeadAttention                   │
│    │   ├── 14 heads × 64 dim                │
│    │   └── ★ FULL Bidirectional Attention   │
│    ├── FFN (896→3584→896, GELU)             │
│    └── LayerNorm                             │
│                                              │
│  LN → Linear(896→896) → Linear(896→1024)    │
│                                              │
│  Output: audio embedding [T', 1, 1024]       │
│  T' ≈ T / 2 (due to Conv subsampling)       │
└─────────────────────────────────────────────┘
```

#### 关键发现 1：AE 使用 Full Bidirectional Attention（非 Causal）

`Qwen3AsrAudio.export()` 的导出代码（`audio.py:520-540`）：

```python
# Qwen3-ASR 只导出 1 个输入 — 没有 attention_mask
onnx_export(model, (input_features,),
            onnx_model,
            input_names=['input_features'],    # ← 仅 1 个输入！
            output_names=['audio_embeds'],
            dynamic_axes={"input_features": {2: "size"}})
```

对比 Qwen2Audio（有 attention_mask 的 block-diagonal attention）：

```python
# Qwen2Audio: 2 个输入，支持外部 attention mask
onnx_export(model, (input_features, attention_mask),
            onnx_model,
            input_names=['input_features', 'attention_mask'],  # ← 2 个输入
            ...)
```

在 `omni.cpp:897` 中体现为两个分支：

```cpp
if (mAudioModule->getInfo()->inputNames.size() > 1) {
    // ✅ Qwen2Audio: 走这里，block-diagonal mask 控制 attention 范围
    audio_embedding = mAudioModule->onForward({input_features, attention_mask})[0];
} else {
    // ❌ Qwen3-ASR: 走这里，AE 内部做 full bidirectional attention
    // 没有外部 mask 可以限制 attention 范围
    audio_embedding = mAudioModule->forward(input_features);
}
```

**→ Qwen3-ASR 的 AE 在 18 层 Transformer 中都使用完整的双向注意力。每一帧可以 attend 到所有其他帧（过去+未来），这使得增量计算从根本上变得困难。**

#### 关键发现 2：Positional Encoding 是 Learned Absolute（非 RoPE）

Whisper 架构使用 `positional_embedding`（学习的绝对位置编码），不是 RoPE。这意味着：
- 位置编码是预训练好的固定向量
- 序列长度变化时，旧位置的 embedding 不变，但注意力范围随序列长度改变
- 新增帧 → 旧帧的 attention 分布改变 → 旧 KV cache 理论上"过期"

#### 关键发现 3：MNN KV Cache 基础设施支持增量操作

`KVMeta.hpp` 提供了增量的基础：

```cpp
struct KVMeta {
    size_t previous;  // KV cache 中已存储的 token 数
    size_t remove;    // 要移除的 token 数
    size_t add;       // 本次 forward 要新增的 token 数
    // sync() 后: previous = previous - remove + add + revertNumber
};
```

`mModulePool` 机制（`Module::clone()` 共享 `RuntimeManager` → 共享 KV cache）意味着多个 forward 调用之间 KV cache 状态可以保持。但 **增量 prefill 路径从未在 MNN 中被测试过**。

### 5.4 四项改造逐一分析

#### 5.4.1 Streaming fbank（`whisper_fbank_streaming()`）

**当前实现** (`audio.cpp:639-665`)：

```cpp
VARP whisper_fbank(waveform, sample_rate, n_mels, n_fft, hop_length, chunk_len) {
    // 对完整 waveform 做一次 STFT → mel → log → normalize
    auto mel_specgram = mel_spectrogram(waveform, &mel_params, &spec_params);
    // ... 后处理 ...
}
```

**需要改造**：
- 保存上一个 chunk 的尾部 `n_fft` 个样本作为 overlap buffer
- 新 chunk 到来时：`[overlap_buffer, new_samples]` → 计算增量 STFT
- 新 mel 帧 append 到累积的 mel spectrogram
- API 设计：
  ```cpp
  // 初始化状态
  void whisper_fbank_streaming_init(FbankState* state, int n_fft, int hop_length);
  // 增量处理，返回累积的 mel features
  VARP whisper_fbank_streaming(FbankState* state, VARP new_waveform);
  // 获取当前全部 mel features（供最终全量推理使用）
  VARP whisper_fbank_streaming_get_all(FbankState* state);
  ```

| 维度 | 评估 |
|------|------|
| 复杂度 | ⭐⭐ 中等 — STFT streaming 是成熟的信号处理技术 |
| 风险 | 🟢 低 |
| 工作量 | **3-5 天** |
| 成功概率 | **85%** |

#### 5.4.2 Incremental AE（核心难点）

计划中描述的是"Causal AE attention"，但 Qwen3-ASR 的 AE 使用 **Full Bidirectional Attention**。

**问题本质**：

```
Bidirectional Attention:
  Frame[i] 的 Q 与 ALL frames[0..N-1] 的 K 做 dot-product
  → 当新增 frames[N..N+M] 时，Frame[i] 的 attention score 会改变
  → 旧的 KV cache 缺少新帧的 K/V 信息，已"过期"
  → 不能简单地只 prefill 新帧然后用旧 KV cache
```

**三条可能路径**：

| 路径 | 方案 | 复杂度 | 精度影响 | 可行性 |
|------|------|:------|:------|:------|
| **A: Causal Mask** | 将 AE 强制改为因果注意力（下三角 mask），复用 KV cache | ⭐⭐⭐⭐⭐ | ❌ 严重 — 未针对 causal 训练的模型无法处理 | 几乎不可行 |
| **B: Sliding Window Context** | 每次只处理 `[左上下文, 新帧]` 窗口，利用注意力的自然局部性 | ⭐⭐⭐ | ⚠️ 中等 — 左上下文外的长程依赖丢失 | **推荐** |
| **C: Block-Diagonal Chunking** | 仿照 Qwen2Audio 导出时加入 attention_mask，用 block-diagonal 限制 attention | ⭐⭐⭐⭐ | ⚠️ 中低 — block 内 full attention 保留 | 较高但需重新导出模型 |

**推荐路径 B — Sliding Window Context**：

不修改 AE 模型本身，每个增量步骤：
1. 保留最近 K 秒的 mel features 作为左上下文
2. `[left_context_mel, new_mel_frames]` → AE Forward
3. 只取新帧对应的输出（丢弃左上下文的重复计算）

```
示例（K=5s 上下文，2s 增量）：
  t=2s:  AE([0 ..2s ])                 → 输出 [0 ..2s ]
  t=4s:  AE([0 ..4s ])                 → 输出 [2 ..4s ]
  t=6s:  AE([1 ..6s ])                 → 输出 [4 ..6s ]
  t=8s:  AE([3 ..8s ])                 → 输出 [6 ..8s ]
  ...
  最终:   AE([全部])                    → 验证/修正用的全量推理

计算量对比（18s 录音，2s 增量）：
  Phase 2 全量: 累积 ~4.0s AE 计算
  Sliding Window (5s): ~300ms × 8 = ~2.4s 累积
  节省 ~40% AE 计算
```

**局限性**：由于 AE 使用 absolute positional embeddings，`[1..6s]` 窗口内的位置编码是从 0 开始的（而非全局位置），这与全量推理的位置编码不一致，可能引入细微精度差异。

| 维度 | 评估 |
|------|------|
| 复杂度 | ⭐⭐⭐ 中高 |
| 风险 | 🟡 中 — 位置编码一致性和精度待验证 |
| 工作量 | **1-2 周** |
| 成功概率 | **55%** |

#### 5.4.3 增量 Embedding 注入

当前 `mAudioEmbeddings` 在每次 `embedding()` 调用后 `clear()`。

**需要改造**：

```
首次 prefill:
  embedding() → [txt_prefix, audio_emb_chunk1]
  → forwardVec() → KV cache = L1 positions
  → decode loop → 产出 tokens

增量 prefill（第 N 次，N>1）:
  不调用 generate_init()  ← 保留 KV cache
  embedding_incremental() → [audio_emb_chunkN]  仅新 tokens
  → forwardVec() → KV cache 扩展 T_N positions
  → decode loop → 继续产出 tokens
```

**关键挑战**：
- `mPositionIds` 需要从 offset 开始（已在 `addPositionIds()` 中维护）
- `embedding()` 需要支持「只构建新 chunk 的 embedding」模式
- `embedding()` 中的 `_Concat` 逻辑需适配增量场景

| 维度 | 评估 |
|------|------|
| 复杂度 | ⭐⭐⭐ 中高 |
| 风险 | 🟡 中 — position ID 连续性需精确管理 |
| 工作量 | **3-5 天** |
| 成功概率 | **65%** |

#### 5.4.4 KV Cache 增量 Prefill（Decoder 侧 — 最大风险）

**当前流程**：

```
response() → generate_init()  ← 重置 gen_seq_len=0, prefill_us=0
  → generate(input_ids)
    → forwardVec(embeds)       ← 全量 prefill
    → decode loop
```

`generate_init()` 中有 `reuse_kv()` 判断（`llm.cpp:790-794`）：
```cpp
if (!mConfig->reuse_kv()) {
    mContext->all_seq_len = 0;
    mContext->history_tokens.clear();
    mMeta->remove = mMeta->previous;
}
```

`reuse_kv()` 是为"相同前缀复用"设计的（多轮对话中重复使用 system prompt 的 KV cache），**不是**为"追加新 tokens"设计。

**需要的增量 Prefill 机制**：

```
首次 prefill (与 Phase 2 相同):
  generate_init() → 重置状态
  embedding() → [txt, audio_chunk1], len = L1
  forwardVec(embeds) → KV cache: previous = L1
  decode → 产出部分 tokens → 遇到阈值暂停

第 N 次增量 prefill (新 audio 到达):
  ✗ 不调用 generate_init()  ← ★ 保留 KV cache
  compute_ae(new_audio) → audio_emb_chunkN [T_N, 1, H]
  embedding_incremental() → [audio_emb_chunkN]
  forwardVec(new_embeds)  ← 仅 prefill T_N tokens
    → mMeta->add = T_N
    → positions 从 all_seq_len 开始
    → attention_mask 形状: [T_N, all_seq_len + T_N]  ← ★ 非标准矩形 mask
  decode → 继续产出 tokens
```

**关键挑战 1 — Attention Mask 形状**：

当前 `gen_attention_mask(seq_len)` 生成 `[seq_len, seq_len]` 的因果方阵。增量 prefill 需要：
- 新 tokens attend 到**所有**之前 tokens（不是 causal 限制在最后 seq_len 内）
- 新 tokens 内部用 causal mask（避免看到自己的未来）
- 即 mask 形状为 `[T_new, prev_len + T_new]` 且只有新 token 内部的未来位置被 mask

这是 MNN 当前 `gen_attention_mask()` 完全不支持的形状。

**关键挑战 2 — Module Selection**：

```cpp
// llm.cpp:504-514
bool inDecode = mContext->gen_seq_len > 0;
int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
if (mModulePool.find(moduleKey) == mModulePool.end()) {
    mModulePool[moduleKey].reset(Module::clone(mModule.get()));
}
selectModule = mModulePool[moduleKey];
```

每次增量 prefill 的 `seqLenKey` 不同（chunk 大小不固定），需要动态 clone 新 Module。

**关键挑战 3 — KVMeta 状态一致性**：

在 prefill → decode（gen_seq_len 递增） → prefill（gen_seq_len>0 但 chunk size > 1） → decode 的交替中，`mMeta->add`/`mMeta->remove`/`gen_seq_len`/`all_seq_len` 的交互逻辑必须精确无误。任何一个 offset 错误都会导致 KV cache 错位。

| 维度 | 评估 |
|------|------|
| 复杂度 | ⭐⭐⭐⭐⭐ 非常高 |
| 风险 | 🔴 高 — MNN 引擎从未经过此路径测试 |
| 工作量 | **2-4 周** |
| 成功概率 | **35%** |

### 5.5 综合成功概率

#### 分模块评估

| 模块 | 成功概率 | 关键风险 |
|------|:------|------|
| Streaming fbank | **85%** | 低风险，成熟信号处理技术 |
| Incremental AE (Sliding Window) | **55%** | Absolute positional embeddings + 上下文窗口外的精度损失 |
| 增量 Embedding 注入 | **65%** | MNN Express API 支持动态 _Concat，需管理 position IDs |
| KV Cache 增量 Prefill | **35%** | MNN 引擎首次走此路径，attention mask 形状不支持 |

**整体端到端可工作概率：30-40%**

> 串联风险：KV Cache 增量 Prefill 是瓶颈模块（35%），且任一模块失败都会导致整体不可用。

#### 主要风险因素

| # | 风险 | 严重性 | 说明 |
|---|------|:------|------|
| 1 | AE Full Bidirectional Attention | 🔴 阻断性 | Qwen3-ASR 的 AE 使用 full attention + absolute positional embeddings，从根本上不适用于 causal streaming。必须用 Sliding Window 折中，精度损失未知 |
| 2 | MNN 增量 Prefill 未经验证 | 🔴 阻断性 | attention mask 需要非标准矩形形状；prefill/decode 模式切换逻辑可能触发未预见的 bug |
| 3 | 精度损失不可接受 | 🟡 业务风险 | Sliding Window 可能引入 5-15% WER 增加，流式场景下是否可接受需要实测 |
| 4 | Position ID 一致性问题 | 🟡 技术风险 | 跨 chunk 的 position ID 管理可能引入隐蔽 bug，只在长语音场景暴露 |

### 5.6 推荐渐进式路线

```
Phase 3a — Streaming fbank + Sliding Window AE [概率 ~75%]
├── 修改 whisper_fbank → streaming 版本（overlap buffer）
├── AE 使用 Sliding Window Context（不修改 AE 模型）
├── 仍然每次重新 prefill Decoder（与 Phase 2 相同）
├── 收益：AE 计算量 ~40% 减少
├── 代价：首 token 延迟不变（仍需等 AE 完成）
├── 工作量：1-2 周
└── 里程碑：实机精度验证（增量 vs 全量 WER 对比）

Phase 3b — Decoder 增量 Prefill [概率 ~40%，依赖 3a 验证通过]
├── KV Cache 跨轮次复用（不调用 generate_init）
├── 改造 attention_mask 支持增量形状
├── 改造 embedding() 支持增量模式
├── 收益：首 token 延迟 ~600ms → ~200ms，边说边出
├── 工作量：3-5 周
└── 里程碑：端到端流式 ASR 可用

Phase 3c — 真正的 Causal AE [概率 ~20%，不在当前工程范围]
├── 重新训练/微调 AE 支持 causal attention
├── 或等待上游 Qwen3-ASR 发布官方 streaming 版本
└── 工作量：数月（涉及模型训练）
```

### 5.7 决策建议

| 条件 | 建议 |
|------|------|
| Phase 2 伪流式满足产品需求 | **暂缓 Phase 3**，当前方案 18s 录音 FINAL 准确率几乎完美 |
| 强需求「边说边出」+ 精度容忍度高 | **启动 Phase 3a**，拿到实机精度数据再评估 3b |
| 有模型训练资源 | **Phase 3c 才是根本解**（训练 causal AE），工程方案本质是 workaround |

> **核心结论**：Phase 3 的核心制约因素是 **Qwen3-ASR AE 的 full bidirectional attention 架构**，这是模型层面的设计选择，不是工程问题。在没有重训 AE 的前提下，任何工程方案都是精度与延迟之间的折中。Phase 2 的伪流式方案在当前阶段已经满足大多数需求。

### 5.8 关键源码索引

| 文件 | 行号 | 关键内容 |
|------|:-----|------|
| `omni.cpp:848-962` | `audioProcess(VARP)` | AE 全量处理入口，包含 fbank + AE forward + embedding 注入 |
| `omni.cpp:897-930` | AE forward 分支 | `inputNames.size()>1` 决定是否使用 attention_mask |
| `omni.cpp:1122-1236` | `embedding()` | 合并 txt+audio embeddings，`mAudioEmbeddings` 消费与清理 |
| `omni.cpp:1299-1479` | `responseInterleaved()` | 完整 prefill→decode 流程（Talker interleaved 模式） |
| `omni.cpp:1481-1496` | `response()` | 调用 `generate_init()` → `generate()` |
| `llm.cpp:640-745` | `forwardVec(VARP)` | Prefill 核心：chunking + forwardRaw → KVMeta 管理 |
| `llm.cpp:776-799` | `generate_init()` | 重置状态（`gen_seq_len=0`） — Phase 3b 需绕过 |
| `llm.cpp:836-920` | `generate()` | Prefill + Decode 完整流程 |
| `llm.cpp:496-553` | `forwardRaw()` | ModulePool 选择 + KV Cache 扩展 |
| `source/core/KVMeta.hpp` | `KVMeta` | `add`/`remove`/`previous`/`sync()` — KV Cache 底层机制 |
| `audio.cpp:639-665` | `whisper_fbank()` | 全量 STFT → mel → normalize — Phase 3a 改造目标 |
| `audio.py:440-511` | `Qwen3AsrAudio` | AE 导出逻辑，确认只有 1 个输入（无 attention_mask） |
| `audio.py:520-540` | `Qwen3AsrAudio.export()` | ONNX 导出 input_names — 确认不支持外部 mask |
| `qwen3_asr_model.py` | 全文 | AE 架构定义：3×Conv2d + 18×Transformer + Learned Positional Embeddings |
| `llmconfig.hpp:264-265` | `audio_type()` | 模型类型判断（`qwen3_asr` 标识） |

---

## 六、性能对比（实机数据：Kirin 9000 / TAS-AL00）

| 指标 | Phase 1 (直传 PCM) | Phase 2 (盲推扩展窗口) | Phase 2.5 纯VAD ❌ | Phase 2.6 VAD+扩展窗口 ✅ |
|------|:-------------------|:------------------|:----------------------|:-------------------------|
| 文件 I/O | **0ms** | **0ms** | **0ms** | **0ms** |
| VARP 创建 | ~5ms (_Const + memcpy) | ~5ms | ~5ms | ~5ms |
| fbank+AE (3s 段) | ~500ms | ~170ms (fbank) + ~220ms (AE) | ~400ms | ~400ms |
| fbank+AE (8s 段) | — | ~800ms | — | ~800ms（max duration 触发） |
| 推理次数（18s 多句） | 1 | 8（7 增量 + 1 FINAL） | 3-4（仅段结束） | **6-8**（每段 1-2 增量 + 1 FINAL） |
| 首次结果可见 | — | ~6-8s（须跳过前 6s 静音） | 段结束即出 | **~1.5s**（语音开始后首次增量） |
| Decode 速度 (FP16) | — | 30-35 t/s | 34-37 t/s | **18-22 t/s**（FP16 权重大） |
| 并发风险 | 无 | 增量+FINAL 竞态 | 低 | **无**（Channel 串行） |
| 中间结果 | — | 差（短音频幻觉） | 无 | **LIVE 卡片**（增量逐步改善） |
| 最终准确率（常见口语） | — | 几乎完美 | 好 | **几乎完美** |
| 最终准确率（罕见/术语） | — | 好（全量上下文） | **差**❌ | **好（全量上下文）✅** |
| 静音时推理浪费 | — | 有（前 6s 3 次空跑） | 无 | **无** |
| 代码量 | 6 文件，~90 行 | 1 文件，+179/-29 行 | 1 文件，1024 行 | **1 文件，~1370 行** |

---

## 七、风险与缓解

### Phase 1 & 2 & 2.6 风险（已解决或已确认）

| 风险 | 概率 | 缓解措施 |
|------|:---:|------|
| ~~`_Input` VARP 跨 RuntimeManager 失效~~ | ~~低~~ | 改用 `_Const` 完全规避 |
| ~~`_Input`/`_Const` shape 格式不匹配导致 crash~~ | ~~高→已命中~~ | 改为与 `AUDIO::load()` 完全一致的 `{N} NHWC` 格式 |
| ~~`keepHistory=true` 导致 prompt 污染~~ | ~~高→已命中~~ | `setKeepHistory(false)`，ASR 每次推理独立 |
| ~~增量+FINAL 并发导致 prompt 污染~~ | ~~高→Phase 2.5 命中~~ | Channel 串行化彻底解决（Phase 2.6） |
| ~~`endCurrentSegment()` 同一帧双重触发~~ | ~~中→Code Review 发现~~ | Guard 改为 `omniVadState != VadState.SPEAKING` |
| ~~FINAL 空响应 LIVE 卡片残留~~ | ~~低→Code Review 发现~~ | 空响应时移除 LIVE 卡片 |
| ~~纯 VAD 分段上下文断裂导致准确率崩溃~~ | ~~高→Phase 2.5 命中~~ | Phase 2.6 恢复扩展窗口 |
| `pending_audio_` 生命周期异常 | 低 | `= nullptr` 确保单次消费 + Channel 串行 |
| Omni 全量 AE 重跑（扩展窗口） | 已确认可接受 | 每段 ≤8s，AE ≤800ms；含 2-3 次增量累计 ~2.5s |
| Timer 每次重建产生 GC 压力 | 低（测试 App） | 生产环境改用 `ScheduledExecutorService` |
| RMS 能量 VAD 在噪音环境下准确率不足 | 中 | 可升级为 Silero VAD（sherpa-onnx） |
| Qwen3-ASR 0.6B 对技术术语/英文缩写的识别能力 | **高（模型能力边界）** | 非管线问题。FP16 优于 INT8，但本质是 0.6B 小模型的词汇覆盖限制。可行缓解：hot-word boosting（logit_bias）、更大模型 |

### Phase 3 风险（详见第五章分析）

| 风险 | 概率 | 严重性 | 缓解措施 |
|------|:---:|:------|------|
| AE Full Bidirectional Attention 架构不适配 streaming | **高** | 🔴 阻断性 | Sliding Window Context 折中（Phase 3a），接受精度折中 |
| MNN 引擎增量 Prefill 路径未经测试 | **高** | 🔴 阻断性 | 先在 PC 端用 `llm_demo` 验证正确性 |
| 精度损失不可接受（WER 增加 5-15%） | 中 | 🟡 业务风险 | 实机对比测试，评估是否满足产品需求 |
| Position ID 跨 chunk 一致性 bug | 中 | 🟡 技术风险 | 仅在长语音（>10s）场景暴露，需充分测试 |
| 后续 Phase 依赖上游模型更新 | **高** | 🔴 根本性 | Phase 3c 需重训 AE，当前工程方案本质是 workaround |

---

## 八、实施结果总结

```
Phase 1（已完成 ✅）:
  Step 1-2: llm_session.h/cpp（核心逻辑）
  Step 3:   llm_mnn_jni.cpp（JNI 桥）
  Step 4:   LlmSession.kt（Kotlin API）
  Step 5:   processor.cpp（key 修复）
  Step 6:   Qwen3AsrTestActivity.kt（调用方）
  → 编译 → APK 安装 → 测试验证

Phase 2（已完成 ✅）:
  → Qwen3AsrTestActivity.kt（StreamingRecorder → 直接集成）
  → UI 增量更新（onProgress → LIVE 卡片）
  → keepHistory 修复（关键精度提升）
  → 编译 → 实机测试 → 准确率验证通过
  → **关键发现**：扩展窗口的 FINAL 推理准确率几乎完美（全量音频上下文）

Phase 2.5 VAD（实施 → 误入歧途 → 废弃 ❌）:
  第一阶段 — VAD + 增量推理（杂交）:
    → Qwen3AsrTestActivity.kt: VAD 状态机 + Timer + 增量 + FINAL (+120/-40 行)
    → 实机测试发现：增量+FINAL 并发导致 prompt 污染和 pending_audio_ 覆盖
    → 尝试 deferred FINAL 修复（仍不够好）
  
  第二阶段 — 纯 VAD 分段（最终方案）:
    → 删除全部增量推理代码（-217 行）
    → processSegment(samples) 单一推理入口
    → 无 Timer、无 LIVE 卡片、无并发风险
    → 代码从 1241 行简化到 1024 行

  **⚠️ 此方案被 Phase 2.6 实机验证推翻**：
    纯分段导致段间声学上下文断裂，罕见/技术术语识别准确率崩溃(~30%)。
    错误认知："VAD 精确知道一句话什么时候结束 → 不需要滑动窗口"。
    实际上滑动窗口不仅是「何时结束」的 workaround，更是 AE Full Bidirectional Attention
    获取完整声学上下文的唯一途径。

Phase 2.6 VAD 引导扩展窗口（当前方案 ✅）:
  核心改动 — Qwen3AsrTestActivity.kt:
    → 恢复扩展窗口：增量 1.5s/3s + FINAL，全部走 segmentChannel 串行
    → VAD 作为生命周期控制器：600ms 静音/8s max 自动触发 FINAL
    → Channel<SegmentTask> 替代 Channel<FloatArray>，单 consumer 避免并发
    → LIVE 卡片 UI 恢复：ensureStreamingCard / updateStreamingResult / finalizeStreamingResult
    → Code Review 修复：endCurrentSegment 双重触发 guard、FINAL 空响应卡片移除
  
  config.json 修正:
    → sampler_type: "greedy", temperature: 0.0, top_k: 1, n_gram: 0
    → ASR 确定性输出，消除随机性对低频 token 的伤害

  FP16 模型支持:
    → findOmniModel() 评分制：FP16(2) > INT8(1) > 其他(0)
    → config.json: tokenizer_file → "tokenizer.txt"（FP16 使用 .txt 非 .mtok）
  
  实机验证（Kirin 9000 / TAS-AL00，FP16 + greedy）:
    → 常见口语：几乎完美 ✅
    → 技术术语：FP16 优于 INT8，但 0.6B 模型词汇覆盖是根本瓶颈
    → VAD 分段准确，扩展窗口正常运作
    → Channel 串行无并发问题

Phase 3a（推荐优先实施，概率 ~75%）:
  → audio.cpp: whisper_fbank_streaming()（overlap buffer + 增量 STFT）
  → omni.cpp: Sliding Window Context AE（左上下文 + 新帧 → 增量输出）
  → 仍然每次重新 prefill Decoder（与当前单次推理相同）
  → 收益验证：AE 计算量减少 ~40%
  → 精度验证：实机 WER 对比（增量 vs 全量）
  → 工作量：1-2 周 → 决策门：是否继续 3b

Phase 3b（依赖 3a 验证通过，概率 ~40%）:
  → llm.cpp: 改造 attention_mask 支持增量形状 [T_new, prev+T_new]
  → llm.cpp: generate_init() 变体（保留 KV cache 状态）
  → omni.cpp: embedding() 增量模式（仅构建新 chunk embedding）
  → omni.cpp: responseStreaming() 新入口
  → 首 token 延迟 ~600ms → ~200ms
  → 工作量：3-5 周

Phase 3c（不在当前工程范围，概率 ~20%）:
  → 重新训练/微调 AE 以支持 causal attention
  → 或等待上游 Qwen3-ASR 发布 streaming 版本
```

### Commits 总览

```
<待提交>   [LLM:Refact] Phase 2.6: VAD-guided expanding window — restore cumulative context (+250/-50 lines)
<待提交>   [LLM:Config] Switch to FP16 model with greedy sampling for deterministic ASR
<待提交>   [LLM:Bugfix] endCurrentSegment double-trigger guard + empty FINAL LIVE card removal
<待提交>   [LLM:Doc] Update OMNI streaming plan: document Phase 2.5 failure and Phase 2.6 correction
d4740636 [LLM:Bugfix] Disable keepHistory in Omni ASR mode to prevent prompt contamination
11d11b14 [LLM:Bugfix] Fix Phase 2 build: use runOnUiThread instead of withContext in onProgress callback
438b072f [LLM:Feature] Qwen3-ASR Phase 2: Sliding-window pseudo-streaming for Omni mode
dc3ebb58 [LLM:Feature] Phase 1 polish: Omni UI chip, null-safety & coroutine fixes
b16b7a8f [LLM:Bugfix] Match AUDIO::load() tensor shape: {N} NHWC instead of {1,N} NCHW
c1087120 [LLM:Bugfix] Use _Const instead of _Input for pending_audio_ VARP
c76cd069 [LLM:Feature] Qwen3-ASR Phase 1: Direct PCM path, bypass WAV file I/O
```
