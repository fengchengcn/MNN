# Qwen3-ASR Omni 模式流式实现方案

> 创建：2026-06-10
> 最后更新：2026-06-10
> 关联文档：[[Qwen3-ASR-OMNI-PARAMETERS]] [[Qwen3-ASR-MEMORY-ANALYSIS]] [[Qwen3-ASR-STREAMING-PLAN]]（旧引擎方案）
> 状态：Phase 1 已完成 ✅ | Phase 2 已完成 ✅ | Phase 3 待定

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
| **Phase 3** | 🟢 FUTURE | 真流式增量 AE（引擎改造） | 引擎层 | 待定 |

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

## 五、Phase 3：真流式增量 AE（未来规划）

### 5.1 目标

Audio Encoder 支持流式输入，LLM Decoder 在 AE 未完成时就启动 prefill。

### 5.2 技术路线

```
当前: 完整 waveform → fbank → AE full attention → embedding → Decoder
                 ↑ 必须等录音结束

流式: chunk[0:t] → fbank[t-delta:t] → AE causal attention (仅新帧)
        → 增量 embedding → 追加到 Decoder KV Cache
          → 逐 token 流式输出
```

### 5.3 需要改造的模块

| 模块 | 文件 | 改动 |
|------|------|------|
| streaming fbank | `audio.cpp` | 新增 `whisper_fbank_streaming()` 状态机 |
| Causal AE attention | `omni.cpp` | AE 使用 causal mask 替代 full mask |
| 增量 embedding 注入 | `omni.cpp` | `mAudioEmbeddings` 支持 append |
| KV Cache 扩展 | `llm.cpp` | 支持 prefill 追加（`mModule` clone 复用） |

### 5.4 预期收益

| 指标 | Phase 2 (伪流式) | Phase 3 (真流式) |
|------|:------|:------|
| 首 token 延迟 | ~600ms | **~200ms** |
| 用户体感 | 每 2s 刷新一次 | **边说边出** |
| AE 计算量 | 每次重跑全量 | 只跑增量帧 |
| 实现复杂度 | 低 | 高（需改引擎） |

---

## 六、性能对比（实机数据：Kirin 9000）

| 指标 | Phase 1 (直传 PCM) | Phase 2 (+伪流式) |
|------|:-------------------|:------------------|
| 文件 I/O | **0ms** | **0ms** |
| VARP 创建 | ~5ms (_Const + memcpy) | ~5ms |
| fbank+AE (2s 音频) | ~500ms | ~170ms (fbank) + ~220ms (AE) |
| fbank+AE (18s 音频) | — | ~1.3s (fbank) + ~860ms (AE) |
| 首 token 延迟 | 780-1400ms | ~600ms（2s 快照） |
| Decode 速度 | — | 30-35 t/s（稳定） |
| 内存额外 | ~N×4B (PCM VARP) | +累积 buffer + LIVE 卡片 |
| 识别准确率（18s 中文） | — | **几乎完美**（修复 keepHistory 后） |
| 代码改动 | 6 文件，~90 行 | 1 文件，+179/-29 行 |

---

## 七、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|:---:|------|
| ~~`_Input` VARP 跨 RuntimeManager 失效~~ | ~~低~~ | 改用 `_Const` 完全规避 |
| ~~`_Input`/`_Const` shape 格式不匹配导致 crash~~ | ~~高→已命中~~ | 改为与 `AUDIO::load()` 完全一致的 `{N} NHWC` 格式 |
| ~~`keepHistory=true` 导致 prompt 污染~~ | ~~高→已命中~~ | `setKeepHistory(false)`，ASR 每次推理独立 |
| `pending_audio_` 生命周期异常 | 低 | `= nullptr` 确保单次消费 |
| Omni 全量 AE 重跑性能差（Phase 2） | 中→已确认 | 18s 音频 AE 耗时 ~860ms；Phase 3 增量 AE 可解决 |
| `audios` key 改为 path 后的兼容性 | 低 | Omni 引擎使用 `find(content)` 自然匹配 |
| timer tick 与 generate 并发 | 低 | `streamingIncrementalInProgress` + `synchronized(this)` 双重保护 |

---

## 八、实施顺序建议

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

Phase 3（待定）:
  → 引擎层 streaming fbank + causal AE
```

### Commits 总览

```
d4740636 [LLM:Bugfix] Disable keepHistory in Omni ASR mode to prevent prompt contamination
11d11b14 [LLM:Bugfix] Fix Phase 2 build: use runOnUiThread instead of withContext in onProgress callback
438b072f [LLM:Feature] Qwen3-ASR Phase 2: Sliding-window pseudo-streaming for Omni mode
dc3ebb58 [LLM:Feature] Phase 1 polish: Omni UI chip, null-safety & coroutine fixes
b16b7a8f [LLM:Bugfix] Match AUDIO::load() tensor shape: {N} NHWC instead of {1,N} NCHW
c1087120 [LLM:Bugfix] Use _Const instead of _Input for pending_audio_ VARP
c76cd069 [LLM:Feature] Qwen3-ASR Phase 1: Direct PCM path, bypass WAV file I/O
```
