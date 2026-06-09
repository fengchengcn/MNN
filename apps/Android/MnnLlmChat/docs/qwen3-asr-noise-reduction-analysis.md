# Qwen3-ASR Demo 降噪处理分析

> 分析日期：2026-06-09
> 分析对象：`apps/Android/MnnLlmChat` 中 Qwen3-ASR-0.6B demo 的降噪处理链路
> 关联文档：[[qwen3-asr-prompt-analysis]]

## 核心结论

Qwen3-ASR demo **有降噪处理**，但全部依赖于 **Android 平台层（DSP + AudioEffect API）**，**没有**在 C++ 原生层做任何自定义音频预处理（如 digital filter、dither、preemphasis、降噪 AI 模型等）。降噪架构从采集到推理分三层：**硬件 DSP → 软件 AudioEffect → 应用层 VAD**。

---

## 一、降噪链路总览

```
麦克风采集（模拟音频）
  │
  ▼
┌──────────────────────────────────────────────────┐
│ Layer 1: 硬件降噪（AudioSource.VOICE_COMMUNICATION） │
│ - 硬件 AEC（回声消除）                               │
│ - 硬件 NS（噪声抑制）                                │
│ - 由 Qualcomm/MediaTek DSP 自动处理                 │
└──────────────────────────────────────────────────┘
  │ 16kHz, Mono, PCM int16
  ▼
┌──────────────────────────────────────────────────┐
│ Layer 2: Android AudioEffect API                  │
│ - AcousticEchoCanceler（软件回声消除）                │
│ - NoiseSuppressor（软件噪声抑制）                     │
│ - 由 Android 框架传递到 vendor 实现                  │
└──────────────────────────────────────────────────┘
  │ short[] → FloatArray [-1.0, 1.0]
  ▼
┌──────────────────────────────────────────────────┐
│ Layer 3: 应用层 RMS-based VAD（静音门限）             │
│ - 实时 RMS 能量计算                                 │
│ - 语音/静音状态判定                                  │
│ - 端点检测（endpoint detection）                     │
│ - Mute 支持（静音时 zero-fill 缓冲区）               │
└──────────────────────────────────────────────────┘
  │ FloatArray push to engine
  ▼
┌──────────────────────────────────────────────────┐
│ C++ 引擎（无额外降噪）                               │
│ → whisper_fbank() → Audio Encoder → LLM Decoder  │
└──────────────────────────────────────────────────┘
```

---

## 二、Layer 1: AudioSource.VOICE_COMMUNICATION（硬件 DSP）

### 代码位置

三个入口点使用相同配置：

| 文件 | 函数 | 行号 |
|------|------|------|
| `Qwen3AsrTestActivity.kt` | `initAudioRecord()` | 574-576 |
| `VoiceChatPresenter.kt` | `startQwen3Record()` | 472-473 |
| `AsrService.kt` | `initMicrophone()` | 52 |

### 配置

```kotlin
// 三个文件完全相同的配置
audioRecord = AudioRecord(
    MediaRecorder.AudioSource.VOICE_COMMUNICATION,  // ← 关键
    sampleRate,                                      // 16000
    AudioFormat.CHANNEL_IN_MONO,                    // 单声道
    AudioFormat.ENCODING_PCM_16BIT,                 // 16-bit PCM
    bufferSize
)
```

### 为什么选 `VOICE_COMMUNICATION`

`AsrService.kt:51-52` 中的注释解释了意图：

> "Use VOICE_COMMUNICATION to enable hardware-level Acoustic Echo Cancellation (AEC)"

这是 Android 的 API 设计：
- `MIC` / `DEFAULT`：原始麦克风信号，无硬件处理
- `VOICE_RECOGNITION`：适度 AEC + NS，为 ASR 优化
- `VOICE_COMMUNICATION`：**最强**硬件 AEC + NS，为实时通话优化

`VOICE_COMMUNICATION` 的选择意味着代码倾向于全双工场景（ASR+TTS 同时运行），硬件 DSP 会自动做：
- **AEC**：从麦克风信号中减去当前扬声器播放的音频
- **NS**：抑制稳态背景噪声（风扇、交通、空调等）
- **AGC**（自动增益控制）：部分厂商实现在此模式下也会开启

> **局限性**：这些硬件处理取决于具体 SoC 厂商实现。高通/联发科旗舰平台效果好，中低端设备或非标准 AOSP 设备可能退化甚至无效。

---

## 三、Layer 2: Android AudioEffect API

### 3.1 AcousticEchoCanceler（软件 AEC）

```kotlin
// 三个文件完全相同的模式
try {
    if (AcousticEchoCanceler.isAvailable()) {
        AcousticEchoCanceler.create(audioRecord!!.audioSessionId).enabled = true
        Log.i(TAG, "AcousticEchoCanceler enabled")
    } else {
        Log.w(TAG, "AcousticEchoCanceler not available")
    }
} catch (_: Exception) {}
```

**设计要点**：
- 三个文件全部使用 `try-catch` 包裹，AEC 不可用时**降级而非崩溃**
- 创建后**没有保存引用**（即创建后交由系统管理，无法主动关闭）
- 三个文件的差异仅在 `VoiceChatPresenter.kt` 中把 AEC 对象保存为局部变量 `val aec = ...`（语义一致）

### 3.2 NoiseSuppressor（软件 NS）

```kotlin
// 三个文件完全相同的模式
try {
    if (NoiseSuppressor.isAvailable()) {
        NoiseSuppressor.create(audioRecord!!.audioSessionId).enabled = true
        Log.i(TAG, "NoiseSuppressor enabled")
    } else {
        Log.w(TAG, "NoiseSuppressor not available")
    }
} catch (_: Exception) {}
```

与 AEC 同样的 fail-fast-safe 模式。

### 3.3 AudioEffect 工作原理

这些 Android AudioEffect 对象附加到 `AudioRecord` 的 `audioSessionId` 上，Android 框架会在录音数据从硬件驱动到应用的路径上自动注入处理：

```
Hardware Driver → HAL → AudioEffect (AEC/NS) → AudioRecord.read() → 应用
```

**关键限制**：
1. **厂商依赖**：最终效果取决于 SoC 厂商的 audio HAL 实现
2. **无参数调优**：当前代码使用默认参数，未针对 ASR 场景调优（AEC 延迟、NS 强度等）
3. **不可观测**：代码不检查处理是否真正生效，只检查 API 可用性
4. **泄漏风险**：三个文件全部未释放 AudioEffect 对象（`AcousticEchoCanceler.create().release()` 未调用），依赖 `AudioRecord.release()` 隐式清理

---

## 四、Layer 3: 应用层 RMS-based VAD

### 4.1 Qwen3AsrTestActivity

```kotlin
// Qwen3AsrTestActivity.kt:46-51
private const val SPEECH_RMS_THRESHOLD = 400.0f   // raw int16 PCM 能量阈值
private const val SILENCE_RMS_THRESHOLD = 100.0f   // 低于此值视为静音
private const val MAX_SILENCE_CHUNKS = 15           // ~1.5s 持续静音触发端点
private const val MAX_TOTAL_CHUNKS = 300             // ~30s 最大录音时长
private const val CHUNK_INTERVAL_MS = 100            // 100ms 每帧
```

**RMS 计算（Qwen3AsrTestActivity.kt:278-285）**：
```kotlin
// RMS on raw int16 values (thresholds are in raw PCM scale)
var sumSq = 0f
for (i in 0 until ret) {
    val s = shortBuf[i].toFloat()
    sumSq += s * s
}
val rms = sqrt(sumSq / ret)
currentRms = rms
```

**VAD 状态机（Qwen3AsrTestActivity.kt:288-309）**：
```
       rms > 400          → speechDetected = true（进入语音状态）
语音状态下 rms < 100       → silenceChunkCount++（累计静音帧）
语音状态下 100 ≤ rms ≤ 400 → silenceChunkCount = 0（重置静音计数）
silenceChunkCount ≥ 15     → 触发 endpoint，停止录音，开始解码
```

### 4.2 VoiceChatPresenter

```kotlin
// VoiceChatPresenter.kt:81-84
private val silenceRmsThreshold = 100.0f   // 与 TestActivity 一致
private val speechRmsThreshold = 400.0f    // 与 TestActivity 一致
private val maxSilenceChunks = 15          // 与 TestActivity 一致
```

**RMS 计算（VoiceChatPresenter.kt:532-534）**：
```kotlin
// RMS energy for silence/speech detection
var sumSq = 0.0f
for (s in floatBuf) sumSq += s * s
val rms = kotlin.math.sqrt(sumSq / ret)
```

> **注意**：VoiceChatPresenter 的 RMS 计算使用 `floatBuf`（[-1, 1] 归一化），而阈值 `100.0` / `400.0` 是为 raw int16 设计的。**这实际上等效于 {0.003f, 0.012f} 归一化 RMS**。虽然绝对值不同，但信噪比阈值比例（4:1）保持一致。

### 4.3 AsrService（Sherpa 路径）

不使用应用层 RMS VAD，而是**依赖 Sherpa-onnx 内置的 endpoint detection**：
```kotlin
// AsrService.kt:179-192
val isEndpoint = recognizer!!.isEndpoint(stream)
// Paraformer 模型：追加 0.8s 尾部 padding
if (isEndpoint && recognizer!!.config.modelConfig.paraformer.encoder.isNotEmpty()) {
    val tailPaddings = FloatArray((0.8 * sampleRateInHz).toInt())
    stream.acceptWaveform(tailPaddings, sampleRateInHz)
    // ... 继续 decode 直到 ready 耗尽
}
```

---

## 五、Mute 功能（零噪音注入）

### 5.1 VoiceChatPresenter — Auto-Mute 回音消除模式

```kotlin
// VoiceChatPresenter.kt:524-526
// Mute handling: 当 AI 正在播放 TTS 时，强制麦克风静音
if (isMuted) {
    shortBuf.fill(0)  // zero-fill 避免反馈回声
}
```

Auto-Mute 工作流程（`VoiceChatPresenter.kt:222-223`）：
```
AI 开始处理/说话 → muteMicrophone(true)  → 录音信号全零
AI 说完话         → muteMicrophone(false) → 恢复录音
用户手动打断      → muteMicrophone(false) → 恢复录音
```

### 5.2 AsrService（Sherpa 路径）

```kotlin
// AsrService.kt:157-158
if (isMuted.get()) {
    buffer.fill(0)  // zero-fill
}
```

### 5.3 Qwen3AsrTestActivity

**不支持 mute 功能** — 独立测试页面不涉及 TTS 播放，无需 mute。

---

## 六、C++ 引擎层：无额外降噪

### 6.1 fbank 特征提取参数

```cpp
// qwen3_asr_engine.cpp:413（startDecode）/ 702-705（runDecoder）
auto feat = MNN::AUDIO::whisper_fbank(wf);
```

`whisper_fbank` 的函数签名（`audio.hpp:163-164`）：
```cpp
VARP whisper_fbank(VARP waveform, int sample_rate = 16000, int n_mels = 128,
                   int n_fft = 400, int hop_length = 160, int chunk_len = 0);
```

调用时使用**全部默认参数**：
- `sample_rate = 16000`
- `n_mels = 128`
- `n_fft = 400`
- `hop_length = 160`
- `chunk_len = 0`

### 6.2 没有启用的降噪能力

`audio.hpp` 中 `fbank()` 函数额外支持两个降噪相关参数，但 `whisper_fbank` 不支持：

| 参数 | 说明 | Qwen3-ASR 实际使用 |
|------|------|-------------------|
| `dither` | 添加微小白噪声防止量化误差 | **不支持**（仅 `fbank()` 有） |
| `preemphasis` | 预加重高频补偿（默认 0.97） | **不支持**（仅 `fbank()` 有） |

> `whisper_fbank` 是 Whisper 模型专用的 fbank 实现，移除了 dither/preemphasis 以匹配 Whisper 训练时使用的 Log-Mel 参数。

---

## 七、全场景对照表

| 降噪手段 | Qwen3AsrTestActivity | VoiceChatPresenter (Qwen3) | AsrService (Sherpa) |
|----------|:---:|:---:|:---:|
| AudioSource.VOICE_COMMUNICATION | ✅ | ✅ | ✅ |
| AcousticEchoCanceler | ✅ | ✅ | ✅ |
| NoiseSuppressor | ✅ | ✅ | ✅ |
| RMS-based VAD | ✅ (静音门限) | ✅ (静音门限) | ❌ (Sherpa 内置) |
| Sherpa endpoint detection | N/A | N/A | ✅ |
| Mute (zero-fill) | ❌ | ✅ | ✅ |
| 硬件 DSP AEC/NS | ✅ | ✅ | ✅ |
| C++ 层降噪处理 | ❌ | ❌ | ❌ |
| fbank dither | ❌ (whisper_fbank 不支持) | ❌ (whisper_fbank 不支持) | N/A (Sherpa 内置) |
| fbank preemphasis | ❌ (whisper_fbank 不支持) | ❌ (whisper_fbank 不支持) | N/A (Sherpa 内置) |

---

## 八、已知问题与改进建议

### 8.1 已知问题

| 问题 | 影响 | 位置 | 状态 |
|------|------|------|------|
| ~~**R-W1**: AudioEffect 未释放~~ | ~~3个文件创建了 AEC/NS 但从未调用 `.release()`~~ | ~~`Qwen3AsrTestActivity.kt:582-583`, `VoiceChatPresenter.kt:482-484`, `AsrService.kt:120-121`~~ | ✅ 已修复 (2026-06-09) |
| ~~**R-W2**: RMS 阈值单位不一致~~ | ~~VoiceChatPresenter 用归一化 float 计算 RMS，阈值与 raw int16 不匹配~~ | ~~`VoiceChatPresenter.kt:533-534`~~ | ✅ 已修复 (2026-06-09) |
| **R-W3**: AudioEffect 无效果监控 | 代码检查 `isAvailable()` 但不验证实际处理效果 — 部分设备返回 true 但降噪失效 | 所有 3 个文件 | |
| **R-W3**: AudioEffect 无效果监控 | 代码检查 `isAvailable()` 但不验证实际处理效果 — 部分设备返回 true 但降噪失效 | 所有 3 个文件 |
| **R-W4**: 无参数调优 | AEC 延迟、NS 强度使用厂商默认值，未针对 ASR 语音质量（非通话可懂度）调优 | 所有 3 个文件 |
| **R-W5**: 无算法级降噪储备 | C++ 层完全无降噪处理，所有降噪绑死在 Android 平台 API 上，无法跨平台复用 | `qwen3_asr_engine.cpp` |

### 8.2 改进建议

| 建议 | 优先级 | 说明 | 状态 |
|------|--------|------|------|
| ~~**S1**: 修复 AudioEffect 泄漏~~ | ~~高~~ | ~~在 `stopAudioHardware()` / `stopQwen3Record()` 中显式 release AEC/NS 对象~~ | ✅ 已实施 (2026-06-09) |
| ~~**S2**: 统一 RMS 计算~~ | ~~中~~ | ~~两处 VAD 使用相同的 PCM scale（raw int16）计算 RMS~~ | ✅ 已实施 (2026-06-09) |
| **S3**: 增加处理效果自检 | 低 | 对比 AEC/NS 开关前后的信噪比变化，记录到日志 |
| **S4**: 探索 RNNoise / SpeexDSP | 低 | 如需要跨平台（iOS/嵌入式）或低端设备降噪，可评估轻量 C 库集成到引擎层 |
| **S5**: 添加 VAD 参数配置 | 低 | 将阈值（400/100/15）改为可调参数，适应不同使用环境（安静办公室 vs 嘈杂户外） |
