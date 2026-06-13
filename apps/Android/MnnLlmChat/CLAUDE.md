# MnnLlmChat — Android Demo App Instructions

Android chat app for MNN LLM/Diffusion/ASR inference. Package: `com.alibaba.mnnllm.android`.

## Quick Start

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
cd apps/Android/MnnLlmChat
./gradlew assembleDebug
adb install -r app/build/outputs/apk/standard/debug/app-standard-debug.apk
```

## Architecture

```
Kotlin UI Layer
  ├── Qwen3AsrTestActivity  ← LAUNCHER (ASR test, direct Omni engine)
  ├── MainActivity           ← Bottom tabs: Models / Market / Benchmark
  ├── ChatActivity           ← Chat interface with model switcher
  └── ChatPresenter          ← Session lifecycle, model switching
        ↓
LlmSession (Kotlin)          ← JNI declarations + model config merging
        ↓  JNI (llm_mnn_jni.cpp)
mls::LlmSession (C++)        ← Wraps Llm::createLLM(config_path)
        ↓
MNN Omni Engine              ← Audio process → Encoder → Decoder
```

## Entry Points

### Qwen3AsrTestActivity (LAUNCHER)

The primary entry point for Qwen3-ASR testing. On start:
1. Scans `/data/local/tmp/mnn_models/` for Omni-compatible models (`audio.mnn` + `config.json` with `is_audio: true`)
2. If multiple models found → shows selection dialog ("选择模型精度")
3. Loads selected model, writes ASR-optimized config (greedy sampling), initializes Silero VAD
4. Two modes: **OMNI-VAD** (VAD-segmented streaming) and **OMNI-BATCH** (full recording → single decode)

Key files:
- `app/src/main/java/com/alibaba/mnnllm/android/asr/Qwen3AsrTestActivity.kt`
- `app/src/main/res/layout/activity_qwen3_asr_test.xml`

### ChatActivity / MainActivity

Full chat interface with model list, marketplace, benchmark, settings. Model switching via `SelectModelFragment` → `ChatPresenter.switchModel()`.

## Model Loading Flow

```
User taps model
  → ChatRouter.startRun(modelId, destDir, sessionId)
  → ChatActivity.onCreate()
    → ChatPresenter.createSession()
      → DefaultLlmRuntimeController.ensureSession()  // Singleton, ONE active session
        → ChatService.createSession() → LlmSession(configPath, history)
    → ChatPresenter.load()
      → LlmSession.load()
        → ModelConfig.loadMergedConfig()     // base + custom_config.json merge
        → initNative(configPath, history, mergedConfig, extraConfig)
          → JNI → mls::LlmSession(config_path, merged, extra)
            → Llm::createLLM(config_path)   // Reads config.json, creates Omni/Llm
            → llm_->load()                   // Actual weight loading
```

## Model Discovery

`LocalModelsProvider.getLocalModels()` scans `/data/local/tmp/mnn_models/`:
- Each subdirectory with `config.json` → registered as `local/<path>` model
- Directory name becomes display name in model list

## Model Switching (In-Place)

`ChatPresenter.switchModel()` properly releases old model before loading new:
1. `destroyCurrentSession()` → `releaseSession()` → JNI `releaseNative()`
2. `ensureSession(forceReload=true)` → new session from scratch
3. `DefaultLlmRuntimeController` is singleton — only one session active at a time

## JNI Layer

`llm_mnn_jni.cpp` bridges Kotlin ↔ C++:
- `initNative(configPath, history, mergedConfig, extraConfig)` → creates `mls::LlmSession`, calls `Load()`
- `submitNative(ptr, prompt, keepHistory, listener)` → `llm_->response()` with streaming
- `setAudioDataNative(ptr, samples, sampleRate)` → `llmSession->SetPendingAudio()` — PCM float data directly, bypasses WAV I/O

## C++ Native Session

`llm_session.cpp` — `mls::LlmSession`:
- Constructor accepts model config JSON path + merged config
- `Load()` calls `Llm::createLLM(config_path)` → `set_config()` → `load()`
- `Response()` handles token-by-token streaming via `Utf8StreamProcessor`
- `SetPendingAudio()` passes PCM float samples to engine via `_Const` VARP

## Key Config

`config.json` fields relevant to ASR:
```json
{
    "is_audio": true,
    "audio_type": "qwen3_asr",
    "audio_model": "audio.mnn",
    "sampler_type": "greedy",
    "temperature": 0.0,
    "top_k": 1,
    "max_new_tokens": 256
}
```

Custom overrides stored in `<filesDir>/configs/<modelId>/custom_config.json`.

## Common Pitfalls

1. **SIGSEGV on load**: Missing `llm_config.json` on device → `tie_embeddings` info missing, tries to load non-existent `embeddings_bf16.bin`
2. **Wrong model format**: `llm.mnn` = 790K → Legacy format, NOT Omni compatible. Must be ~494K.
3. **Model auto-selects FP16**: `findBestOmniModel()` used scoring (FP16=2, INT8=1). Replaced with selection dialog.
4. **RMS normalization silent fail**: `_Const` tensor `writeMap()` returns temp buffer, doesn't persist. Fixed in `omni.cpp`.
5. **Gradle UP-TO-DATE after manual .so copy**: Must `./gradlew clean assembleDebug`

## Build Variants

- `standardDebug` — Standard build with MNN CPU backend
- `googleplayDebug` — Google Play variant with additional backends

## Related Paths

| What | Where |
|------|-------|
| MNN libs | `app/src/main/jniLibs/` (untracked, copied from build) |
| Native source | `app/src/main/cpp/` |
| Model configs | `app/src/main/assets/` |
| MNN models (Mac) | `../../../../mnn-models/` |
| MNN models (phone) | `/data/local/tmp/mnn_models/` |
| Engine source | `../../../../transformers/llm/engine/` |
