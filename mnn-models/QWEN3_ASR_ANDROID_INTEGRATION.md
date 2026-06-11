# Qwen3-ASR Android Integration (LEGACY — Old Engine)

> ⚠️ **This document describes the OLD engine path** (`qwen3_asr_engine.cpp` + JNI).
> The current recommended approach is the **Omni engine** (`omni.cpp` via `LlmSession`), which
> handles audio encoding, KV cache, and sampling automatically.
>
> See [[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]] for the Omni migration details.

## Architecture (Old Engine)

```
Android App (Kotlin)
  VoiceChatPresenter → Qwen3AsrEngine.kt (JNI) → qwen3_asr_engine.cpp
    ├── audio_encoder.mnn  (MNN Module)
    ├── llm_kv_8bit.mnn    (MNN Module + weights)
    └── embeddings_bf16.bin (embedding table, mmap)
```

## Key Files (Old Engine)

```
apps/Android/MnnLlmChat/app/src/main/cpp/
├── qwen3_asr_engine.h          — C++ engine class
├── qwen3_asr_engine.cpp        — C++ engine implementation
└── qwen3_asr_jni.cpp           — JNI bridge

apps/Android/MnnLlmChat/app/src/main/java/com/alibaba/mnnllm/android/asr/
└── Qwen3AsrEngine.kt           — Kotlin wrapper
```

## Model Detection (Current)

The app auto-detects the engine mode:
1. `audio.mnn` + `config.json` with `is_audio: true` → **QWEN3_OMNI** (preferred)
2. `audio_encoder.mnn` exists → **QWEN3_OLD** (this document)
3. Default → **SHERPA** (Zipformer)

## Preserved Pitfalls & Fixes

### OOM on Low-Memory Devices
- **Symptom**: lmkd kills app during model loading (RSS > 2.5 GB)
- **Root cause**: Audio Encoder + LLM Decoder loaded simultaneously, plus full embedding table in RAM
- **Fixes applied**:
  - mmap embedding table instead of full load (saves ~622 MB)
  - Serialized AE/Decoder loading (never both in memory at once)
  - `setHint(USE_CACHED_MMAP, 1)` for weight lazy loading
  - Thread count reduced to 2, `Memory_Low` mode

### Tokenizer Garbled Text
- **Symptom**: Chinese output as garbled characters
- **Root cause**: `tokenizer.txt` is MNN SentencePiece binary format, but code read it as plain text line-by-line
- **Fix**: Use `MNN::Transformer::Tokenizer::createTokenizer()` for proper BPE byte-level decoding

### SELinux WAV Write Denied
- **Symptom**: `Failed to write temp WAV (errno=13 EACCES)`
- **Root cause**: SELinux blocks untrusted_app from writing to `/data/local/tmp/`
- **Fix**: Write WAV to app cache directory (`/data/data/<pkg>/cache/`)

### LLM_SUPPORT_AUDIO Macro Missing
- **Symptom**: Model loads in 10ms, returns "no speech detected"
- **Root cause**: `LLM_SUPPORT_AUDIO` not defined in CMakeLists.txt, decoder compiled as empty stub
- **Fix**: Add `target_compile_definitions` with `LLM_SUPPORT_AUDIO`

### Multi-threaded runDecoder() SIGSEGV
- **Symptom**: Crash in `MNN::ThreadPool::enqueueInternal` on 3rd utterance
- **Root cause**: Two silence-detection threads triggered `endAudio()` within 4ms, both calling `runDecoder()` concurrently
- **Fix**: `std::mutex` + `std::try_to_lock` — non-blocking, duplicate call returns empty result

## Performance (Old Engine, Mate 30 Kirin 990)

| Metric | Value |
|--------|-------|
| AE inference | ~650-1060ms |
| Decoder prefill | ~500-740ms |
| Decode speed | ~20 tok/s (~50ms/token) |
| Subsequent utterance | ~1.5-2.0s e2e |
| RTF | ~0.25 (4× real-time) |
