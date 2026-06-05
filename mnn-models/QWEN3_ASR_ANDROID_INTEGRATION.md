# Qwen3-ASR Android Integration Guide

## Overview

This document describes how to integrate Qwen3-ASR into the MnnLlmChat Android app.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Android App (Kotlin)                   │
│  ┌────────────────────────────────────────────────────┐ │
│  │  VoiceChatPresenter → AsrService (audio capture)    │ │
│  │                           ↓                         │ │
│  │       Qwen3AsrEngine.kt (Kotlin wrapper)            │ │
│  └───────────────────────┬─────────────────────────────┘ │
│                          │ JNI                           │
│  ┌───────────────────────▼─────────────────────────────┐ │
│  │  qwen3_asr_jni.cpp (JNI bridge)                     │ │
│  │  qwen3_asr_engine.cpp (C++ inference engine)        │ │
│  │    ├── audio_encoder.mnn  (MNN Module)              │ │
│  │    ├── llm_kv_8bit.mnn    (MNN Module + weights)    │ │
│  │    └── embeddings_bf16.bin (embedding table)        │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files (C++ — JNI Layer)
```
apps/Android/MnnLlmChat/app/src/main/cpp/
├── qwen3_asr_engine.h          — C++ engine class header
├── qwen3_asr_engine.cpp        — C++ engine implementation
└── qwen3_asr_jni.cpp           — JNI bridge for Kotlin
```

### New Files (Kotlin)
```
apps/Android/MnnLlmChat/app/src/main/java/com/alibaba/mnnllm/android/asr/
└── Qwen3AsrEngine.kt           — Kotlin wrapper class
```

### Modified Files
```
apps/Android/MnnLlmChat/app/src/main/cpp/CMakeLists.txt
  → Added qwen3_asr_engine.cpp and qwen3_asr_jni.cpp to build
```

## Build Instructions (Windows + Android Studio)

### Prerequisites
- Android Studio with NDK installed
- Android device or emulator (arm64-v8a)

### Step 1: Cross-compile MNN Libraries

Using the existing MNN Android build script:

```bash
# From MNN project root, with ANDROID_NDK set:
cd project/android
export ANDROID_NDK=/path/to/ndk
./build_64.sh
```

This produces `project/android/build_64/lib/` containing:
- `libMNN.so`
- `libMNN_CL.so`
- `libllm.so`
- `libMNNAudio.so`
- `libMNN_Express.so`
- `libMNN_Transform.so`

**Important:** The standard `build_64.sh` doesn't include LLM/AUDIO flags. Modify it:

```bash
cmake ../../../ \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_BUILD_AUDIO=ON \
  -DMNN_ARM82=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_OPENCV=ON \
  -DMNN_IMGCODECS=ON \
  -DMNN_OPENCL=ON \
  -DMNN_BUILD_DIFFUSION=ON \
  -DNATIVE_LIBRARY_OUTPUT=. \
  -DNATIVE_INCLUDE_OUTPUT=.
make -j$(nproc)
```

### Step 2: Build Android App

Open `apps/Android/MnnLlmChat/` in Android Studio.
Sync Gradle → Build → Run on device.

### Step 3: Deploy Model Files

Copy the model directory to the device:

```bash
adb push mnn-models/Qwen3-ASR-0.6B-MNN/ /sdcard/Android/data/com.alibaba.mnnllm.android/files/models/qwen3_asr/
```

Required files:
```
qwen3_asr/
├── audio_encoder.mnn         (190 MB)
├── llm_kv_8bit.mnn           (0.5 MB)
├── llm_kv_8bit.mnn.weight    (575 MB)
├── embeddings_bf16.bin       (297 MB)
├── tokenizer.txt             (3 MB)
└── config.json               (—)
```

Total: ~1.1 GB

### Step 4: Use in App

```kotlin
// Initialize engine
val engine = Qwen3AsrEngine()
engine.init("/sdcard/.../qwen3_asr", numThreads = 4)

// During recording, push audio chunks
val pcmChunk: FloatArray = ...  // 16kHz mono PCM, normalized [-1, 1]
engine.pushAudio(pcmChunk)

// When recording stops
engine.endAudio()

// Get result
val text = engine.getResultText()
Log.i("ASR", "Transcription: $text")

// Reset for next utterance
engine.reset()

// Cleanup when done
engine.release()
```

## Integration with Voice Chat Pipeline

The existing `AsrService.kt` uses sherpa-mnn for ASR. To use Qwen3-ASR instead:

1. Add a model type check in the voice chat configuration.
2. When model type is `qwen3_asr`, use `Qwen3AsrEngine` instead of `OnlineRecognizer`.
3. The audio capture pipeline (`AsrService.kt` lines 52-55) stays the same:
   - 16kHz, 16bit, mono PCM
   - VOICE_COMMUNICATION audio source (with AEC/NS)
4. Replace the sherpa `processSamples()` loop with Qwen3-ASR pushAudio calls.
5. Note: Qwen3-ASR is not streaming (processes full utterance at once),
   unlike sherpa which gives partial results. Use endpoint detection timing
   to decide when to call `endAudio()`.

## Performance (Expected on Android)

| Component | Snapdragon 8 Gen 2 | Snapdragon 8 Elite |
|:----------|:------------------:|:------------------:|
| Audio Encoder | ~200-300ms | ~100-150ms |
| Decoder Prefill | ~200-300ms | ~100-150ms |
| Decode/step | ~20-30ms | ~15-20ms |
| **Total (3s audio)** | **~1.0-1.5s** | **~0.6-1.0s** |
| **RTF** | **0.33-0.5** | **0.2-0.33** |

## Troubleshooting

- **App crashes on model load**: Ensure `llm_kv_8bit.mnn.weight` exists
  and the model directory path is accessible.
- **Empty transcription**: The audio may not contain recognizable speech.
  Check if the model works on the same audio with asr_demo.
- **Poor accuracy**: The 8-bit quantized model has cosim=0.997 vs FP32.
  For production use, consider calibrating the quantization.
- **Tokenizer produces garbled text**: Qwen3 uses BPE tokenization.
  The basic tokenizer.txt lookup may not produce readable text.
  Use the MNN LLM engine's built-in tokenizer for proper decoding,
  or implement SentencePiece/BPE decoding on the Kotlin side.
