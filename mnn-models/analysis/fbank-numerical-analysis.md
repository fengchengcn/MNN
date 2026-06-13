---
date: 2026-06-14
status: active
tags: [qwen3-asr, fbank, accuracy, numerical-analysis]
category: analysis
aliases: [FBank数值分析, FBank Numerical Analysis]
related: [[root-cause-analysis]], [[progress]], [[scripts/compare_pipeline_v2]]
---
# Qwen3-ASR Accuracy Gap Analysis: MNN vs sherpa-onnx

**Date**: 2026-06-14 (updated)
**Models**: Qwen3-ASR-0.6B (Whisper-style encoder-decoder ASR)
**Comparison**: MNN omni inference vs sherpa-onnx ONNX Runtime inference

---

## Summary

The MNN Qwen3-ASR deployment exhibited significantly worse recognition accuracy than sherpa-onnx. After systematic investigation and real-device A/B testing, the impact factors are ranked:

1. **🔴 Audio Pipeline (P0 — BIGGEST IMPACT):** Android `AcousticEchoCanceler` + `NoiseSuppressor` hardware effects distort speech before it reaches the model. Removing them (matching sherpa-onnx's raw MIC passthrough) yields the single largest accuracy improvement. See [[progress]] Pitfall #12.
2. **🟡 Frontend (P1):** FBank numerical alignment — preemphasis/mel-scale/STFT differences between `whisper_fbank` fallback and kaldi-native-fbank. Mitigated by `whisper_fbank_knf()` with `MNN_USE_KALDI_NATIVE_FBANK`.
3. **🟢 Backend (P2):** Minor — both use similar greedy AR decoding; decoding is NOT the primary cause.

> **Key insight (2026-06-14)**: The FBank analysis below was written before the AEC/NS impact was quantified. On real Android devices, hardware audio effects are the **dominant** accuracy killer — more impactful than fbank numerical differences combined. See §0 for details.

---

## 0. Audio Pipeline: AEC/NS Hardware Effects (🔴 P0 — DOMINANT FACTOR)

> **Verified 2026-06-14** via A/B test on real device. Removing AEC/NS alone brought MNN accuracy close to sherpa-onnx.

### The Difference

| Property | sherpa-onnx | MNN (before fix) | MNN (after fix) |
|----------|-------------|------------------|-----------------|
| `AudioSource` | `MIC` | `MIC` | `MIC` |
| `AcousticEchoCanceler` | **None** | **Enabled** | **None** ✅ |
| `NoiseSuppressor` | **None** | **Enabled** | **None** ✅ |
| Hardware AGC | None (MIC source) | None (MIC source) | None (MIC source) |

### Why AEC/NS Destroy ASR Accuracy

Android's `AudioEffect` API (`AcousticEchoCanceler`, `NoiseSuppressor`) wraps vendor-specific DSP algorithms (Qualcomm Hexagon, MediaTek Tensilica, etc.). These are tuned for **voice calls** (narrowband, 8-16kHz), not **ASR** (fullband, 16kHz+).

The effects introduce:
1. **Spectral distortion**: Aggressive noise suppression removes high-frequency consonants (`/s/`, `/f/`, `/sh/`, `/t/`) that are critical for phoneme discrimination
2. **Dynamic range compression**: AEC squashes the energy envelope, which FBank relies on for formant tracking
3. **Nonlinear artifacts**: DSP algorithms introduce harmonics and intermodulation products not present in natural speech
4. **Device-dependent quality**: Low-end phones have worse DSP implementations, making accuracy unpredictable

### Code Fix

```kotlin
// Qwen3AsrTestActivity.kt — removed from initAudioRecord():
// aec = AcousticEchoCanceler.create(...)    ← DELETED
// noiseSuppressor = NoiseSuppressor.create(...)  ← DELETED

// Now: raw MIC → PCM Short → Float → fbank (same as sherpa-onnx)
```

### Verification Method

1. BATCH mode (no VAD) + no AEC/NS vs sherpa-onnx BATCH mode
2. Same audio source, same recording, same utterance
3. Result: accuracy gap largely closed

**Code pointers**: `Qwen3AsrTestActivity.kt:initAudioRecord()` — AEC/NS code removed 2026-06-14.
`AudioRecorder.kt` (sherpa-onnx) — no AEC/NS, confirming this is the reference implementation.

---

## 1. Architecture Comparison

### sherpa-onnx Pipeline (3-stage ONNX)
```
raw PCM audio
  → kaldi-native-fbank OnlineWhisperFbank (mel spectrogram)
  → NormalizeWhisperFeatures (log10 + clamp + normalize)
  → conv_frontend.onnx (CNN subsampling)
  → encoder.onnx (Transformer audio encoder)
  → decoder.onnx (LLM autoregressive decoder)
  → text output
```

### MNN Pipeline (2-stage MNN)
```
raw PCM audio
  → MNN::AUDIO::whisper_fbank (mel spectrogram + normalize)
  → audio.mnn (audio encoder module)
  → internal LLM decoder
  → text output
```

> **Key insight**: MNN's `audio.mnn` covers sherpa-onnx's `conv_frontend.onnx` + `encoder.onnx` combined, and MNN's internal LLM decoder covers sherpa-onnx's `decoder.onnx`. The decoder stage is architecturally similar.

> **⚠️ 2026-06-14 更新**：进一步实验发现，即使 FBank 完全对齐（使用 kaldi-native-fbank），MNN AE 和 sherpa-onnx AE 的输出仍然完全不同（cosim ~0.30，65 vs 53 frames）。根因是 **conv_frontend 图结构不等价**：同一份 HF checkpoint 权重，sherpa-onnx 通过图追踪保留了原始模型的 Pad→Conv2d×3→Slice 操作链（subsampling ~6.5×），而 MNN llmexport.py 的手写 `forward()` 缺失了 Pad 和 Slice（纯 stride subsampling 8×）。详见 [[root-cause-analysis]] §"Sherpa-onnx AE vs MNN AE 对比"。

---

## 2. Feature Extraction: Detailed Diff

### 2.1 Preemphasis (🔴 CRITICAL)

| Property | sherpa-onnx (kaldi-native-fbank) | MNN `whisper_fbank` |
|----------|----------------------------------|---------------------|
| Preemphasis coefficient | **0.97** | **None** |

Preemphasis boosts high frequencies to compensate for natural vocal tract roll-off. Without it:
- High-frequency consonants (`/s/`, `/f/`, `/sh/`, `/t/`) are attenuated
- The model "sees" a spectrally tilted input it was never trained on
- After passing through N transformer layers, the error compounds

**Code pointer**: `tools/audio/source/audio.cpp:639-665` (`whisper_fbank`) — no `preemphasis` parameter and no preemphasis step. Compare `fbank()` (line 464) which does have preemphasis, and `conformer_fbank()` (line 530) which also applies it.

### 2.2 Mel Scale Formula (🔴 CRITICAL)

| Property | sherpa-onnx (kaldi-native-fbank) | MNN `whisper_fbank` |
|----------|----------------------------------|---------------------|
| Mel scale | **HTK** (`htk=true`) | **Slaney** (`htk=false`) |

```cpp
// MNN whisper_fbank — audio.cpp:647
mel_params.htk = false;  // Slaney mel scale

// kaldi-native-fbank WhisperFbank uses HTK mel scale
```

HTK formula: $m = 2595 \cdot \log_{10}(1 + f/700)$
Slaney formula: $m = f/f_{sp}$ below 1000Hz, $m = \log(f/\text{min\_log\_hz})/\text{logstep}$ above 1000Hz

The two formulas produce **different filterbank center frequencies** in the low-mid range (0-1000Hz), where most speech energy resides. This shifts the energy distribution across mel bins, producing features the model cannot interpret correctly.

**Code pointer**: `audio.cpp:311-323` (`hz_to_mel`) — two branches for HTK vs Slaney. `whisper_fbank` selects the wrong branch.

### 2.3 Mel Filterbank Normalization (🟡 HIGH)

| Property | sherpa-onnx (kaldi-native-fbank) | MNN `whisper_fbank` |
|----------|----------------------------------|---------------------|
| Mel norm | `norm=false` (area normalization by default) | `norm=true` (Slaney mode, no area norm) |

```cpp
// audio.cpp:371
float enorm = (htk && norm) ? 1.0 : 2.0 / (right - left);
```

When `htk=true && norm=false`: area normalization → `enorm = 2.0/(right-left)`
When `htk=false && norm=true`: Slaney norm → `enorm = 1.0`

kaldi-native-fbank uses `norm=false` (or the equivalent), which applies area normalization to mel filter weights.

### 2.4 Center Padding Mode (🟡 HIGH)

| Property | sherpa-onnx (kaldi-native-fbank) | MNN `whisper_fbank` |
|----------|----------------------------------|---------------------|
| Center padding | **Zero padding** (CONSTANT) | **Reflect padding** (REFLECT) |

```cpp
// MNN whisper_fbank — audio.cpp:653
spec_params.center = true;  // Reflect padding is hardcoded in spectrogram()

// spectrogram defaults (audio.cpp:386-387):
int pad_mode = REFLECT;  // Default for center=true in spectrogram()
```

Reflect padding mirrors the edge samples, while zero padding fills with zeros. This creates different boundary values for the first/last ~12 frames of STFT output.

**Code pointer**: `audio.cpp:405-407` — `_Pad(waveform, ..., REFLECT)`

### 2.5 Post-Normalization (🟢 MATCHES)

```cpp
// MNN whisper_fbank (audio.cpp:657-659):
log_specgram = _Log(_Maximum(mel_specgram, _Scalar<float>(1e-10))) / _Log(_Scalar<float>(10.0));
log_specgram = _Maximum(log_specgram, _ReduceMax(log_specgram) - _Scalar<float>(8.0));
log_specgram = (log_specgram + _Scalar<float>(4.0)) / _Scalar<float>(4.0);

// sherpa-onnx NormalizeWhisperFeatures (math.cc:113-130):
feats = feats.max(1e-10f).log10();
float max_v = feats.maxCoeff() - 8.0f;
feats = feats.max(max_v);
feats = (feats + 4.0f) / 4.0f;
```

Both are mathematically equivalent: $\text{clamp}(x, 10^{-10}) \cdot \log_{10}$, then $\max(x, \max(x)-8)$, then $(x+4)/4$.

### 2.6 Last Frame Removal (🟡 MEDIUM)

MNN removes the last mel spectrogram frame:
```cpp
// audio.cpp:655-656
mel_specgram = _Slice(mel_specgram, ..., {mel_specgram->getInfo()->dim[0] - 1, -1});
```

kaldi-native-fbank does NOT remove the last frame. This causes MNN to lose 10ms of audio information per segment.

### 2.7 STFT Implementation (🟡 MEDIUM)

- **MNN**: Uses custom `OpType_Stft` with `_Raster`/expression graph
- **kaldi-native-fbank**: Uses direct FFT via `kissfft` library

The custom STFT implementation can introduce subtle numerical differences in:
- FFT precision
- Window application order
- Power spectrum computation

---

## 3. Decoding Comparison (🟢 MINOR DIFFERENCES)

| Property | sherpa-onnx | MNN |
|----------|-------------|-----|
| Decoding method | AR greedy (temp≈0) | AR greedy (temp=0.0) |
| Beam search | No | No |
| Language model | No (hotwords only) | No |
| Top-P | 0.8 (default, inactive at temp≈0) | 1.0 (inactive) |
| Repetition penalty | No | No penalty (1.0) |
| EOS handling | Fallback: skip EOS on first token | Standard |

Both use autoregressive greedy decoding. The decoding strategy is NOT the cause of the accuracy gap.

---

## 4. Fix Priority (Updated 2026-06-14)

| Priority | Issue | Impact | Effort | Status |
|----------|-------|--------|--------|--------|
| **P0** | **AEC + NoiseSuppressor** | 🔴 **DOMINANT** — single largest accuracy killer on Android | Low | ✅ **FIXED** |
| **P1** | Missing preemphasis (0.97) | High — all high-freq consonants affected | Low | ✅ Fixed via `whisper_fbank_knf` (knf path) |
| **P1** | Wrong mel scale (Slaney → HTK) | High — all frequency bins shifted | Low | ✅ Fixed via `whisper_fbank_knf` (knf path) |
| **P2** | Mel norm mismatch | Medium | Low | ✅ Fixed via `whisper_fbank_knf` (knf path) |
| **P2** | Center padding (REFLECT → CONSTANT) | Medium — boundary frames affected | Low | ✅ Fixed via `whisper_fbank_knf` (knf path) |
| **P3** | Last frame removal | Low-Medium | Low | ✅ Fixed via `whisper_fbank_knf` (knf path) |
| **P3** | STFT numerical precision | Low | High | ⚠️ Acceptable (knf uses kissfft) |

> **Note**: P1-P3 issues only manifest when `whisper_fbank_knf()` falls back to `whisper_fbank()` (i.e., `MNN_USE_KALDI_NATIVE_FBANK` not defined). On the Android build (`project/android/build_64/`), `MNN_USE_KALDI_NATIVE_FBANK` IS defined, so these are mitigated. The P0 AEC/NS issue was the remaining gap — now fixed.

### Remaining Accuracy Factors (Post-Fix)

| Factor | Likelihood | Notes |
|--------|:----------:|-------|
| MNN `precision: "low"` for INT8 decode | 🟢 Low | Device already uses `precision: "high"` (FP32) |
| RMS normalization in omni.cpp | 🟢 Low | Verified necessary; disabling causes total failure |
| Model conversion artifacts (audio.mnn vs ONNX) | 🟢 Low | Desktop AE experiment: cosim > 0.998, 5/5 MATCH |
| **Decoder 模型差异 (llm.mnn vs llm.onnx)** | 🟡 **已验证存在** | Desktop 同输入对比：first token 5/5 match，但数值 cosim ~0.97，系统性缩放差 ~4-5× |

### 2026-06-14 Desktop Comparison: AE Isolation

**Experiment**: Isolate audio encoder vs decoder contribution to accuracy gap.

**Method**: Compute FBank from audio → Run through MNN `audio.mnn` and ONNX `audio_encoder.onnx` separately → Feed BOTH outputs into ONNX `llm.onnx` Decoder → Compare first token.

**Result**: 5/5 audio files MATCH. MNN AE output near-identical to ONNX AE (cosim > 0.998), and first token identical when both feed the same ONNX decoder.

| Audio | AE cosim | Dec cosim | First token | Verdict |
|-------|:--------:|:---------:|:-----------:|:-------:|
| 0121.wav | 0.9989 | 0.9999 | 11528 | ✅ MATCH |
| 0123.wav | 0.9994 | 0.9999 | 11528 | ✅ MATCH |
| 0125.wav | 0.9993 | 0.9999 | 11528 | ✅ MATCH |
| 0127.wav | 0.9993 | 1.0000 | 11528 | ✅ MATCH |
| 0129.wav | 0.9988 | 0.9998 | 11528 | ✅ MATCH |

**Conclusion**: Audio encoder is NOT the error source. Accuracy gap originates from the MNN Decoder (`llm.mnn`) export quality.

**Related**: [[root-cause-analysis]] for desktop experiment details and INT8 vs FP16 output comparison.

> ⚠️ **注意**：桌面端 5 样本 INT8/FP16 对比无 ground truth，无法判断孰优孰劣。手机实测 FP16 效果更好，以手机为准。

---

## 5. Recommended Fix

### Option A: Fix existing `whisper_fbank` (Recommended — low effort, high impact)

Modify `tools/audio/source/audio.cpp:639` to:
1. Accept and apply `preemphasis` parameter (default 0.97)
2. Use `mel_params.htk = true`
3. Use `mel_params.norm = false` (HTK area normalization)
4. Use `spec_params.pad_mode = CONSTANT` instead of REFLECT
5. Remove the last-frame slice

### Option B: Integrate kaldi-native-fbank (Maximum accuracy)

Add kaldi-native-fbank as a CMake dependency and expose a `whisper_fbank_knf()` wrapper that guarantees pixel-level alignment with sherpa-onnx.

---

## 6. Verification

After fixing:
1. Run the same audio through MNN `whisper_fbank` and sherpa-onnx `WhisperFbank + NormalizeWhisperFeatures`
2. Compare output tensors element-wise — they should match within 1e-5
3. Run end-to-end ASR test and compare WER with sherpa-onnx baseline

Existing comparison scripts: `compare_pipeline.py`, `compare_pipeline_v2.py`, `compare_pipeline_full.py` in this directory.
