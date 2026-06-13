---
date: 2026-06-12
status: active
tags: [qwen3-asr, fbank, accuracy, numerical-analysis]
category: analysis
aliases: [FBank数值分析, FBank Numerical Analysis]
related: [[root-cause-analysis]], [[scripts/compare_pipeline_v2]]
---
# Qwen3-ASR Accuracy Gap Analysis: MNN vs sherpa-onnx

**Date**: 2026-06-12
**Models**: Qwen3-ASR-0.6B (Whisper-style encoder-decoder ASR)
**Comparison**: MNN omni inference vs sherpa-onnx ONNX Runtime inference

---

## Summary

The MNN Qwen3-ASR deployment exhibits poor recognition accuracy compared to sherpa-onnx. After detailed code analysis, the root causes are:

1. **Frontend (P0):** Feature extraction numerical misalignment — preemphasis missing, wrong mel scale
2. **Frontend (P1):** STFT/Window/Padding implementation differences
3. **Backend (P2):** Minor — both use similar greedy AR decoding; decoding is NOT the primary cause

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

> **Key insight**: MNN's `audio.mnn` covers sherpa-onnx's `conv_frontend.onnx` + `encoder.onnx` combined, and MNN's internal LLM decoder covers sherpa-onnx's `decoder.onnx`. The decoder stage is architecturally similar. **The accuracy gap originates entirely from the feature extraction step BEFORE the model.**

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

## 4. Fix Priority

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | Missing preemphasis (0.97) | High — all high-freq consonants affected | Low |
| **P0** | Wrong mel scale (Slaney → HTK) | High — all frequency bins shifted | Low |
| **P1** | Mel norm mismatch | Medium | Low |
| **P1** | Center padding (REFLECT → CONSTANT) | Medium — boundary frames affected | Low |
| **P2** | Last frame removal | Low-Medium | Low |
| **P2** | STFT numerical precision | Low | High |

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
