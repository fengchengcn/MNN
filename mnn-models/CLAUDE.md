# MNN Models — Project Instructions

This directory manages **exported MNN model files** and **project documentation** for deploying LLM/ASR models via MNN. The primary active project is **Qwen3-ASR-0.6B** on Android via the Omni engine.

## Directory Conventions

### Naming: `<ModelName>-<Engine>-<Quantization>`

| Field | Values |
|-------|--------|
| `Engine` | `Omni` (llmexport.py) or `Legacy` (deprecated) |
| `Quantization` | `INT8`, `INT4`, `FP16`, `FP32` |

**Examples**: `Qwen3-ASR-0.6B-Omni-INT8`, `Qwen3-ASR-0.6B-Omni-FP16`

Full spec: see [[MODEL-EXPORT-GUIDE]].

### Document Organization (Obsidian-style)

```
mnn-models/
├── Qwen3-ASR-MOC.md           ← Master Map of Content (start here)
├── MODEL-EXPORT-GUIDE.md      ← Export & naming conventions
├── plans/                     ← Planning, migration, progress tracking
├── analysis/                  ← Technical deep-dives (accuracy, memory, performance)
├── archive/                   ← Deprecated docs for historical reference
├── scripts/                   ← Python validation/comparison scripts
├── Qwen3-ASR-0.6B/            ← Original HuggingFace model
└── Qwen3-ASR-0.6B-Omni-{INT8,FP16}/  ← Exported MNN models
```

Every `.md` has YAML frontmatter with `date`, `status`, `tags`, `aliases`, `related`.

## Model Export

**Always use `llmexport.py`** (except for legacy requirements):

```bash
cd ../transformers/llm/export

# INT8 (recommended for production)
python3 llmexport.py \
    --path mnn-models/Qwen3-ASR-0.6B \
    --dst_path mnn-models/Qwen3-ASR-0.6B-Omni-INT8 \
    --export mnn --mnnconvert build/MNNConvert \
    --quant_bit 8 --embed_bit 16

# FP16 (accuracy testing)
python3 llmexport.py \
    --path mnn-models/Qwen3-ASR-0.6B \
    --dst_path mnn-models/Qwen3-ASR-0.6B-Omni-FP16 \
    --export mnn --mnnconvert build/MNNConvert \
    --quant_bit 16
```

**Crucial**: `--transformer_fuse` defaults to `False` (a store_true flag) — do NOT pass it for audio encoder models, or the encoder will be incorrectly fused.

### Output Checklist

- `llm.mnn` ≈ 494K (FusedAttention) — **NOT** 790K
- `llm_config.json` exists with `tie_embeddings`
- No `embeddings_bf16.bin` or `llm_kv.mnn` (those are Legacy format)

## Android Deployment

```bash
MODEL=Qwen3-ASR-0.6B-Omni-INT8
adb shell mkdir -p /data/local/tmp/mnn_models/$MODEL
adb push mnn-models/$MODEL/* /data/local/tmp/mnn_models/$MODEL/
```

The app (`Qwen3AsrTestActivity`) scans `/data/local/tmp/mnn_models/` for directories containing `audio.mnn` + `config.json` with `is_audio=true`. Multiple models trigger a selection dialog.

## Key Technical Facts

- **Framework**: MNN Omni engine (high-level multimodal API)
- **Factory**: `Llm::createLLM(config_path)` → returns `Omni*` when `config.json` has `is_audio: true`
- **Audio path**: PCM float samples → RMS normalize (−6dBFS target) → kaldi-native-fbank → Audio Encoder → LLM Decoder
- **RMS normalization**: Fix applied 2026-06-13 in `omni.cpp` — creates `_Input` tensor (writable) instead of writing to `_Const` (silently ignored). Verify with logcat: `"Omni Audio Normalized: RMS -XX → -6.02 dBFS"`
- **Sampling**: Must use greedy (temp=0, top_k=1) for ASR — rare token confidence is only 30-70%
- **VAD**: Silero VAD with extension window (Phase 2.6) — keeps Full Bidirectional Attention context
- **Memory**: INT8 ≈ 826MB on disk, FP16 ≈ 1.5GB. Omni engine ≈ 270-470MB runtime vs 460-790MB Legacy
- **GPU**: Ruled out for mobile LLM decode — OpenCL 15-60× slower, Vulkan hangs. CPU only.

## Related Paths

| What | Where |
|------|-------|
| Export scripts | `../transformers/llm/export/llmexport.py` |
| Omni engine | `../transformers/llm/engine/src/omni.cpp` |
| Android app | `../apps/Android/MnnLlmChat/` |
| MNNConvert | `../build/MNNConvert` |
