---
date: 2026-06-15
status: active
tags: [mnn, export, guide, model, naming]
category: reference
aliases: [模型导出规范, Export Guide]
related: [[Qwen3-ASR-MOC]]
---

# MNN 模型导出规范

## 导出目录命名规范

```
<ModelName>-<Engine>-<Quantization>
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `ModelName` | 模型名称 + 参数量 | `Qwen3-ASR-0.6B` |
| `Engine` | `Omni`（llmexport.py 导出）或 `Legacy`（旧脚本导出） | `Omni` |
| `Quantization` | `INT8` / `INT4` / `FP16` / `FP32` | `INT8` |

**示例**:
- `Qwen3-ASR-0.6B-Omni-INT8` — llmexport.py 导出，INT8 量化，Omni 引擎可用
- `Qwen3-ASR-0.6B-Omni-FP16` — llmexport.py 导出，FP16，Omni 引擎可用
- `Qwen3-ASR-0.6B-Omni-INT4` — llmexport.py 导出，INT4 量化

**反例（容易混淆，避免使用）**:
- ~~`Qwen3-ASR-0.6B-MNN`~~ — 看不出引擎和量化
- ~~`Qwen3-ASR-MNN-INT8`~~ — 缺参数量，不知道哪个引擎
- ~~`Qwen3-ASR-0.6B-MNN-FP16`~~ — 看不出引擎（实际是 Legacy，但命名像 Omni）

---

## 导出脚本选择

| 脚本 | 路径 | 输出格式 | 引擎 |
|------|------|---------|------|
| **llmexport.py** ⭐ | `transformers/llm/export/llmexport.py` | Omni 统一格式 | Omni（推荐） |
| export_qwen3_asr.py | `transformers/llm/export/export_qwen3_asr.py` | 旧分离格式 | Legacy（已废弃） |

**关键区别**:
- Omni 格式：`llm.mnn` 含 FusedAttention（~494K），embeddings 嵌入 llm.mnn.weight 内，通过 `llm_config.json` 的 `tie_embeddings` 字段读取
- Legacy 格式：`llm.mnn` 含分解算子（~790K），需要独立的 `llm_kv.mnn`、`embeddings_bf16.bin`

> **规则**: 新导出全部使用 `llmexport.py`，除非有特殊的旧引擎兼容需求。

---

## 导出命令

### 基本 INT8 导出（推荐）

```bash
cd transformers/llm/export

python3 llmexport.py \
    --path /path/to/huggingface/model \
    --dst_path /path/to/output/ModelName-Omni-INT8 \
    --export mnn \
    --mnnconvert /path/to/build/MNNConvert \
    --quant_bit 8 \
    --embed_bit 16 \
    --transformer_fuse False
```

### FP16 导出

```bash
python3 llmexport.py \
    --path /path/to/huggingface/model \
    --dst_path /path/to/output/ModelName-Omni-FP16 \
    --export mnn \
    --mnnconvert /path/to/build/MNNConvert \
    --quant_bit 16 \
    --transformer_fuse False
```

### INT4 导出（低内存设备）

```bash
python3 llmexport.py \
    --path /path/to/huggingface/model \
    --dst_path /path/to/output/ModelName-Omni-INT4 \
    --export mnn \
    --mnnconvert /path/to/build/MNNConvert \
    --quant_bit 4 \
    --embed_bit 16 \
    --transformer_fuse False
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--path` | ✅ | HuggingFace 原始模型路径 |
| `--dst_path` | ✅ | 输出目录 |
| `--export mnn` | ✅ | 导出目标格式 |
| `--mnnconvert` | ✅ | MNNConvert 可执行文件路径（`build/MNNConvert`） |
| `--quant_bit` | — | 权重量化位宽：4/8/16（默认 8） |
| `--embed_bit` | — | Embedding 量化位宽，建议 16（默认与 quant_bit 相同） |
| `--transformer_fuse` | ⚠️ | **Qwen3-ASR 等 encoder-decoder 模型必须设为 False**，否则 encoder 层会被错误融合 |
| `--lm_quant_bit` | — | LLM 部分单独量化位宽（不设则与 `--quant_bit` 相同） |

---

## 输出文件清单

导出完成后，输出目录应包含以下文件：

### 通用文件（所有模型）

```
ModelName-Omni-INT8/
├── config.json            # 推理配置（供 App 读取）
├── export_args.json       # 导出参数记录（供调试用）
├── llm.mnn                # LLM 模型结构（含 FusedAttention，~494K）
├── llm.mnn.json           # LLM 模型结构 JSON 描述
├── llm.mnn.weight         # LLM 权重（含 embeddings，~604MB @ INT8）
├── llm_config.json        # 引擎内部配置（tie_embeddings 偏移量等）
└── tokenizer.txt          # Tokenizer 文件
```

### Qwen3-ASR 专用：双模型 Audio Encoder

Qwen3-ASR 使用分离的 conv_frontend + encoder 双模型（而非单文件 audio.mnn）：

```
├── conv_frontend.mnn      # Conv 前端模型结构（14 KB，chunk=100 fold-into-batch）
├── conv_frontend.mnn.weight  # Conv 前端权重（~11 MB @ INT8）
├── encoder.mnn            # Encoder 模型结构（335 KB，PE 按 chunk 重复 0..12）
├── encoder.mnn.weight     # Encoder 权重（~189 MB @ INT8）
```

**config.json 配置**：
```json
{
    "audio_model": "conv_frontend.mnn",
    "audio_encoder": "encoder.mnn"
}
```

> **历史说明**：早期导出为单文件 `audio.mnn` + `audio.mnn.weight`，因 PE 模式与训练不一致导致长音频幻觉。2026-06-15 改为双模型方案，PE 按 chunk 重复 0..12 复刻训练行为，详见 [[analysis/root-cause-analysis]] 和 [[plans/replicate-onnx-export-plan]]。

**关键检查项**:
- ✅ `llm.mnn` 大小 ~494K（非 ~790K），说明使用了 FusedAttention
- ✅ 存在 `llm_config.json`，且包含 `tie_embeddings` 字段
- ✅ 不存在 `embeddings_bf16.bin`（Omni 格式 embedding 内嵌在 weight 中）
- ✅ 不存在 `llm_kv.mnn`（Omni 格式不需要分离的 KV cache 模型）
- ✅ **Qwen3-ASR 专用**：存在 `conv_frontend.mnn` + `encoder.mnn`（双模型），不存在 `audio.mnn`（单模型已废弃）
- ✅ config.json 中 `audio_model: "conv_frontend.mnn"` 且 `audio_encoder: "encoder.mnn"`

---

## 部署到 Android 设备

```bash
# 创建目标目录
adb shell mkdir -p /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-Omni-INT8

# 推送模型文件
adb push Qwen3-ASR-0.6B-Omni-INT8/* /data/local/tmp/mnn_models/Qwen3-ASR-0.6B-Omni-INT8/

# 确认 config.json 中的路径与部署路径一致
# App 加载时会读取 config.json，从模型目录加载各文件
```

---

## 常见问题

### 1. 闪退 / SIGSEGV

**症状**: App 启动后直接崩溃，logcat 无 MNN 相关输出

**原因**: 缺少 `llm_config.json`。Omni 引擎通过 `tie_embeddings` 信息从 `llm.mnn.weight` 中读取 embeddings；缺失该文件会导致尝试加载不存在的 `embeddings_bf16.bin` 而崩溃。

**解决**: 确保 `llm_config.json` 已推送到设备。

### 2. llm.mnn 大小不对

- Omni 格式：**~494K**（含 28× FusedAttention）
- Legacy 格式：**~790K**（含分解的 MatMul 算子）

如果导出后 llm.mnn 是 790K，说明 `llmexport.py` 的 fuse 逻辑生效了，检查 `--transformer_fuse` 参数。

### 3. export_args.json 记录导出参数

每次导出会自动生成 `export_args.json`，记录完整参数。遇到问题时先检查此文件，避免凭记忆猜测导出参数。

### 4. RMS 归一化未生效

如果识别精度差、需要大声说话，检查 `omni.cpp` 中 `audioProcess()` 的 RMS 归一化是否正常工作。正常工作时 logcat 会输出：
```
Omni Audio Normalized: RMS -XX.XX dBFS -> -6.02 dBFS (gain X.Xx)
```

---

## 当前工作模型

| 目录 | 量化 | 状态 |
|------|------|------|
| `Qwen3-ASR-0.6B` | — | 原始 HF 模型 |
| `Qwen3-ASR-0.6B-Omni-INT8` | INT8 | ✅ 当前部署使用 |
