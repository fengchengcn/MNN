---
date: 2026-06-04
status: completed
tags: [qwen3-asr, research, mnn, migration]
category: plan
aliases: [初步调研, Preliminary Research]
related: [[llmexport-migration]], [[progress]]
---
# Qwen3-ASR-0.6B MNN 迁移：初步调研摘要

> 日期：2026-06-04 | 状态：**已完成，仅供历史参考**
> 所有决策已执行，模型已通过 llmexport.py 导出并集成到 Omni 引擎。
> 详见：[[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]] [[Qwen3-ASR-MNN-PROGRESS]]

## 模型架构要点

```
Qwen3ASRForConditionalGeneration
  └── thinker
        ├── audio_tower (Whisper-style: 3×Conv2d + 18×Transformer, ~180M)
        ├── model (Qwen3 Decoder: 28层, hidden=1024, GQA 16Q/8KV, ~600M)
        ├── embed_tokens (vocab=151936)
        └── lm_head (tied with embed_tokens)
```

- Audio Encoder: Full Bidirectional Attention, Learned Absolute Positional Encoding
- LLM Decoder: 标准 Qwen3 架构（RMSNorm, GQA, QK-Norm, SwiGLU, RoPE）
- 推理流程: 音频 → Mel fbank(128维) → AE → prefix embeddings → Decoder 自回归

## 关键决策记录

| # | 决策 | 结论 | 依据 |
|---|------|------|------|
| 1 | 推理框架选型 | **MNN** (非 sherpa-onnx) | sherpa-onnx 无 LLM 推理能力，Qwen3-ASR 85% 计算在 Decoder |
| 2 | 导出路径 | **llmexport.py** (非 ONNX→MNNConvert) | llmexport.py 支持 FusedAttention 融合，解锁 GPU 加速 |
| 3 | 跨 App 架构 | 独立进程 + AIDL IPC | 节省内存，单实例共享 |
| 4 | 录音策略 | App 端录音 → PCM IPC | Android 10+ 后台录音限制 |
| 5 | 量化策略 | INT8 用于生产，4-bit 精度不足 | 8-bit cosim=0.997 vs FP32；4-bit cosim=0.527 |
| 6 | GPU 后端 | **CPU 优先**，GPU 待 FusedAttention 后评估 | Mali OpenCL/Vulkan 实测性能远不如 CPU（Decode memory-bound） |

## 实际 vs 预估

| 维度 | 原预估 | 实际结果 |
|------|--------|----------|
| 总工作量 | 7-10 天 | ~4-5 天（Omni 引擎已有完整音频管线） |
| inputs_embeds 注入 | 需验证 | ✅ 源码确认已支持 |
| Decoder 架构兼容 | 需适配 | ✅ 标准 Qwen3，仅改权重前缀 |
| Audio Encoder 导出 | 需从零开发 | ✅ Gemma4Audio 模板充分复用 |
| CPU 推理 RTF | 预估 0.3-0.47 | ✅ 实测 ~0.25（4× 实时） |

## 最初工作量估算（已被实际推翻）

原估 7-10 天。实际因 Omni 引擎已有 `whisper_fbank()`、`mAudioModule`、embedding 注入机制，工程量大幅降低。
最终通过 llmexport.py 迁移（WP1-WP6）完成，实际约 4-5 天。
