---
date: 2026-06-13
status: active
tags: [moc, index, qwen3-asr]
category: index
aliases: [MOC, 目录, 导航, Home]
---

# Qwen3-ASR MNN 项目导航

> 🗺️ 项目文档地图（Map of Content）。所有文档均通过 `[[wikilink]]` 互相引用。

---

## 📦 模型文件

| 目录 | 引擎 | 量化 | 大小 | 状态 |
|------|------|------|------|------|
| [[Qwen3-ASR-0.6B]] | — | — | 1.8G | 原始 HF 模型 |
| [[Qwen3-ASR-0.6B-Omni-INT8]] | Omni | INT8 | 826M | ✅ 生产部署 |
| [[Qwen3-ASR-0.6B-Omni-FP16]] | Omni | FP16 | 1.5G | 🧪 精度测试 |

---

## 📋 规划与进度

| 文档 | 说明 | 状态 |
|------|------|------|
| [[plans/preliminary-research]] | 初步调研（2026-06-04） | ✅ completed |
| [[plans/llmexport-migration]] | llmexport.py 迁移计划（WP1-WP6） | ✅ completed |
| [[plans/omni-streaming]] | Omni 流式推理方案（Phase 1-3） | 🔄 active |
| [[plans/sherpa-ae-mnn-integration]] | Sherpa AE MNN 集成方案（conv_frontend + encoder） | ✅ implemented |
| [[plans/progress]] | 项目总进度 & 里程碑 & 踩坑记录 | 🔄 active |

---

## 🔬 技术分析

| 文档 | 说明 | 状态 |
|------|------|------|
| [[analysis/root-cause-analysis]] | 识别精度根因分析 | 🔄 active |
| [[analysis/fbank-numerical-analysis]] | FBank 特征提取数值差异 + AEC/NS 根因分析 | 🔄 active |
| [[analysis/android-memory]] | Android 真机内存 & GPU 实测 | ✅ completed |
| [[analysis/memory-model]] | Omni vs 旧引擎内存模型理论分析 | ✅ completed |
| [[analysis/omni-parameters]] | Omni 推理参数配置 & greedy 采样 | 🔄 active |
| [[analysis/export-pipeline-analysis]] | 导出链路分析：双模型 AE vs llmexport 正道 | 🔄 active |

---

## 🗄️ 归档（Legacy）

| 文档 | 说明 | 状态 |
|------|------|------|
| [[archive/android-integration-legacy]] | 旧引擎 Android 集成文档 | 🗃️ archived |
| [[archive/streaming-legacy]] | 旧引擎流式优化方案 | 🗃️ archived |

---

## 📐 参考

| 文档 | 说明 |
|------|------|
| [[MODEL-EXPORT-GUIDE]] | 模型导出命名规范 & 操作指南 |

---

## 🔧 验证脚本

| 脚本 | 用途 |
|------|------|
| `scripts/compare_ae_onnx_vs_mnn.py` | Audio Encoder ONNX vs MNN 对比 |
| `scripts/compare_onnx_vs_mnn.py` | Decoder ONNX vs MNN 对比 |
| `scripts/compare_pipeline.py` | 隔离 AE 误差对 decoder 首 token 影响 |
| `scripts/compare_pipeline_full.py` | 完整 prompt 构造 + 全链路对比 |
| `scripts/compare_pipeline_v2.py` | FBank→AE→Decoder 全链路四组合对比 |
| `scripts/test_ae_end_to_end.py` | ONNX AE 端到端验证（产出参考输出） |
| `scripts/test_full_pipeline.py` | 最完整：动态编译 C++ 做全链路对比 |
| `/tmp/run_comparison.py` | **2026-06-14 桌面端 AE 隔离实验**（macOS 适配版） |

---

## 🏷️ 标签索引

- `#qwen3-asr` — 所有 Qwen3-ASR 相关文档
- `#omni` — Omni 引擎相关
- `#fbank` — 特征提取相关
- `#memory` — 内存相关
- `#streaming` — 流式推理相关
- `#accuracy` — 精度分析相关
- `#legacy` — 已废弃文档

---

## 📊 项目时间线

```
06-04  初步调研 → 确认 MNN + llmexport.py 方案
06-05  ONNX 导出成功
06-06  KV cache + INT8 量化
06-07  Android E2E 跑通（旧引擎）
06-08  流式优化 + FP16（旧引擎）
06-09  启动 llmexport.py 迁移
06-10  Omni 引擎导出 + Android Omni 集成
06-11  WP1-WP6 完成，旧引擎移除
06-12  FBank 数值差异深度分析
06-13  模型目录整理 + 导出规范 + FP16 导出
06-14  **AEC/NS 移除 — 精度最大杀手确认并修复**
06-14  桌面端 AE 隔离实验：MNN AE ≈ ONNX AE（cosim > 0.998）
06-14  **🔴 根因：AE 架构不等价**（sherpa-onnx vs llmexport.py）
06-14  **方案三：MNNConvert 双模型 AE（conv_frontend.mnn + encoder.mnn）**
06-14  Step 3+4 验证通过：Express API 串联 cosim=1.0/0.997，Decoder first token 完全一致
06-14  **双模型 AE 实机部署完成**：手机 logcat 确认端到端运行，AE 耗时 ~1.4s
06-14  **发现官方 modeling 代码**：`github.com/QwenLM/Qwen3-ASR` 含完整 `modeling_qwen3_asr.py`，手写版漏掉 Chunk/Pad/Slice/Window。详见 [[analysis/export-pipeline-analysis]]
```
