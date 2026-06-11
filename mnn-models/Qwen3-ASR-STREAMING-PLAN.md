# Qwen3-ASR 旧引擎流式优化（LEGACY）

> ⚠️ **此文档描述旧引擎** (`qwen3_asr_engine.cpp`) 的流式优化。
> 当前推荐路径是 Omni 引擎（`omni.cpp`），参见 [[Qwen3-ASR-OMNI-STREAMING-PLAN]]。
>
> 状态：Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ (全部完成于 2026-06-08)

## 三阶段优化总结

| Phase | 内容 | 效果 |
|-------|------|------|
| **1** | AE 常驻 + WAV 移除 + warmup 移位 | 后续 utterance 从 ~5s → ~1.3s (3.2x) |
| **2** | 增量解码 (startDecode/decodeStep) | 逐 token 流式输出，~50ms/token |
| **3** | CPU 微调 (4线程 + FP16 + Power_High) | ~20 tok/s，FP16 精度无损 |

## 实测性能（Mate 30, Kirin 990, 8GB）

| 指标 | 首次 | 后续 |
|------|------|------|
| AE 推理 | ~1062ms | ~657-897ms |
| Prefill | ~741ms | ~495-663ms |
| Decode | ~592ms (11t) | ~325-344ms (7t) |
| 吞吐 | 18.6 tok/s | 20.3-21.5 tok/s |
| 端到端 | ~5.7s | ~1.5-2.0s |

## 保留的重要踩坑

### 多线程并发 runDecoder() → SIGSEGV
- **现象**: 第 3 次 utterance 时在 `MNN::ThreadPool::enqueueInternal` 崩溃
- **根因**: Kotlin 层 silence-detection 两个不同线程在 4ms 内各自检测到 endpoint，同时调用 `nativeEndAudio()` → `runDecoder()`
- **修复**: `std::mutex` + `std::try_to_lock`，重复调用直接返回空结果

### Embedding mmap 内存优化
- 原始方案：全量加载 151936×1024 float32 embedding 表 → 622 MB
- 优化方案：mmap bf16 文件，按需缺页读取（ASR 全程仅访问 ~113 个不同 token < 0.1%）
- 节省：~622 MB → 0 MB 常驻内存

### Decoder 权重 mmap
通过 `RuntimeManager::setHint()` 启用：
```cpp
m_rt->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
m_rt->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);   // 延迟分配
m_rt->setHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0); // 最小化 winograd
m_rt->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1); // 动态量化
```

## 向 Omni 引擎迁移

旧引擎的手写 decode loop 已被 Omni 引擎取代。迁移收益：
- 内置采样策略（top-k/top-p/temperature）
- FusedAttention 融合（解锁 GPU 加速）
- DiskEmbedding（更优的内存管理）
- 代码维护：手写 ~300 行 → 引擎内置 0 行

详见 [[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]]
