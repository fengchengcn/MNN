---
date: 2026-06-10
status: active
tags: [qwen3-asr, omni, streaming, vad]
category: plan
aliases: [Omni流式方案, Streaming Plan]
related: [[omni-parameters]], [[progress]], [[llmexport-migration]]
---
# Qwen3-ASR Omni 模式流式实现方案

> 创建：2026-06-10 | 最后更新：2026-06-11
> 关联：[[Qwen3-ASR-OMNI-PARAMETERS]] [[Qwen3-ASR-MEMORY-ANALYSIS]] [[Qwen3-ASR-MNN-PROGRESS]]
> **状态：Phase 1 ✅ | Phase 2 ✅ | Phase 2.6 ✅ (VAD+扩展窗口) | Phase 3 已分析 (30-40% 成功概率)**
>
> **当前生产方案：Phase 2.6 VAD 引导扩展窗口。** Phase 2.5 纯 VAD 分段已被推翻。
> 旧引擎流式方案见 [[Qwen3-ASR-STREAMING-PLAN]] (LEGACY)。

---

## 一、总览

### 核心思路

直传 PCM 绕过 WAV 文件 I/O，滑动窗口（每 1.5s/3s 增量推理）+ VAD 控制生命周期。

```
AudioRecord(100ms chunks) → Silero VAD → 语音段累积
  → 每 1.5s/3s: setAudioData(snapshot) + generate("<audio>stream</audio>")
    → onProgress 逐 token 更新 LIVE 卡片
  → VAD 静音/FINAL: 全量快照 → 永久结果卡片
```

### 实施阶段

| 阶段 | 目标 | 状态 |
|------|------|:----|
| Phase 1 | 直传 PCM，绕过 WAV 文件 I/O（6 文件） | ✅ |
| Phase 2 | 滑动窗口伪流式（1 文件） | ✅ |
| Phase 2.5 | 纯 VAD 分段（误入歧途） | ❌→Phase 2.6 |
| Phase 2.6 | VAD 引导扩展窗口（当前方案） | ✅ |
| Phase 3 | 真流式增量 AE（引擎改造） | 已分析 |

---

## 二、Phase 1 踩坑（直传 PCM）

核心：FloatArray → `_Const` VARP → `pending_audio_` → Omni 引擎注入，绕过 WAV 磁盘 I/O。

### 关键踩坑

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `GetFloatArrayLength` 编译失败 | JNI 无此方法，正确 API 是 `GetArrayLength` | `GetArrayLength(samples)` |
| 2 | `_Input` VARP 导致 SIGSEGV | `_Input` 是占位符，张量信息未初始化，`spectrogram` 访问 `dim[1]` 时崩溃 | 改用 `_Const`（张量信息完整） |
| 3 | `{1, N} NCHW` shape 仍 crash | `AUDIO::load()` 返回 `{N}` 1D NHWC，shape 不匹配 | 改为 `{num_samples}` 1D NHWC |
| 4 | `pending_audio_.reset()` 编译失败 | MNN VARP 是 class 不是 `shared_ptr` | `pending_audio_ = nullptr` |

---

## 三、Phase 2 踩坑（滑动窗口）

### 关键踩坑

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `withContext(Dispatchers.Main)` 编译失败 | `onProgress()` 是 Java 回调接口，非 suspend 函数 | `runOnUiThread {}` |
| 2 | **Prompt 污染导致幻觉** | `keepHistory=true` 使每次 `generate()` 追加 `<audio>` 到历史。14 次增量后 prompt 被 14 个 `<audio>stream</audio>` 污染 | `setKeepHistory(false)` |

**坑 #2 是精度杀手**：修复前 FINAL 输出幻觉文本（与真实语音无关），修复后准确率恢复正常。

### 实机数据（Kirin 9000, FP16+greedy, 18s 录音）

| 时刻 | AE 耗时 | Decode | 增量结果 |
|------|:------|:------|------|
| 2-6s | 172-403ms | 390-735ms | 静音期，空输出 |
| 8s | 402ms | 1.0s | `安装成功。现在注意观察。` |
| 10-16s | 503-884ms | 1.3-2.0s | 逐渐收敛，部分幻觉 |
| **FINAL** (17.4s) | **861ms** | **2.7s** | **`安装成功。现在注意观察日志中的关键变化。明天星期几？`** ✅ |

**关键发现**: Full-Attention AE 的"自动纠错"效应 — 早期增量因音频不足产生幻觉，FINAL 全量推理自然修正。

---

## 四、Phase 2.5 教训（纯 VAD 分段 → 误入歧途 ❌）

> **核心教训**: VAD 准确知道"一句话何时结束"，但如果因此扔掉扩展窗口（每次只发当前段音频），AE 的 Full Bidirectional Attention 失去累积上下文，罕见/技术术语识别崩溃。

**实机证据**: 同一段技术讨论录音，Phase 2 FINAL ~95% 准确率，Phase 2.5 纯分段 ~30%：
- 分段 #1: `U V D，精确检测，一句话结束`（VAD→UVD）
- 分段 #2: `有威力，精确检测，一句话结束，一次性扔个点`（幻觉）
- 分段 #3: `增量记忆多于现有`（增量推理→增量记忆，多余且有害→多于现有）

**根因**: Qwen3-ASR AE 使用 Full Bidirectional Attention（18 层，每帧 attend 全部帧）。分段独立推理切断段间声学上下文，对低频词致命。

**正确做法**: VAD 控制生命周期 + 扩展窗口保留全量上下文，两者缺一不可。

---

## 五、Phase 2.6：VAD 引导扩展窗口 ✅

> 最后更新：2026-06-13（补充实际实现细节 & 踩坑修复）

### 核心设计

- **VAD**: Silero VAD（LSTM 神经网络，`silero_vad.onnx` → MNN Interpreter，`silero_vad_jni.cpp` → `Vad.kt`）
- **扩展窗口**: 每次推理发送 `[0..current]` 全部累积音频
- **并发**: `Channel.UNLIMITED` + 单 consumer 协程串行，`sendCumulativeTask()` 调用 `trySend` 不阻塞录音线程

### 累积滑动窗口实现（2026-06-13）

Android 端 `Qwen3AsrTestActivity` 中的实际实现：

```
AudioRecord(100ms chunks) → Silero VAD → 语音段检测
  → accumulatedSegments.add(segment)  // 累积，非独立分发
  → 段间插入 0.4s 零填充静音
  → sendCumulativeTask(): 拼接全部累积段 → trySend 到 channel
  → consumer 串行处理，每次推理用全部累积音频
```

**句间断句**（2026-06-13 新增）：
- 通过 wall-clock 段间间隔区分句内/句间停顿
- 间隔 < 1.5s：同一句子，继续累积，自动纠错
- 间隔 ≥ 1.5s：句间边界 → `flushCurrentSentence()` 发最终推理，清空 buffer 开始新句

### Silero VAD 配置（实机使用值）

| 参数 | 值 | 说明 |
|------|:------|------|
| `threshold` | 0.5 | 语音概率阈值 |
| `minSpeechDuration` | 0.15s | 最短语音长度 |
| `minSilenceDuration` | 0.4s | 停顿判定阈值 |
| `maxSpeechDuration` | 15.0s | 最大语音时长安全网 |
| `windowSize` | 512 (32ms@16kHz) | 推理窗口 |

### 关键参数

| 参数 | 值 | 说明 |
|------|:------|------|
| 段间静音间隔 | 0.4s | 累积时段间插入的零填充（匹配 VAD minSilenceDuration） |
| 句间边界 | **1.5s** | wall-clock 段间间隔阈值，超过则断句 |
| 录制循环周期 | 100ms | `CHUNK_INTERVAL_MS` |
| Locked-gain | 固定增益 | 前 1s 估算 ambient RMS → 锁定增益 → 全段统一应用 |

### 踩坑记录（2026-06-13 修复）

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | **重复输出**：同一句子显示两次 | 句边界 flush 发 final 任务，但最后 interim 已覆盖全部累积音频 | `lastInferenceSegmentCount` 跟踪上次推理时段数，flush 时段数未增加 → 跳过 final |
| 2 | **最后一句不输出** | flush 时 drain channel 排空所有任务（含当前句子），然后跳过 final → 整句丢失 | 去掉 channel drain，consumer 按序处理 |
| 3 | **句子/轮次标注错乱** | `sentenceIndex`/`cumulativeRound` 共享变量，consumer 读取时已被后续 flush 修改 | `SegmentTask` 构建时捕获 `sentIndex`/`sentRound`，consumer 读快照值 |

### config.json 采样修正

ASR 是确定性任务，必须用 greedy：

| 参数 | 修正前 | 修正后 |
|------|--------|--------|
| `sampler_type` | `mixed` | **`greedy`** |
| `temperature` | 0.1 | **0.0** |
| `top_k` | 40 | **1** |
| `top_p` | 0.9 | **1.0** |
| `n_gram` | 8 | **0** |

> 罕见 token 置信度仅 30-70%，任何非零 temperature 噪声足以偏离正确路径。

### FP16 模型支持

`findOmniModel()` 评分制：FP16(2) > INT8(1) > 其他(0)。FP16 权重 ~1.1GB，低频 token embedding 精度优于 INT8。

### 各方案对比

| 维度 | Phase 2 (盲推) | Phase 2.5 (纯VAD) ❌ | Phase 2.6 (VAD+扩展) ✅ |
|------|:---|:---|:---|
| 语音检测 | 无 | RMS VAD | **Silero VAD** |
| 推理窗口 | 扩展 ✅ | 分段独立 ❌ | 扩展 ✅ |
| 首次结果 | ~6-8s | 段结束 | 每段累积后 |
| 最终准确率 | 高 | **低（上下文断裂）** | **高** |
| 罕见/术语 | 好 | **差 ~30%** | **好 ~95%** |
| 并发安全 | 竞态 | 单次 | **Channel 串行** |
| 句间断句 | 无 | 无 | **1.5s wall-clock 间隔** |

### 已知限制

1. **Full-Attention AE 全量重算**: 每次增量重跑全量 fbank+AE。≤8s 段可接受（Kirin 9000: ~400-800ms AE）
2. **无跨句上下文**: 句间独立推理（1.5s 间隔后的新句从零开始）
3. **Channel.UNLIMITED**: 不排队列积压，但同一句子多次 interim 推理均有意义（越来越完整）

---

## 六、Phase 3 分析：真流式增量 AE

> **结论: 整体成功概率 30-40%。核心制约是 Qwen3-ASR AE 的 Full Bidirectional Attention 架构，非工程问题。**
>
> 当前 Phase 2.6 已满足多数需求，建议暂缓 Phase 3。

### 关键架构发现

1. **AE 使用 Full Bidirectional Attention**（非 Causal）: ONNX 导出仅 1 个输入 `input_features`，无 `attention_mask`。`omni.cpp:897` 中 `inputNames.size() == 1` 分支走 full attention。
2. **Absolute Positional Embeddings**（非 RoPE）: 学习的位置编码，序列长度变化时旧帧 attention 分布改变，旧 KV cache "过期"。
3. **MNN KVMeta 支持增量操作**（`KVMeta::add/remove/previous`），但增量 prefill 路径从未被测试。

### 四项改造评估

| 模块 | 成功率 | 关键风险 | 工作量 |
|------|:------|------|:------|
| Streaming fbank（overlap buffer） | **85%** | 低，成熟信号处理技术 | 3-5 天 |
| Incremental AE（Sliding Window） | **55%** | 位置编码一致性，精度损失未知 | 1-2 周 |
| 增量 Embedding 注入 | **65%** | position ID 连续性管理 | 3-5 天 |
| KV Cache 增量 Prefill | **35%** | attention mask 形状不支持、引擎首次走此路径 | 2-4 周 |

### 渐进路线

```
Phase 3a (概率 ~75%): Streaming fbank + Sliding Window AE
  → AE 计算量 ~40% 减少，首 token 延迟不变，1-2 周

Phase 3b (概率 ~40%): Decoder 增量 Prefill
  → 首 token 延迟 ~600ms→~200ms，边说边出，3-5 周

Phase 3c (概率 ~20%): Causal AE（需重训模型，不在工程范围）
```

### 关键源码索引

| 文件 | 关键内容 |
|------|------|
| `omni.cpp:848-962` | `audioProcess()` — fbank+AE+embedding 注入 |
| `omni.cpp:897-930` | AE forward 分支（`inputNames.size()>1` 判断） |
| `omni.cpp:1481-1496` | `response()` → `generate_init()` → `generate()` |
| `llm.cpp:776-799` | `generate_init()` — Phase 3b 需绕过 |
| `llm.cpp:640-745` | `forwardVec()` — prefill chunking + KVMeta |
| `audio.cpp:639-665` | `whisper_fbank()` — Phase 3a 改造目标 |
| `audio.py:520-540` | `Qwen3AsrAudio.export()` — 确认仅 1 输入 |
| `source/core/KVMeta.hpp` | KV Cache 底层机制 |

---

## 七、性能对比（实机: Kirin 9000 / TAS-AL00）

| 指标 | Phase 1 | Phase 2 | Phase 2.6 |
|------|:---|:---|:---|
| 文件 I/O | **0ms** | **0ms** | **0ms** |
| fbank+AE (3-8s) | ~500ms | ~400-800ms | ~400-800ms |
| 首次结果可见 | — | ~6-8s | **~1.5s** |
| Decode 速度 (FP16) | — | 30-35 t/s | 18-22 t/s |
| 并发风险 | 无 | 增量+FINAL 竞态 | **无**（Channel 串行） |
| 最终准确率 | — | 高 | **高** |
| 罕见/术语 | — | 好 | **好（全量上下文）** |

---

## 八、风险总览

### Phase 1/2/2.6（已解决）

| 风险 | 缓解 |
|------|------|
| `_Input` VARP 崩溃 | 改用 `_Const` |
| Shape 格式不匹配 | `{N} NHWC` 匹配 `AUDIO::load()` |
| Prompt 污染 | `setKeepHistory(false)` |
| 增量+FINAL 竞态 | Channel 串行化 |
| 纯 VAD 分段准确率崩溃 | 恢复扩展窗口 |
| Qwen3-ASR 0.6B 词汇覆盖限制 | **模型能力边界**，FP16 优于 INT8 |

### Phase 3（未来）

| 风险 | 概率 | 严重性 |
|------|:---:|:------|
| AE Full Bidirectional Attention 不适配 streaming | 高 | 🔴 阻断性 |
| MNN 增量 Prefill 未验证 | 高 | 🔴 阻断性 |
| 精度损失（WER +5-15%） | 中 | 🟡 业务风险 |
