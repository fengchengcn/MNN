---
date: 2026-06-15
status: completed
tags: [qwen3-asr, accuracy, debug, retrospective, root-cause]
category: analysis
aliases: [精度调试全历程, Accuracy Debugging Journey]
related: [[root-cause-analysis]], [[export-pipeline-analysis]], [[fbank-numerical-analysis]], [[progress]], [[../plans/replicate-onnx-export-plan]]
---

# Qwen3-ASR MNN 精度调试全历程

> 2026-06-04 ~ 2026-06-15 从"精度追不上 sherpa-onnx"到"长音频幻觉消失"的完整归因链条。包含两次错误归因及其纠正过程。

---

## 问题总览

```
问题层次       归因                                 日期      结论
─────────────────────────────────────────────────────────────────────
Layer 1    AEC/NS 硬件音效                          06-14    ✅ 真因（修复后大幅提升）
Layer 2    FBank 数值差异                            06-14    ❌ 误判（对齐后无改善）
Layer 3    Sampling 参数                             06-14    ❌ 误判（改 greedy 后无改善）
Layer 4    "AE 架构不等价" → 真正是 2 个代码 bug      06-14    🔴 真因（cosim 0.30→0.993）
Layer 5    "非分块 conv 不一致" → 真正是 PE 模式错误   06-15    🔴 真因（长音频幻觉消失）
```

---

## Layer 1：AEC + NoiseSuppressor 硬件音效（真因）

### 现象

MNN ASR 识别精度显著差于 sherpa-onnx，即使 fbank 通过 kaldi-native-fbank 对齐后仍无明显改善。

### 归因过程

最初怀疑是模型导出质量或推理精度问题，逐项排查了 FBank、sampling、FP16 精度等多个方向均无明显改善。最终回溯到 Android 音频采集链路，发现 `Qwen3AsrTestActivity.initAudioRecord()` 开启了 `AcousticEchoCanceler` + `NoiseSuppressor`。

### 真正原因

Android 硬件 DSP 音效为**语音通话**场景优化（窄带），用于 ASR（全频带）会导致：
1. 高频辅音（/s/, /f/, /sh/）被噪声抑制误删 → 音素识别错误
2. 动态范围被 AEC 压缩 → FBank formant 跟踪失效
3. 非线性 DSP 谐波 → 引入训练数据中不存在的频谱特征
4. 不同手机的 DSP 质量差异大 → 精度不可预测

sherpa-onnx 的 `AudioRecorder.kt` 使用 raw MIC → PCM，无任何预处理。MNN 多加的 AEC/NS 反而不对齐。

### 修复

移除 AEC/NS 初始化代码，raw MIC 直通，与 sherpa-onnx 音频链路完全对齐。

### 教训

**ASR 的音频链路必须与训练 pipeline 完全一致。任何"增强"（降噪、回声消除、AGC）都是偏差，不是优化。**

### 残留问题

修复后识别质量大幅提升，但精度仍然追不上 sherpa-onnx。

---

## Layer 2：FBank 数值差异（误判）

### 归因

怀疑 C++ whisper_fbank 实现与 Python 参考有数值差异。

### 排查

逐项对齐：hann window 定义、mel filterbank 公式（HTK/Slaney）、preemphasis 系数（0.97）、log clamp（`max(spec, 1e-10)`）、dBFS 归一化公式 `(x+4)/4`。最终切换到 kaldi-native-fbank（`whisper_fbank_knf()`）彻底排除实现差异。

### 结论

**FBank 不是误差源。遍历所有参数对齐后精度无改善。**

---

## Layer 3：Sampling 参数（误判）

### 归因

ASR 任务使用了 `mixed` 采样（temperature=0.1, top_k=40, top_p=0.9），不利于确定性识别任务。

### 修复

改为 greedy 采样：`temperature=0, top_k=1, top_p=1.0, n_gram=0`。

桌面端对比实验（§5 AE 隔离实验，§6 Decoder 对比实验）排除了 Audio Encoder 和 FP16 推理精度。

### 结论

**不是主要误差源。**

---

## Layer 4："AE 架构不等价" → 真正是 2 个代码 bug（真因，第一次错误归因）

### 现象

桌面端对比实验：sherpa-onnx AE（conv_frontend.onnx → encoder.int8.onnx）vs MNN AE（audio.mnn），相同 fbank 输入：

| 指标 | sherpa-onnx AE | MNN AE |
|------|:------------:|:-----:|
| 输出帧数 | 65 | 53 |
| cosim | **0.30** | — |
| first token | 15 | 15（意外 match）|

cosim 仅 0.30，帧数差异 65 vs 53。

### 错误归因（06-14 早期）

手写 `AudioEncoder.forward()` 缺失了官方模型的 **Chunk/Pad/Slice** 操作，导致"图结构不等价"、"subsampling ratio 不同（~6.5× vs ~8×）"。这一假设认为 Pad/Slice 在 Whisper-style conv 中的边界处理改变了等效下采样率，从而系统性偏移了输出帧数和特征值。

基于此错误归因，启动了三条线并行推进：
1. **MNNConvert 方案**：直接转换 Wasser1462 的 ONNX → MNN，绕过手写 forward()（→ [[sherpa-ae-mnn-integration]]）
2. **双模型导出方案**：llmexport.py 自控导出 conv_frontend + encoder（→ [[../plans/replicate-onnx-export-plan]]）
3. **完全重写 forward()**：移植官方 Chunk/Pad/Slice/Window 到 llmexport.py

### 翻转（06-14 晚些时候）

发现官方 modeling 代码在独立 GitHub 仓库 `github.com/QwenLM/Qwen3-ASR`（含完整 `modeling_qwen3_asr.py`），与手写版逐行对比后定位到**两个代码实现 bug**，并非架构缺失：

| # | Bug | 位置 | 现象 |
|---|-----|------|------|
| 1 | `conv_out` 多余随机 bias | `nn.Linear(..., bias=True)` 但 safetensors 中无 bias 权重 | 896 个未初始化随机值，每次加载结果不同 |
| 2 | PE 公式 interleaved 顺序 | `[sin0,cos0,sin1,cos1,...]` 应为 `[sin0,...,sinN,cos0,...,cosN]` | 每个位置 max diff = 2.0，经 18 层 Transformer 放大 |

**两个 bug 叠加的破坏力**：bias 引入 896 维随机偏移 + PE 每位置差 2.0 → 18 层 Transformer 逐层放大 → 输出 cosim 仅 0.30。

### 修复

```python
# Bug 1: conv_out bias
self.conv_out = nn.Linear(conv_hidden * 16, d_model, bias=False)  # 原: bias=True

# Bug 2: PE formula
pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)  # 原: interleaved
```

### 修复后验证

| 指标 | 修复前 | 修复后 |
|------|:------:|:------:|
| MNN vs Wasser1462 cosim (T=100, 同帧数) | 0.30 | **0.993** |
| PyTorch vs ONNX max diff | 未测 | **2.36e-6** |
| 帧数 (T=500) | 53 | 63 |

### 经验教训

1. **"Chunk/Pad/Slice 导致架构不等价"理论被彻底推翻**——这些操作是官方代码的长音频内存优化，非精度要求。短音频（≤30s）单窗口内全序列 attention 与分窗版数学等价。
2. **错误归因的连锁反应**："架构不等价"假设驱动了 MNNConvert 方案、双模型方案设计、C++ 适配等一系列工程，虽然最终方向和产出被保留，但若早一步发现官方 modeling 代码，可节省 1-2 天。
3. **cosim 0.30 是两个确定性 bug**，而非架构性的"图结构不等价"——这个教训说明在归因到"架构差异"之前，应先穷举代码实现错误。

### 残留问题

修复后桌面端短序列 cosim 0.993 验证通过。但手机实测出现**长度依赖性退化**——短句（≤10s）正常，长句（≥16s）幻觉严重。且**两种方案（简化单模型 + 双模型照抄 ONNX）表现完全一致**。

---

## Layer 5："非分块 conv 不一致" → 真正是 PE 模式错误（真因，第二次错误归因）

### 现象

两种方案在手机实测中表现完全一致：
- 短句（≤10s）：识别准确率高 ✅
- 长句（≥16s）：语音幻觉严重 ❌

### 错误归因（06-14 晚）

"模型训练时对每 100 mel 帧分块独立 conv，各 chunk 边界有 zero-padding artifact。推理时全序列连续 conv 无此 artifact → 多 chunk 时中间表示偏离训练分布。"

基于此假设，判断"根本修复需移植完整 Chunk/Pad/Slice 逻辑，但这些 op 无法被 ONNX trace"。缓解措施：靠 VAD 模式天然切短段绕过。

### 翻转（06-15）—— 关键诊断信号

**两种方案的 conv 策略完全不同**（简化单模型：连续 conv；双模型：fold-into-batch chunked conv），但**长音频退化行为完全一致**。这说明根因**不在 conv，而在两者共享的 encoder 部分**。

排查共享部分，定位到 **Positional Encoding 模式**：

```python
# 官方训练时（modeling_qwen3_asr.py:712-717）：
positional_embedding = (
    self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
    #                                            ^^^^^^^^^^^^^^^^^^^^^^^^
    #                                            ALWAYS 13（每 chunk conv 输出帧数）
)
padded_embed = padded_embed + positional_embedding
# PE 切片到 13，broadcast 到所有 chunk → 每个 chunk PE 重置为 0..12

# 我们的实现（两种方案共享）：
seq_len = x.size(1)
x = x + self.positional_embedding[:seq_len, :]
# PE 切片到 seq_len → PE[0..seq_len-1] 连续递增，从不重置
```

### 真正原因

模型在训练期间**只见过 PE 位置 0..12**（每 chunk 13 帧，PE 在 chunk 边界重置）。我们的实现使用连续递增 PE 0..seq_len-1：

| 音频时长 | mel 帧 | enc 帧 | 官方 PE（重复） | 我们的 PE（连续） | 外推倍数 |
|---------|:----:|:----:|:----------:|:----------:|:------:|
| 3s | 300 | ~38 | 0..12 × 3 | **0..37** | 3× |
| 7s | 700 | ~88 | 0..12 × 7 | **0..87** | 7× |
| **16s** | 1600 | ~200 | 0..12 × 16 | **0..199** | **15×** |
| 30s | 3000 | ~375 | 0..12 × 30 | **0..374** | **29×** |

短序列外推倍数低（3×），sinusoidal PE 尚可容忍，注意力模式近似训练分布。长序列外推 15×+ 时，Q·K^T 注意力权重完全脱离训练分布 → 注意力图近似随机噪声 → decoder 收到退化特征 → 生成幻觉文本。

**A/B 测试确认**：逐层替换 Wasser1462 ONNX 文件（conv_frontend.onnx / encoder.int8.onnx），定位到问题在 **encoder PE**，非 conv_frontend。完整 Wasser1462 双模型通过长音频测试（排除 decoder/C++ 路径问题）。

### 修复

```python
# 修复前：PE 连续递增
pe = self.positional_embedding[:seq_len, :]        # PE[0..seq_len-1]

# 修复后：PE 按 chunk 重复（复刻训练行为）
N = seq_len // 13                                   # chunk 数量
pe = self.positional_embedding[:13, :].repeat(N, 1)[:seq_len, :]  # PE[0..12] × N
```

同步确认两个配套参数：
- **chunk_size = 100**（`n_window * 2`，非 `conv_chunksize=500`——后者是批处理上限，防 OOM）
- **不移除 partial chunk 的 ghost frames**（与 Wasser1462 ONNX 行为对齐，训练时也保留）

### 修复后验证

| 指标 | 修复前 | 修复后 |
|------|:------:|:------:|
| 帧数 vs Wasser1462 (T=50~1600) | 不匹配 | **100% 匹配** |
| PyTorch vs Wasser1462 ONNX cosim | 低 | **1.000000** |
| MNN E2E cosim @ T=1600 | — | **0.9996** |
| 手机短句 | ✅ | ✅ |
| 手机长句 (≥16s) | ❌ 幻觉 | ✅ **修复** |

### 单模型视角

PE 修复对单模型同样适用——只需改一行 PE 代码。单模型修复后理论上也能解决长音频问题。但双模型还有一个次要优势：fold-into-batch conv 与训练时的分块 conv 严格一致（帧数 100% 匹配，边界 artifact 完全复现训练行为），单模型连续 conv 有约 1 帧偏差。

---

## 三个根因总结

| # | 根因 | 位置 | 发现日期 | 影响 |
|---|------|------|:------:|------|
| 1 | `conv_out` 多余随机 bias | `qwen3_asr_model.py` | 06-14 | 896 个随机值，每次加载不同 |
| 2 | PE 公式 interleaved 顺序 | `qwen3_asr_model.py` | 06-14 | 每个位置 max diff=2.0 |
| 3 | PE 连续递增 vs 按 chunk 重复 | `qwen3_asr_model.py` | 06-15 | 长音频外推 15×+ → 幻觉 |

三个根因均在 `qwen3_asr_model.py` 中，均源于缺官方 modeling 代码时的手写 forward()。

---

## 经验教训

### 方法论

1. **cosim 高 ≠ 端到端精度好**：修复 bug #1/#2 后短序列 cosim 0.993，但 PE 错误在短序列上外推 3×、长序列 15×。cosim 验证的 T=100 测试用例完全没覆盖到这个维度。**验证用例必须覆盖任务的全部实际输入范围。**

2. **两种方案表现一致 = 共享根因**：简化单模型和双模型的 conv 策略完全不同，但长音频退化完全一致 → 根因必然在共享部分（encoder PE + attention），不在差异部分（conv）。**当多个独立方案复现同一 bug 时，交集就是诊断方向。**

3. **先穷举代码错误，再归因架构差异**：Layer 4 的 "架构不等价" 假设驱动了大规模工程响应（MNNConvert、双模型拆分、C++ 适配），但 root cause 只是两个 `bias=True` 和 PE 公式——纯代码实现错误。**假设代价阶梯：代码 bug → 配置错误 → 数值精度 → 架构差异。按此顺序排查，不要跳级。**

4. **"官方代码不存在" 假设的代价**：最初以为 Qwen3-ASR modeling 代码未开源（HuggingFace 仓库只有权重），手写了整个 forward()。实际上官方代码在独立 GitHub 仓库 `github.com/QwenLM/Qwen3-ASR`。**检查模型是否有独立代码仓库（非 HuggingFace/ModelScope），不能只看模型发布页。**

### 错误归因的价值

每次错误归因都产出了有价值的中间产物：

| 错误归因 | 产出 | 最终保留 |
|---------|------|:------:|
| "FBank 差异" | kaldi-native-fbank 集成，log floor guard | ✅ 保留 |
| "Sampling 参数" | greedy 采样配置 | ✅ 保留 |
| "架构不等价" | MNNConvert 双模型链、C++ 双模型路径、ONNX 导出验证管线 | ✅ 保留（C++ 路径 + 验证管线） |
| "非分块 conv" | 双模型方案的最终完善 | ✅ 保留（次要精度优化） |

### 关于"单模型能否搞定"

PE 修复是核心，单模型加上 PE 修复理论上也能解决长音频问题。双模型相比单模型多对齐了一个次要因素（fold-into-batch conv 帧数 100% 匹配），但主因是 PE。最终选择双模型是因为它在 PE 修复过程中已经建好并验证完成，而非单模型搞不定。

---

## 时间线

```
06-04  初步调研 → MNN + llmexport.py 方案
06-05  ONNX 导出成功
06-07  Android 端到端跑通（旧引擎）
06-10  Omni 引擎迁移完成
06-14  🔴 AEC/NS 根因发现并修复 → 精度大幅提升，但仍追不上 sherpa-onnx
06-14  桌面端 AE 隔离实验：FBank 和 Sampling 方向均排除
06-14  🔴 sherpa-onnx AE vs MNN AE cosim ~0.30 → "架构不等价" 假设
06-14  MNNConvert 双模型方案验证通过（cosim 1.0/0.997）
06-14  发现官方 modeling 代码 → 翻转：2 个代码 bug，非架构不等价
06-14  修复 bias + PE formula → cosim 0.993（短序列）
06-14  手机实测：短句正常，长句幻觉 → "非分块 conv" 假设
06-15  🔴 翻转：两种方案 conv 不同但退化一致 → 定位到 encoder PE 模式
06-15  PE 修复 + 验证：PyTorch vs Wasser1462 cosim 1.0，手机长音频正常 ✓
```

---

## 参考

- 官方 modeling 代码：`github.com/QwenLM/Qwen3-ASR` → `modeling_qwen3_asr.py`
- 本地副本：`/tmp/modeling_qwen3_asr.py`
- Wasser1462 ONNX：`/Users/bxy/Documents/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/`
- 精度根因分析：[[root-cause-analysis]]
- 导出链路分析：[[export-pipeline-analysis]]
- 双模型 ONNX 方案：[[../plans/replicate-onnx-export-plan]]
- FBank 分析：[[fbank-numerical-analysis]]
- 项目进度：[[../plans/progress]]
