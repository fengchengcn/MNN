# Qwen3-ASR Omni 模式推理参数分析

> 创建：2026-06-10
> 基于：MNN Omni 引擎源码 + config.json + C++ LlmConfig
> 关联文档：[[Qwen3-ASR-MEMORY-ANALYSIS]] [[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]]

---

## 一、上下文长度管理

### 1.1 关键参数

| 参数 | 值 | 来源 | 含义 |
|------|:------|------|------|
| `max_all_tokens` | **2048** | `llmconfig.hpp:152` 默认值 | 总上下文窗口上限（输入+输出） |
| `max_new_tokens` | **512** | `llmconfig.hpp:156` 默认值 | 单次生成最多输出 token 数 |
| `keep_history` | **false**（ASR 场景） | Kotlin 调用方控制 | 多轮对话历史是否保留 |

> **注意**：`config.json` 中未显式设置这两个参数，因此走 C++ 默认值。

### 1.2 ASR 调用上下文流程

```
每次 ASR 调用：
  WAV 文件 → fbank 特征提取 → Audio Encoder（18 层 Transformer）
  → embedding 矩阵 [T, 1, 896]
  → 注入特殊 token：audio_start(151669) + audio_pad×N(151676) + audio_end(151670)
  → LLM Decoder 逐 token 生成文本
  → 遇到 EOS (<|im_end|>) 或达到 max_new_tokens=512 时停止
```

### 1.3 不同语音长度的 token 消耗

| 场景 | Audio Embedding 长度 | 预估输出 tokens | 总长度 | 是否触及 2048 上限 |
|------|:---------------------|:----------------|:-------|:-------------------|
| 3 秒短语音 | ~40 维 AE + 3 个特殊 token | ~15 tokens | ~58 | ❌ 远未触及 |
| 10 秒中文语音 | ~125 维 AE + 3 个特殊 token | ~30 tokens | ~158 | ❌ 远未触及 |
| 30 秒长语音 | ~375 维 AE + 3 个特殊 token | ~125 tokens | ~503 | ❌ 仍有富余 |
| 30 秒 + 512 tokens 满输出 | ~375 维 AE | 512 tokens | ~890 | ❌ 仍在窗口内 |

**结论**：Qwen3-ASR 场景下，上下文长度完全不是瓶颈。即使 30 秒长语音 + 满额输出，也只用了 ~44% 的 2048 窗口。

### 1.4 上下文管理机制

| 维度 | 说明 |
|------|------|
| **无状态调用** | `keep_history=false`，每次 ASR 调用独立，不保留上文 |
| **Audio Embedding 注入** | Audio Encoder 输出直接作为 LLM Decoder 的 prefix embeddings，不走 tokenize 路径 |
| **Position IDs** | Omni 引擎内部维护 `mPositionIds`，为 audio/text 混合序列分配位置编码 |
| **KV Cache** | MNN 内部管理，28 层 × 8 KV 头 × seq_len × 128 dim × FP16 = ~17MB（10 秒语音） |
| **kvcache_mmap** | 支持但未启用；开启后可将 KV Cache 卸载到磁盘换取更大上下文 |
| **终止条件** | 优先级：USER_CANCEL > 命中 EOS > 达到 max_new_tokens |

---

## 二、推理生成参数（完整清单）

### 2.1 当前生效配置

来自 `Qwen3-ASR-MNN-INT8/config.json`：

```json
{
  "sampler_type": "mixed",
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "min_p": 0.05,
  "tfs_z": 1.0,
  "typical": 0.95,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "penalty_window": 0,
  "n_gram": 8,
  "ngram_factor": 1.0
}
```

### 2.2 参数详解

#### 采样策略

| 参数 | 当前值 | 默认值 | 作用 |
|------|:-------|:-------|------|
| `sampler_type` | `mixed` | `mixed` | 采样器类型：`greedy` / `temperature` / `mixed` |
| `mixed_samplers` | 未设置（用默认） | `["topK","tfs","typical","topP","min_p","temperature"]` | mixed 模式下的采样器流水线 |

#### 概率截断

| 参数 | 当前值 | 范围 | 作用 |
|------|:-------|:-----|------|
| `top_k` | 40 | 1–vocab_size | 仅从概率最高的 K 个 token 中采样 |
| `top_p` | 0.9 | 0.0–1.0 | Nucleus 采样：累计概率 ≤ p 的最小 token 集合 |
| `min_p` | 0.05 | 0.0–1.0 | 过滤概率低于 max_prob × min_p 的 token |
| `tfs_z` | 1.0 | 0.0–1.0 | Tail-free sampling（1.0 = 无过滤，0.95 = 典型过滤） |
| `typical` | 0.95 | 0.0–1.0 | Typical sampling：选择与信息熵期望最接近的 token 集合 |

#### 温度

| 参数 | 当前值 | 默认值 | 作用 |
|------|:-------|:-------|------|
| `temperature` | **0.8** | 1.0 | 控制 softmax 的平滑程度。越低越 greedy（确定性高），越高越随机 |

> Temperature=0.8 说明模型偏好**中等随机性**——比 greedy 温和一些多样性，但不过于发散。适合 ASR 输出（需要准确转写）。

#### 惩罚机制

| 参数 | 当前值 | 默认值 | 作用 |
|------|:-------|:-------|------|
| `repetition_penalty` | 1.0 | 1.0 | 重复惩罚（1.0=禁用，1.1=轻度惩罚已出现 token，<1.0=鼓励重复） |
| `presence_penalty` | 0.0 | 0.0 | 存在惩罚（正数降低已出现 token 概率） |
| `frequency_penalty` | 0.0 | 0.0 | 频率惩罚（正数按出现次数成比例降低概率） |
| `penalty_window` | 0 | 0 | 惩罚生效的历史窗口（0=全部历史） |

#### N-gram 控制

| 参数 | 当前值 | 默认值 | 作用 |
|------|:-------|:-------|------|
| `n_gram` | 8 | 8 | N-gram 匹配大小 |
| `ngram_factor` | 1.0 | 1.0 | N-gram 缩放因子（>1.0 惩罚已出现 n-gram） |

### 2.3 Mixed Sampler 流水线

当前 `sampler_type: "mixed"` 的执行顺序（由 `sampler.cpp:124-146` 决定）：

```
penalty（置前）→ topK → tfs → typical → topP → min_p → temperature
```

C++ 源码逻辑：
1. 遍历 `mixedSamplers` 列表，对每个采样器调用 `configSampler()`
2. `penalty` 被强制移到流水线最前端（`sampler.cpp:128-139`）
3. 最后一个采样器的类型决定最终 `select_type`（temperature 采样 vs greedy argmax）
4. 每个阶段从上一阶段的输出 token 集合中进一步过滤

### 2.4 与其他采样模式的对比

| 模式 | 流水线 | 适用场景 |
|------|:-------|---------|
| `greedy` | 仅 argmax | 需要最高确定性、可复现 |
| `temperature` | temperature → argmax | 简单随机采样 |
| `mixed`（当前） | penalty → topK → tfs → typical → topP → min_p → temperature | 高质量文本生成，平衡多样性与准确性 |

---

## 三、temperature 参数详解

### 3.1 生效位置

- `config.json` 第 9 行：`"temperature": 0.8`
- 覆盖了 C++ 默认值 1.0（`llmconfig.hpp:448`）
- 在 Mixed 流水线中，temperature 是**最后一个阶段**

### 3.2 源码实现

```cpp
// sampler.cpp:20-47
void SamplerState::ensureProbs(float temperature) {
    float invTemp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= invTemp;  // logit 缩放
    }
    // softmax → probs
    float maxLogit = *std::max_element(logits, logits + vocab_size);
    float sum = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
}
```

### 3.3 Temperature 效果

| Temperature | 效果 | 适合 |
|:------------|:-----|:-----|
| 0.0–0.3 | 高度确定性，几乎 greedy | 需要精确转写的 ASR |
| **0.6–0.9**（当前 0.8） | 中等随机性 | 平衡准确性与自然的 ASR |
| 1.0 | 原始分布 | 通用文本生成 |
| >1.0 | 增加低概率 token 机会 | 创意写作 |

---

## 四、运行时动态修改参数

### 4.1 Kotlin 端 API

```kotlin
// 修改最大生成 token 数
llmSession?.updateMaxNewTokens(256)

// 修改完整推理配置（JSON 字符串）
llmSession?.updateConfig("""{
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.05
}""")
```

### 4.2 Android UI 中的配置面板

MnnLlmChat 主 App 已有 `SettingsBottomSheetFragment` 支持以下参数的 UI 调节：

- Temperature（滑动条）
- Top-P / Top-K / Min-P
- Max New Tokens
- Penalty（repetition / presence / frequency）
- Sampler 类型切换（greedy / temperature / mixed）

> **注意**：当前 `Qwen3AsrTestActivity.kt` 中没有对接这个设置面板，参数完全由 `config.json` 决定。如需在测试页面中调节参数，需要调用 `updateConfig()`。

---

## 五、与旧引擎的对比

| 维度 | 旧引擎 (qwen3_asr_engine) | 新引擎 (Omni) |
|------|:--------------------------|:--------------|
| 上下文管理 | per-utterance Executor，每次新建 | Module pool 复用（prefill + decode） |
| max_new_tokens | 硬编码在 C++ 中 | config.json 可配，默认 512 |
| 采样方式 | 简单的 argmax + repetition penalty | Mixed 流水线（6 个采样器级联） |
| Temperature | 无（硬 argmax） | **有，当前 0.8** |
| KV Cache | 动态增长，无硬上限 | 2048 总 token 窗口 |
| 参数热更新 | 需要重新编译 | `updateConfig()` 运行时即时生效 |

---

## 六、建议

1. **ASR 场景可以降低 temperature 到 0.3–0.5**：转写任务需要更高的确定性，当前 0.8 偏高
2. **可以考虑用 greedy 采样**：ASR 输出不需要多样性，greedy 最快且结果稳定
3. **max_new_tokens=512 足够**：中文 ASR 输出通常 <100 tokens，无需调高
4. **repetition_penalty 建议设为 1.05**：防止 ASR 输出出现循环/重复词（当前 1.0 禁用）
