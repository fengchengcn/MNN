# Qwen3-ASR Omni 模式推理参数

> 创建：2026-06-10 | 更新：2026-06-11（greedy 采样已确认为 ASR 最优配置）
> 关联：[[Qwen3-ASR-OMNI-STREAMING-PLAN]]

## 上下文长度

| 参数 | 值 | 来源 |
|------|------|------|
| `max_all_tokens` | 2048 | `llmconfig.hpp` 默认 |
| `max_new_tokens` | 512 | `llmconfig.hpp` 默认 |

ASR 场景远未触及上限：30 秒长语音 + 满额输出仅用 ~44% 窗口。

## 当前采样配置（ASR 优化后）

2026-06-11 从 `mixed` 改为 `greedy`——ASR 是确定性任务，不需要随机性：

```json
{
  "sampler_type": "greedy",
  "temperature": 0.0,
  "top_k": 1,
  "top_p": 1.0,
  "min_p": 0.0,
  "n_gram": 0,
  "repetition_penalty": 1.0
}
```

### 为什么 ASR 必须用 Greedy

- 罕见 token（技术术语、英文缩写）在模型中的置信度仅 30-70%
- 任何非零 temperature 引入的随机噪声足以让模型选择错误 token
- Mixed sampling 的 tfs/typical/topP 过滤链可能误杀低频但正确的 token
- Greedy argmax 确保同段音频每次输出一致，适合 ASR

## Mixed Sampler 流水线（原配置，已废弃供参考）

```
penalty（置前）→ topK → tfs → typical → topP → min_p → temperature
```

## 运行时动态修改参数

```kotlin
// 修改完整配置
llmSession?.updateConfig("""{
    "temperature": 0.6,
    "max_new_tokens": 1024
}""")
```

> Qwen3AsrTestActivity 未对接 Settings 面板，参数由 `config.json` 决定。

## 旧引擎 vs Omni 对比

| 维度 | 旧引擎 | Omni |
|------|--------|------|
| 上下文管理 | per-utterance Executor | Module pool 复用 |
| 采样 | 硬 argmax + rep_penalty | Mixed 流水线 / Greedy |
| KV Cache | 动态增长，无硬上限 | 2048 token 窗口 |
| 参数热更新 | 需重新编译 | `updateConfig()` 即时生效 |
