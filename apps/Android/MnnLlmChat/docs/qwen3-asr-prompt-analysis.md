# Qwen3-ASR-0.6B Prompt 结构分析

> 分析日期：2026-06-09
> 分析对象：`apps/Android/MnnLlmChat` 中 Qwen3-ASR-0.6B 的推理提示词

## 核心结论

Qwen3-ASR-0.6B 是一个**纯 ASR（语音识别）模型**，本质是音频 → 文本的端到端模型。
提示词（Prompt）在 C++ 推理引擎内部硬编码构建，采用 **ChatML 格式**，对 Android 用户不可见。
系统提示词只有一句话：**`"You are a helpful assistant."`**

---

## 一、提示词构建入口

提示词构建发生在引擎初始化阶段，调用链如下：

```
Qwen3AsrEngine.init()
  └─ buildPromptTokens()          // qwen3_asr_engine.cpp:253
       ├─ 加载 tokenizer
       ├─ 编码 system message
       └─ 构建 prefix_tokens / suffix_tokens
```

具体代码位置：`app/src/main/cpp/qwen3_asr_engine.cpp:253-285`

---

## 二、完整 Token 序列

推理时，最终的 Token 序列由三部分拼接而成：

```
prefix_tokens  +  AUDIO_START  +  [AUDIO_PAD × T帧]  +  suffix_tokens
```

其中 `T` 为音频编码器输出的帧数。

展开为 ChatML 格式（人可读）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|> × T<|audio_end|><|im_end|>
<|im_start|>assistant
```

---

## 三、Token 构成详解

### 3.1 Prefix Tokens（前缀）

```cpp
// qwen3_asr_engine.cpp:266-277
m_prefix_tokens = {151644, 8948, 198};  // <|im_start|>system\n
auto sys_msg = tok->encode("You are a helpful assistant.");
m_prefix_tokens.insert(..., sys_msg);   // 系统消息内容
m_prefix_tokens.push_back(198);         // \n
m_prefix_tokens.push_back(151645);      // <|im_end|>
m_prefix_tokens.push_back(198);         // \n
m_prefix_tokens.push_back(151644);      // <|im_start|>
m_prefix_tokens.push_back(872);         // user
m_prefix_tokens.push_back(198);         // \n
```

### 3.2 Suffix Tokens（后缀）

```cpp
// qwen3_asr_engine.cpp:279-282
m_suffix_tokens = {151670,          // <|audio_end|>
                   151645, 198,     // <|im_end|>\n
                   151644, 77091, 198}; // <|im_start|>assistant\n
```

### 3.3 Special Token ID 速查表

| Token ID | 符号 | 说明 |
|----------|------|------|
| `151643` | `<\|endoftext\|>` | EOS，解码终止 |
| `151644` | `<\|im_start\|>` | ChatML 消息开始 |
| `151645` | `<\|im_end\|>` | ChatML 消息结束 |
| `151669` | `<\|audio_start\|>` | 音频起始标记 |
| `151670` | `<\|audio_end\|>` | 音频结束标记 |
| `151676` | `<\|audio_pad\|>` | 音频填充占位符（会被 AE 输出替换） |
| `8948` | `system` | system 角色 |
| `872` | `user` | user 角色 |
| `77091` | `assistant` | assistant 角色 |

---

## 四、Fallback 路径（无 Tokenizer 时）

当 `tokenizer.txt` 文件缺失时，走硬编码 fallback，此路径下 **system 消息为空**：

```cpp
// qwen3_asr_engine.cpp:257-262
m_prefix_tokens = {151644, 8948, 198, 151645, 198,   // <|im_start|>system\n<|im_end|>\n
                   151644, 872, 198};                 // <|im_start|>user\n
m_suffix_tokens = {151670,                            // <|audio_end|>
                   151645, 198,                       // <|im_end|>\n
                   151644, 77091, 198};               // <|im_start|>assistant\n
```

即等价于：

```
<|im_start|>system
<|im_end|>
<|im_start|>user
<|audio_start|>...<|audio_end|><|im_end|>
<|im_start|>assistant
```

---

## 五、两个推理路径的 Prompt 使用

### 5.1 批量模式（`runDecoder()`）

- 调用链：`endAudio()` → `runDecoder()`（`qwen3_asr_engine.cpp:673`）
- 使用 `m_prefix_tokens` + `AUDIO_START` + `AUDIO_PAD × T` + `m_suffix_tokens`
- 完整 decode loop，返回最终结果

### 5.2 流式模式（`startDecode()` / `decodeStep()`）

- 调用链：`startDecode()` → `decodeStep()` 循环（`qwen3_asr_engine.cpp:380-625`）
- 使用**完全相同的** prefix/suffix token 模板
- 区别：decode loop 可通过 `decodeStep()` 逐 token 获取（非阻塞），支持实时部分结果显示

两种模式下，prompt 结构完全一致。

---

## 六、Android 层调用关系

```
┌── Qwen3AsrTestActivity.kt ───┐    独立 ASR 测试 Activity
│  engine.init(dir, cache, 4)   │    直接调用 Qwen3AsrEngine
│  engine.pushAudio(floatBuf)   │    无后续 LLM 对话
│  engine.getResultText()       │
└───────────────────────────────┘

┌── VoiceChatPresenter.kt ─────┐    语音对话模式
│  qwen3AsrEngine.init(...)     │    ASR 识别结果 → chatPresenter.sendMessage(text)
│  → 识别文本                   │    → 后续 LLM 对话（由用户配置的 system prompt 决定）
└───────────────────────────────┘
```

**关键区别**：
- `Qwen3AsrTestActivity` 只是 ASR 测试，识别文本直接显示，不经过 LLM
- `VoiceChatPresenter` 将 ASR 结果传递给 `chatPresenter.sendMessage(text)`，再由用户配置的 LLM 对话 prompt 处理（这是另一套 prompt 体系）

---

## 七、模型文件依赖

引擎初始化需要的文件（放在 `/data/local/tmp/mnn_models/Qwen3-ASR-0.6B/`）：

| 文件名 | 用途 |
|--------|------|
| `audio_encoder.mnn` | 音频编码器（Whisper-style fbank → 声学特征） |
| `llm_kv_8bit.mnn` | LLM 解码器（8-bit KV cache quantized） |
| `llm_kv_8bit.mnn.weight` | 外部权重文件 |
| `embeddings_bf16.bin` | Token embeddings（bf16，mmap 加载，零 RAM 占用） |
| `tokenizer.txt` | Tokenizer 词表（用于解码 + 编码 system prompt） |

---

## 八、关键设计决策记录

1. **System prompt 硬编码**：`"You are a helpful assistant."` 写在 `buildPromptTokens()` 中，非配置文件控制。若需要支持多语言或多场景，需改为参数化。
2. **Tokenizer 依赖性**：System prompt 的 token 化依赖 `tokenizer.txt` 文件。若文件缺失，回退到**空 system message** 的硬编码 fallback。
3. **ChatML 格式固定**：Qwen3 系列标准 ChatML 模板，不支持运行时切换格式。
4. **Repetition penalty = 1.15**：解码时使用 `REP_PENALTY = 1.15`，max new tokens = 100。
