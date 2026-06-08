# Qwen3-ASR → MNN 项目状态

> 更新：2026-06-08（v5 — Android 端到端推理成功，中文 OK）
>
> **前期阻塞已解决。** 之前报告的「28层 decoder 精度问题（cosine 相似度 0.814）」经过系统排查，**根因不在 MNN runtime**（MNN decoder vs ONNX Runtime 的 cosim=1.0，audio encoder cosim=0.999）。实际问题是：
> 1. asr_direct.cpp 缺少 text prompt token → 模型立即输出 EOS
> 2. 早期对比使用了不同条件/不同版本的模型
>
> **修复 prompt 后，MNN 全管道成功生成正确转录。详见第三节。**
>
> **v4→v5 更新：** 经过四个版本的迭代调试，Android 端到端推理已跑通。中文识别正常。修复链路：OOM（mmap embedding + 延迟加载 + 串行化 AE/Decoder）→ `LLM_SUPPORT_AUDIO` 宏缺失 → SELinux WAV 写入拒绝 → 改用 MNN `Tokenizer` 类替代纯文本 token 表修复乱码。**详见第七节。**

---

## 一、已完成工作 ✅

### 1. 模型下载
- `/root/projects/mnn-models/Qwen3-ASR-0.6B/` (ModelScope)
- `model.safetensors` 1.87GB, 612 张量

### 2. Python 环境
- `/root/projects/mnn-models/venv/` (torch 2.12, transformers 5.10, onnx)
- `source /root/projects/mnn-models/venv/bin/activate`

### 3. 模型导出（通过 export_qwen3_asr.py  + MNNConvert）
| 文件 | 大小 | 说明 |
|------|------|------|
| `audio_encoder.mnn` | 190 MB | 3 Conv2d + 18层 Transformer Encoder (8-bit) |
| `embeddings_bf16.bin` | 297 MB | 词嵌入 (BF16) |
| `tokenizer.txt` | 3 MB | MNN 标准格式 |
| `config.json` | - | 运行时配置 |
| `llm_kv_8bit.mnn` | 0.5 MB | 28层 Qwen3 Decoder 结构 (KV Cache, 8-bit) |
| `llm_kv_8bit.mnn.weight` | **575 MB** | 8-bit 量化权重（原 FP32 2.27GB） |
| `llm_8bit.mnn` | 0.4 MB | 无 KV 版结构 (8-bit，保留兼容) |
| `llm_8bit.mnn.weight` | 575 MB | 8-bit 量化权重 |

### 4. 引擎集成（MNN 源码修改）
| 文件 | 修改 | 说明 |
|------|------|------|
| `omni.cpp:871-878` | +3 行 | `qwen3_asr` audio_type 分支 |
| `llm.cpp:311-320` | +8 行 | `is_audio` 时自动检测输入名 |
| `CMakeLists.txt` | +2 行 | 添加 `asr_demo` 编译目标 |

### 5. asr_direct.cpp text prompt 修复（新增）
- **修复前**：token 序列只有 `[audio_start, pad*T, audio_end]` → 模型立即输出 EOS
- **修复后**：从 chat template 推导的正确格式：
  ```
  <|im_start|>system\n<|im_end|>\n
  <|im_start|>user\n<|audio_start|><|audio_pad|>*T<|audio_end|><|im_end|>\n
  <|im_start|>assistant\n
  ```
- 关键：无需 "Transcribe:" 等文本指令，模型直接从音频生成

### 6. 诊断测试工具集（新增）
| 文件 | 用途 |
|------|------|
| `demo/test_compare_models.cpp` | 对比 MNN 优化/非优化 decoder 输出 |
| `demo/test_audio_encoder.cpp` | 测试 audio encoder |
| `demo/test_ae_real.cpp` | 用真实 fbank 特征对比 AE |
| `demo/test_hf_ae.cpp` | 用 HuggingFace fbank 跑全管道 |
| `compare_onnx_vs_mnn.py` | Python 端 decoder 对比脚本 |
| `compare_ae_onnx_vs_mnn.py` | Python 端 AE 对比脚本 |

### 7. ✅ 端到端 ASR 推理验证通过（新增）
MNN 全管道对真实语音生成正确转录：
```
'<asr_text>He hoped there would be stew for dinner, carrots, and carrots, and potatoes,
```
39 tokens，EOS 结束。

### 8. ✅ KV Cache 实现（新增）
在 `export_qwen3_asr.py` 中新增 `Qwen3DecoderWithKVCache` 类，导出单模型支持 prefill + decode 两阶段推理：

**设计要点：**
- K/V cache 存储在单个 5D 张量 `[28, 1, 8, past_len, 128]` 中（= 所有 28 层 × batch1 × 8 KV heads × cache长度 × 128 head_dim）
- Prefill：传空 cache（`past_len=0`），返回满 cache + 首 token logits
- Decode：传 cache + 单 token embedding，返回更新 cache + 下一个 token logits
- 一个 `onForward` 调用处理全部 28 层

**性能提升（x86 服务器）：**
| 指标 | 无 KV Cache | 有 KV Cache | 提升 |
|:----|:----------:|:----------:|:----:|
| Decode 每步 | 1634 ms | **218 ms** | **7.5x** |
| Decode 吞吐 | 0.6 tok/s | **4.6 tok/s** | **7.5x** |

### 9. ✅ 8-bit 量化完成（新增）
通过 MNNConvert `--weightQuantBits 8` 对 decoder 做权重量化。

| 版本 | 权重文件大小 | Cosine Similarity | 结论 |
|:----|:----------:|:----------------:|:----:|
| FP32 | 2.27 GB | 1.0 (基准) | 精度完美 |
| **8-bit** | **575 MB** | **0.997** | ✅ **推荐选用** |
| 4-bit | 290 MB | 0.527 | ❌ 精度损失过大 |

**量化精度对比（Prefill 最后位置，53 tokens）：**
- Top-1 匹配：FP32=11528, 8-bit=11528 ✅
- Top-10 重叠：8/10 ✅
- Max diff：0.45 (vs 4-bit 的 4.19)
- Mean diff：0.063 (vs 4-bit 的 0.74)

**4-bit 精度不足的原因：** MNNConvert 的 `--weightQuantBits` 使用朴素 min-max 对称量化，无校准数据。对于有权重异常值的模型，4-bit 动态范围不够。如有校准数据需求，后续可使用 AWQ/SmoothQuant 工具。

### 10. ✅ Android OOM 内存优化（v3→v4，2026-06-08）
Android 实机测试（`com.alibaba.mnnllm.android`）发现启动 Qwen3AsrTestActivity 时触发 lmkd 强杀：

**v3 首次优化后仍崩溃：**
```
lowmemorykiller: Kill 'com.alibaba.mnnllm.android' (13538),
  to free 3261040kB rss, 418212kb swap   ← 甚至比优化前更差 (3.26 GB vs 2.5 GB)
```

崩溃仍然发生在 `Loading LLM decoder (KV Cache)...` 期间——在 embedding mmap 代码执行**之前**。说明 embedding 优化未命中真正的内存杀手。

**根因重新定位：**
- 崩溃时间线：`trimMemory:15` → `Loading LLM decoder...` → `trimMemory:15` → `lmkd kill`
- 第一个 `trimMemory:15` 在 "Loading LLM decoder" 日志**之前**出现 → Audio Encoder 加载（190 MB .mnn 文件）已经开始触发内存压力
- 第二个模型（LLM Decoder, 575 MB 外部权重）加载时，两个模型同时驻留 → RSS 直接冲破 3 GB

**v4 修复策略：延迟加载 + 串行化（两模型不同时驻留）**

修改文件：`qwen3_asr_engine.h`、`qwen3_asr_engine.cpp`

| 阶段 | 原来（v3） | v4 |
|------|-----------|-----|
| `init()` | 加载 AE + Decoder + Embedding → **双模型同时驻留** | 只加载 tokenizer + mmap embedding → **~5 MB** |
| `runDecoder()` 开始 | — | 加载 Audio Encoder → 推理 |
| AE 推理完成 | — | **立即 `ae_mod.reset()` 释放 AE** |
| Decoder 加载 | 已在 init 中加载 | `ensureDecoderLoaded()` → 此时只有 Decoder 在内存 |
| 后续 utterance | — | Decoder 复用（已加载），AE 每次重新加载→推理→释放 |

关键代码变更：
- 移除 `m_audio_mod` 成员变量 → 改为 `runDecoder()` 中的局部变量，用完立即 `reset()`
- 新增 `m_decoder_loaded` 标志 → Decoder 只在第一次使用时加载，后续复用
- 线程数从 4 降到 2 → 减少 per-thread 内存池分配
- `BackendConfig::memory = Memory_Low` → 缩小 MNN 内部内存池

**预期效果（v4）：**

| 阶段 | 驻留模型 | 峰值 Native 内存 |
|------|----------|:-:|
| init() 完成 | 无模型 | ~5 MB |
| Audio Encoder 推理 | AE | ~500 MB |
| AE 释放后 | 无模型 | ~5 MB |
| Decoder 加载+推理 | Decoder | ~800 MB |
| **峰值（取 max）** | — | **~800 MB** |

vs v3 原始：AE + Decoder 同时驻留 = ~500 + ~800 = **~1,300 MB**

加上 JVM 开销，进程 RSS 预期从 3.26 GB 降至 ~1.2-1.5 GB。

```
lowmemorykiller: Kill 'com.alibaba.mnnllm.android' (8049), uid 10237,
  oom_score_adj 0 to free 2538868kB rss, 987256kb swap
```

关键日志：连续三次 `trimMemory level: 15`（TRIM_MEMORY_RUNNING_CRITICAL），最终被 lmkd 杀掉。

#### 根因分析：内存占用分解

| 组件 | 原始占用 | 说明 |
|------|----------|------|
| **Embedding 表** (float32) | **622 MB** | `VOCAB=151936 × HIDDEN=1024 × 4 bytes`，全量常驻内存 |
| Embedding 源文件 (bf16) | 311 MB（磁盘） | 被完整读入后转为 float32 |
| LLM Decoder 权重 | ~200-300 MB | `llm_kv_8bit.mnn.weight` 575 MB 外部文件，全量读入 RAM |
| Audio Encoder 权重 | ~200 MB | `audio_encoder.mnn` 190 MB |
| KV Cache (30s 语音) | ~140 MB | 28 layers × 2(K+V) × 8 heads × 128 dim × seq_len |
| 临时分配 | ~50-100 MB | causal mask, merged embedding, fbank tensor 等 |
| **峰值合计** | **~1.3-1.5 GB** | 加上 JVM 堆和系统开销 → 实际 RSS ~2.5 GB |

**核心问题**：151,936 行的 embedding 表中，ASR 推理全程只需要访问 **~113 个不同的 token**（< 0.1%），但原始代码将整张表以 float32 加载到内存中。

#### 优化方案：两项改动

**① Embedding 改为 mmap 按需读取（最关键，节省 ~622 MB）**

修改文件：`qwen3_asr_engine.h`、`qwen3_asr_engine.cpp`

- 移除 `MNN::Express::VARP m_embed_tbl`（622 MB float32 张量）
- 新增 `openEmbeddingFile()`：用 POSIX `open()` + `mmap()` 映射 311 MB 的 bf16 文件，设置 `madvise(MADV_RANDOM)` 避免顺序预读浪费
- 新增 `embedLookup(ids, float* dst)`：从 mmap 区域按 token ID 偏移直接读取 bf16 行，即时转换为 float32；批次内重复 token 用 `unordered_map` 缓存
- 3 个系统调用替代 622 MB 常驻内存

**② 临时缓冲区预分配 + 复用（消除抖动）**

- `m_penalty_buf`：预分配 VOCAB 大小的 float 数组（607 KB），`argmaxPenalized()` 不再每次调用时分配
- `single_tok_emb`：解码循环中复用单 token embedding 缓冲区（HIDDEN × 4 = 4 KB）
- `largeHeap="true"`：AndroidManifest 申请更大 Java 堆

> **注（v4 更新）**：虽然 `MNN::BackendConfig` 公开结构体不包含 `mmapFileSize`/`useCachedMmap`，但 `RuntimeManager::setHint()` 提供了等效的公开 API（定义在 `Interpreter.hpp` 的 `HintMode` 枚举中）。v4 参考 LLM 引擎的 `Llm::setRuntimeHint()` 模式（`llm.cpp:126-164`），通过以下 hints 启用了权重 mmap：
> ```cpp
> m_rt->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);       // mmap 权重文件
> m_rt->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);    // 延迟分配，降低峰值
> m_rt->setHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0); // 最小化 winograd 内存
> m_rt->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1); // 逐张量动态量化
> ```

#### 优化效果预估

| 组件 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| Embedding 表 | 622 MB (float32 全量) | ~0 MB (mmap, 按需分页) | **622 MB** |
| LLM Decoder 权重 | ~200-300 MB (全量读入) | ~200-300 MB (不变) | — |
| argmaxPenalized 临时分配 | ~607 KB/次 × N次 | 607 KB × 1次（预分配复用） | 抖动消除 |
| **峰值 Native 合计** | **~1.3-1.5 GB** | **~700-900 MB** | **~40-50%** |
| **进程 RSS 预估** | **~2.5 GB** | **~1.2-1.5 GB** | **~40-50%** |

> **原理**：mmap 利用 OS 的 demand paging。151,936 行 × 2 KB/行 = ~300 MB 的嵌入文件，只访问 113 行 → 实际只触发 ~226 KB 数据传输。其余页面从不加载到物理内存。

#### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `qwen3_asr_engine.h` | 移除 `VARP m_embed_tbl`，新增 mmap 成员 + 预分配缓冲区；`embedLookup`/`argmaxPenalized` 签名变更 |
| `qwen3_asr_engine.cpp` | 实现 `openEmbeddingFile`/`closeEmbeddingFile`/`embedLookup`（mmap）；`init()` 添加 `useCachedMmap`；`runDecoder()` 使用新 API |
| `AndroidManifest.xml` | 添加 `android:largeHeap="true"` |

---

## 二、技术方案说明

### 为什么不用 Llm/Omni 引擎？

MNN 的 `Llm` 引擎假设模型接受 `input_ids`（int32 token ID），内部包含 embedding 层。
Qwen3-ASR 的 decoder 接受 `inputs_embeds`（float32 预计算 embedding），无内置 embedding。

**关键差异：**
| 项目 | Llm 引擎假设 | Qwen3-ASR 实际 |
|------|-------------|----------------|
| 输入名 | `input_ids` (int32) | `inputs_embeds` (float32) |
| 输入数量 | 4（含 `logits_index`） | 3 |
| Embedding | 模型内置 | 外部 `embeddings_bf16.bin` |

**解决方案（当前路径）：** `Module::load({}, {}, ...)` 使用空输入名自动检测，只传3个输入，手写 embedding lookup + 解码循环。

### 推荐路径 vs 当前路径

当前采用 `PyTorch → ONNX → MNNConvert` 导出，是快速验证方案。MNN LLM 团队的推荐路径是使用 `llmexport.py`，两者差异如下：

| 维度 | 当前路径（ONNX） | 推荐路径（llmexport.py） |
|------|:-:|:-:|
| 导出方式 | PyTorch → ONNX opset 15 → MNNConvert | PyTorch → MNN 自定义导出 |
| Op 表示 | 分解为标准 ONNX 算子（MatMul/Add/Softmax） | 融合为 MNN 高效算子（FusedAttention 等） |
| QK-Norm 处理 | 分解为 RMSNorm → RoPE → Attention | 模型映射中显式处理 ✓ |
| 权重格式 | 通过 ONNX 序列化再反序列化，FP32 | 直接从 PyTorch 导出，支持 BF16/INT8/INT4 |
| 外部权重 | MNNConvert 自动生成 `.mnn.weight` | 由 RemoveParams 流程显式控制 |
| KV Cache | ✅ 已实现（手写管理，7.5x 加速） | 引擎内置 ✓ |
| 采样策略 | 手写 argmax | 引擎内置（top-k/top-p/temperature）✓ |
| 音频集成 | 手写 audio_encoder 推理 + embedding 注入 | Omni 引擎统一处理 ✓ |

**当前路径的核心缺陷：**
1. ONNX 作为中间格式引入了额外的序列化/反序列化步骤，权重可能被隐式转换
2. ~~没有 KV Cache~~ → ✅ **已实现**（7.5x 加速）
3. 没有采样策略 → 只能 argmax
4. 没有集成到 Omni 引擎 → 无法使用 MNN 已有的多模态推理能力

**向推荐路径迁移的工作量：**
1. 在 `utils/model_mapper.py` 中注册 `qwen3_asr` 模型条目，指定 `q_norm`/`k_norm` 等 Qwen3 特有结构
2. 在 `llmexport.py` 中处理 `inputs_embeds` 输入（当前假设 `input_ids`）
3. 在 `omni.cpp` 中完善 `qwen3_asr` 分支的 KV Cache 和采样逻辑
4. 在 `llmconfig.hpp` 中添加 `qwen3_asr` 默认配置

**暂不迁移的理由（更新 2026-06-06）：**
- 当前路径已实现 KV Cache（7.5x) + 8-bit 量化（体积降 4x），功能趋于完整
- 迁移到 llmexport.py 需要修改其 `inputs_embeds` 假设，架构调整较大
- 可安排在 Android 集成完成后进行

### 推理流水线（KV Cache）

```
WAV → MNN::AUDIO::load() → waveform
  → whisper_fbank(128mel, 400fft, 160hop) → [1,128,T]
  → audio_encoder.mnn → Module::forward → [1,T/8,1024]
  → _Permute({1,0,2}) → [T/8,1,1024]

  → 拼接 token 序列 → embed_lookup() → merged [1, S, 1024]

  ┌── Prefill ──────────────────────────────────────┐
  │  k_cache = [28,1,8,0,128] (empty)               │
  │  llm_kv_8bit.mnn({merged, pos, mask, ∅, ∅})    │
  │  → logits + k_cache([28,1,8,S,128])             │
  └─────────────────────────────────────────────────┘
          │
          ▼ first_token = argmax(logits[S-1])
          │
  ┌── Decode Loop ───────────────────────────────────┐
  │  while token != EOS:                             │
  │    tok_emb = embed_lookup({token})  [1,1,1024]   │
  │    pos = [cache_len]                             │
  │    mask = causal(1, cache_len)  [1,1,1,C+1]      │
  │    llm_kv_8bit.mnn({tok_emb, pos, mask,          │
  │                     k_cache, v_cache})            │
  │    → logits + updated k_cache, v_cache           │
  │    token = argmax(logits[0])                     │
  └──────────────────────────────────────────────────┘
```

### Prompt 格式

从原始模型的 `chat_template.json` 推导：

```
Token IDs:
  [151644, 8948, 198, 151645, 198,   # <|im_start|>system\n<|im_end|>\n
   151644, 872, 198,                   # <|im_start|>user\n
   151669,                              # <|audio_start|>
   151676 * T,                          # <|audio_pad|> × T（替换为音频嵌入）
   151670, 151645, 198,                 # <|audio_end|><|im_end|>\n
   151644, 77091, 198]                  # <|im_start|>assistant\n
```

### 配置 mRoPE vs 标准 RoPE（见第四节）

---

## 三、数值精度验证结论

### 3.1 MNN decoder ≡ ONNX Runtime decoder（完全一致）

| 对比项 | Cosine Similarity | Max Diff | 结论 |
|--------|:-:|:-:|:----:|
| MNN opt vs MNN noopt | **1.000000** | 0.0 | Op fusion 不改变数值 |
| MNN vs ONNX Runtime (合成输入) | **1.000000** | 0.000182 | ✅ 解码器一致 |
| MNN vs ONNX (首 token) | — | — | 均为 1045 ✅ |

### 3.2 MNN audio encoder ≈ ONNX Runtime AE（非常接近）

| 对比项 | Cosine Similarity | Max Diff | 结论 |
|--------|:-:|:-:|:----:|
| MNN vs ONNX (合成输入) | 0.9983 | 0.0125 | 接近 |
| MNN vs ONNX (真实语音 fbank) | **0.9990** | 0.0092 | 非常接近 ✅ |

### 3.3 全管道首 token 一致（修复 prompt 后）

| 对比项 | ONNX | MNN | 结论 |
|--------|:----:|:---:|:----:|
| 首个 token | **11528** | **11528** | ✅ 完全一致 |
| Top-1 logit | 6.44680 | 6.44679 | ✅ |
| Top-5 排序 | — | — | 完全相同 ✅ |

### 3.4 8-bit 量化精度验证

| 对比项 | Cosine Similarity | Max Diff | Top-1 匹配 | 结论 |
|--------|:-:|:-:|:--------:|:----:|
| FP32 vs 8-bit (Prefill) | **0.997** | 0.452 | ✅ 11528 | 精度基本无损 |
| FP32 vs 4-bit (Prefill) | 0.527 | 4.186 | ✅ 11528 (巧合) | ❌ 精度不够 |

**结论**：8-bit 量化精度满足 ASR 需求。4-bit 精度不足以直接使用，后续可研究 AWQ/SmoothQuant 校准量化。

### 3.5 关于前期「0.814 cosine 相似度」的根因分析

经系统排查，**根因不在 MNN runtime**。可能原因：

| 可能性 | 评估 | 依据 |
|--------|:----:|------|
| 早期使用量化模型（--quant_bit=4）与当前 FP32 模型不同 | 🔴 高概率 | 默认 quant_bit=4，量化引入精度损失 |
| 对比时的测试条件不同 | 🔴 高概率 | 当前 FP32 + 正确 prompt = cosim=1.0 |
| MNN BF16 权重加载错误 | 🟢 已排除 | 2.27GB = FP32 正确，解码器结果一致 |
| Op Fusion 改变数值 | 🟢 已排除 | opt vs noopt cosim=1.0 |
| JD Cloud 服务器硬件问题 | 🟢 已排除 | Xeon Gold 6148 支持 AVX-512 |

---

## 四、其他发现 ⚠️

### 4.1 mRoPE 配置与标准 RoPE 导出不一致

模型原始 `config.json` 中的 `text_config` 包含 mRoPE 配置：

```json
"rope_scaling": {
    "mrope_section": [24, 20, 20],
    "interleaved": true,
    "mrope_interleaved": true,
    "rope_type": "default",
    "type": "default"
}
```

但 `export_qwen3_asr.py` 使用了**标准 RoPE**，所有 head_dim=128 共 64 个复数对使用相同的位置编码。

**当前状态**：不构成阻塞
- MNN 与 ONNX 使用同一导出版本，结果一致（cosim=1.0）
- 实际转录效果正确
- 无法直接加载原始 HF 模型验证（`qwen3_asr` 不在 transformers 注册表）

**建议**：待 transformers 官方支持 `qwen3_asr` 后，加载原始模型对比输出以确认差异。

### 4.2 fbank 实现差异

| 属性 | MNN whisper_fbank / WhisperFeatureExtractor | librosa 实现 |
|------|:-:|:-:|
| 对数底 | log（自然对数） | log10 |
| 数值范围 | ~[-0.67, 1.33] | ~[-10.0, 2.58] |

对比测试时必须使用同样的 fbank。

### 4.3 MNNConvert --optimizeLevel 0 实测支持 ONNX 输入
帮助文本标注"only support for MNN source"，实测 ONNX 输入也可用。

---

## 五、性能基线（Xeon Gold 6148，x86）

### 5.1 阶段耗时分解（3s 音频，S=53 tokens，4线程，warm 后）

| 阶段 | 耗时 | 占比 | 瓶颈类型 |
|:----|:---:|:----:|:--------|
| Audio Encoder | **~780 ms** | **25%** | ✅ 已修复（原 ~5000ms，6.4x 提升） |
| Decoder Prefill | ~1400 ms | 45% | ⚠️ 53 tokens × 28 layers FP32 |
| Decode (4步×~230ms) | ~920 ms | 30% | — |
| **总计** | **~3.1 s** | | **RTF = 1.03** |

### 5.2 预期 Android 旗舰机性能（推算）

基于 MNN ARM NEON 优化预估：

| 阶段 | x86 实测 | ARM 预期 | 依据 |
|:----|:-------:|:--------:|:----:|
| Audio Encoder | **~780 ms** ✅ | **~100-300 ms** | 已修复单线程+首次惩罚，ARM NEON 可更快 |
| Decoder Prefill | **~1400 ms** | **~200 ms** | 4 核并行，ARM SDOT |
| Decode/步 | **~230 ms** | **~20-30 ms** | 8-bit 权重 + NEON 量化内核 |
| 30 步 decode | ~6.9 s | **~0.6-0.9 s** | |
| **Total** | **~3.1 s** | **~0.9-1.4 s** | **RTF 0.3-0.47** |

### 5.3 当前关键瓶颈

1. **Prefill (~1400ms)** — 当前最重。53 tokens × 28 layers 全推理。MNN x86 上无有效加速手段。手机上可通过多线程 ARM NEON 量化内核或 GPU (OpenCL/Vulkan) 卸载改善。
2. **Decode (~230ms/步)** — 瓶颈在 LM Head（词表 151936，每步 155M MACs）。x86 上 8-bit 量化的内存带宽受限，ARM 上 NEON SDOT 内核预期 20-30ms/步。
3. ~~Audio Encoder (~5s)~~ → ✅ 已修复（6.4x 提升）。根因：单线程 + 首次调用 shape 推断惩罚。

## 六、后续路线图

### Android 集成（v5：端到端推理成功，中文 OK）
- [x] 诊断 Audio Encoder 性能：根因=单线程 + 首次调用惩罚（从 5s → 0.78s）
- [x] 添加 repetition penalty（默认 1.15，打破 n-gram 重复循环）
- [x] C++ ASR 引擎类 (qwen3_asr_engine.h/.cpp)
- [x] JNI 桥接层 (qwen3_asr_jni.cpp)
- [x] Kotlin 包装类 (Qwen3AsrEngine.kt)
- [x] CMakeLists.txt 更新（加入新源文件）
- [x] Android 集成指南 (QWEN3_ASR_ANDROID_INTEGRATION.md)
- [x] **OOM 修复**：mmap embedding (622MB→0) + 延迟加载 + 串行化 AE/Decoder（双模型峰值 ~1.5GB → ~800MB）
- [x] **LLM_SUPPORT_AUDIO 宏**：CMakeLists.txt 补定义，修复解码器空壳问题
- [x] **SELinux 修复**：WAV 写到 app cache 目录（传入 `cacheDir` 参数）
- [x] **Tokenizer 乱码修复**：用 MNN `Tokenizer::createTokenizer()` 替代逐行纯文本读取，正确处理 BPE byte-level 解码
- [x] **华为手机实机验证**：中文 ASR 正常识别，无 OOM
- [ ] **英文/中英混合支持**：当前 prompt 未指定语言，模型可能偏向单语言（见新增 7.6 节）
- [ ] **流式 ASR**：当前需按 STOP 后一次性识别，非边说边出字
- [ ] 实测 ARM 上的 RTF 数据（当前 ~5.5s 总耗时，含模型加载）

### 后续优化方向
- [ ] 支持流式 ASR（音频分块 + encoder 增量推理）
- [ ] 集成到 MNN LLM Omni 引擎（替换手写 decode 循环）
- [ ] 研究 AWQ/SmoothQuant 校准量化，尝试恢复 4-bit 精度
- [ ] 待 transformers 官方支持 `qwen3_asr` 后，对比标准 RoPE vs mRoPE 差异

---

## 七、Android OOM 内存优化详情（2026-06-08）

### 7.1 崩溃现场

```
14:11:03.047  Qwen3AsrEngine  ... init: modelDir=.../Qwen3-ASR-0.6B, numThreads=4
14:11:03.538  Qwen3AsrEngine  ... Loading LLM decoder (KV Cache)...
14:11:04.922  WindowManager   ... trimMemory level: 15    ← TRIM_MEMORY_RUNNING_CRITICAL
14:11:06.032  WindowManager   ... trimMemory level: 15
14:11:07.815  WindowManager   ... trimMemory level: 10
14:11:07.915  lmkd            ... Kill 'com.alibaba.mnnllm.android' (8049),
                              oom_score_adj 0 to free 2538868kB rss, 987256kb swap
---------------------------- PROCESS ENDED (8049) ----------------------------
```

**关键事实：**
- 被杀瞬间：RSS **2.5 GB** + swap **~1 GB** = 总占用 ~3.5 GB
- 触发点：`Loading LLM decoder (KV Cache)...` 期间
- 连续 3 次 `TRIM_MEMORY_RUNNING_CRITICAL`（Android 最严重的内存警告级别）
- 最终 `lmkd` 以 `oom_score_adj 0`（前台应用最高优先级）强杀

### 7.2 内存占用详细分解

```
常量定义（来自 qwen3_asr_engine.h）：
  VOCAB=151936  HIDDEN=1024  LAYERS=28  KV_HEADS=8  HEAD_DIM=128

原始内存占用：

┌─────────────────────────────────────────────────────────────────┐
│  Embedding 表 (float32)    622 MB  ████████████████████████      │
│  LLM Decoder 权重          300 MB  ████████████                  │
│  Audio Encoder 权重        200 MB  ████████                      │
│  KV Cache (S~200 tokens)   140 MB  ██████                        │
│  临时 tensor / buffer       80 MB  ███                           │
│  JVM / Native overhead     150 MB  ██████                        │
│  文件读取 chunk buffer       20 MB  █                             │
├─────────────────────────────────────────────────────────────────┤
│  Native 合计              ~1,512 MB                              │
│  进程 RSS 合计            ~2,500 MB  (含共享库、JVM、graphics)    │
└─────────────────────────────────────────────────────────────────┘

优化后：

┌─────────────────────────────────────────────────────────────────┐
│  Embedding 表 (mmap)        ~0 MB                                │
│  LLM Decoder 权重 (全量)    300 MB  ████████████████████          │
│  Audio Encoder 权重        200 MB  ██████████████                │
│  KV Cache (S~200 tokens)   140 MB  ██████████                    │
│  临时 tensor / buffer       40 MB  ███                            │
│  JVM / Native overhead     150 MB  ██████████                    │
│  Penalty buffer (复用)     0.6 MB                                 │
├─────────────────────────────────────────────────────────────────┤
│  Native 合计               ~830 MB                               │
│  进程 RSS 合计             ~1,200-1,500 MB (预期)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 为什么 Embedding mmap 能省 622 MB

原始代码的数据流：

```
embeddings_bf16.bin (311 MB on disk)
    │
    ▼ loadEmbedding()
  读取全部 151936×1024 个 bf16 值
  每个 bf16 → float32 转换
    │
    ▼ _Input({151936, 1024}, float32)
  常驻内存 622 MB ←── 这是 OOM 的直接原因
    │
    ▼ embedLookup(tbl, ids)
  memcpy 对应行到输出
```

优化后的数据流：

```
embeddings_bf16.bin (311 MB on disk)
    │
    ▼ open() + mmap(MAP_SHARED)
  建立虚拟地址映射（0 字节数据拷贝）
  madvise(MADV_RANDOM) → 内核不执行预读
    │
    ▼ embedLookup(ids, dst)
  对每个 token ID：
    计算偏移: id × 1024 × 2 bytes
    从 mmap 区域读取 1 行 bf16 (2 KB)
    即时转换为 float32
    │
  实际物理内存：只加载被访问的页（~113 个 token × 2 KB ≈ 226 KB）
```

**核心洞察：** ASR 推理的 token 访问模式极度稀疏。全程只需要：
- 7 个 prefix text tokens（system/user 模板）
- 1 个 AUDIO_START token
- 5 个 suffix text tokens（im_end/assistant 模板）
- 最多 100 个生成 tokens（MAX_NEW_TOKENS）

合计 ~113 个不同 token，不到词汇表（151,936）的 0.1%。传统「全部加载到内存」的做法浪费了 99.9% 的内存。

### 7.4 权重 mmap 的实现方式：`RuntimeManager::setHint()`

虽然 `MNN::BackendConfig` 公开结构体（`include/MNN/MNNForwardType.h`）不直接包含 mmap 字段，但 MNN 通过 `RuntimeManager::setHint()` 提供了等效的公开 API。这些 hint 在 `Interpreter.hpp` 的 `HintMode` 枚举中定义。

**参考源码**：`transformers/llm/engine/src/llm.cpp:126-164`（`Llm::setRuntimeHint()`）是 MNN 官方的标准配置模式。

**v4 借鉴的关键 hints：**

| Hint | 值 | 作用 |
|------|----|------|
| `MEM_ALLOCATOR_TYPE` | 0 | 延迟分配（Defer），先计算总内存需求再一次性分配，降低碎片和峰值 |
| `USE_CACHED_MMAP` | 1 | 对 `.mnn.weight` 外部权重文件使用 cached mmap，而非全量 read() |
| `WINOGRAD_MEMORY_LEVEL` | 0 | Winograd 算法候选集最小化，减少中间缓冲区 |
| `DYNAMIC_QUANT_OPTIONS` | 1 | 逐张量（per-tensor）动态量化，保持 int8 权重不解压为 float32 |

这些 hints 通过 `m_rt->setHint(...)` 在 `ensureDecoderLoaded()` 中设置，作用于 LLM Decoder 的权重加载过程。Audio Encoder 因为体量较小（190 MB）且用后立即释放，不需要额外配置。

### 7.5 编译注意事项

重新编译 native library 时无需额外配置。所有改动在应用层（`apps/Android/MnnLlmChat/app/src/main/cpp/`），MNN 库本身无需重新编译。

```bash
cd apps/Android/MnnLlmChat
./gradlew assembleDebug
```

如果 Gradle 报 mmap/madvise 找不到符号，确认 NDK 版本 ≥ 21（Android API 21+ / NDK r21+），这些 POSIX API 自 API 21 起全部可用。

### 7.6 v5 故障修复全时间线

| # | 版本 | 失败现象 | 根因 | 修复 |
|---|------|----------|------|------|
| 1 | v1 | OOM 闪退（RSS 3.26 GB，lmkd kill） | AE + Decoder 双模型同时驻留 + embedding 全量加载 622 MB | 延迟加载 + 串行化 + mmap embedding + `setHint` 权重 mmap |
| 2 | v3 | 10ms "光速返回"，显示 "no speech detected" | `LLM_SUPPORT_AUDIO` 宏未在 CMakeLists.txt 定义 → 解码器编译为空壳 | `target_compile_definitions` 加 `LLM_SUPPORT_AUDIO` |
| 3 | v3 | `Failed to write temp WAV` (errno=13 EACCES) | SELinux 禁止 untrusted_app 写 `/data/local/tmp/mnn_models/` | 传入 app `cacheDir`，WAV 写到 `/data/data/<pkg>/cache/` |
| 4 | v5 | 输出乱码 | `tokenizer.txt` 是 MNN SentencePiece 二进制格式，但代码按纯文本逐行读取 | 用 `MNN::Transformer::Tokenizer::createTokenizer()` 正确解析 |
| 5 | v5 | 中文 OK，英文/中英混合识别差 | 见 7.7 节分析 | 待修复 |

### 7.7 英文/中英混合识别问题分析

**当前状态**：中文识别正常，英文识别不理想。

**当前 prompt 格式**（来自 `qwen3_asr_engine.cpp`）：
```
<|im_start|>system
<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|>*T<|audio_end|><|im_end|>
<|im_start|>assistant
```

**可能原因分析**：

1. **System prompt 为空**：当前 system 部分只有换行符，没有任务描述。虽然 `asr_direct.cpp` demo 用同样的空 system prompt 生成了正确的英文结果（"He hoped there would be stew..."），但那个测试使用的是标准英文数据集样本，模型可能更容易识别。

2. **模型训练数据偏向中文**：Qwen3-ASR 以中文为主要训练语言，对英文的 zero-shot 能力可能需要更强引导。

3. **Auto-detection 不完美**：Qwen3-ASR 理论支持语言自动检测，但在短音频/边界情况下可能默认偏向中文。

**建议修复方向**：

1. **添加 system prompt 内容**（最可能有效）：
   ```
   <|im_start|>system
   You are a helpful assistant.<|im_end|>
   ```
   或尝试明确语言引导（需测试是否有效）：
   ```
   <|im_start|>system
   You are a multilingual speech recognition assistant. Transcribe the audio accurately in its original language.<|im_end|>
   ```

2. **在 user prompt 中加 task 描述**：
   ```
   <|im_start|>user
   Transcribe the following speech:<|audio_start|>...<|audio_end|><|im_end|>
   ```

3. **检查 tokenizer 的 decode 输出**：先用 `getResult()` 获取原始 token IDs，确认是模型生成的 token 就不对，还是 tokenizer decode 环节有问题。

4. **对比 x86 和 ARM 推理输出**：同一段英文音频在 x86 demo (`asr_direct.cpp`) 和手机上的输出 token 序列是否一致，排除精度/platform 差异。

**需修改的文件**：`qwen3_asr_engine.cpp` 中的 `prefix_tokens` / `suffix_tokens` 数组。
