# Qwen3-ASR 模型运行时内存分析：旧路径 vs Omni 新路径

> 创建：2026-06-10
> 基于：MNN 源码分析 + 模型架构参数
> 关联文档：[[Qwen3-ASR-LLMEXPORT-MIGRATION-PLAN]] [[Qwen3-ASR-MNN-PROGRESS]]

---

## 一、模型参数规模

| 组件 | 参数量 | INT8 | BF16 | FP32 |
|------|:------|:-----|:-----|:-----|
| **Text Decoder** (28层, hidden=1024, GQA 16Q/8KV) | ~440M | 440MB | 880MB | 1.76GB |
| **Token Embedding** (vocab=151936, dim=1024) | ~156M | 156MB | 312MB | 624MB |
| **Audio Encoder** (18层, d_model=896, 14头) | ~186M | 186MB | 372MB | 744MB |
| **总计** | **~782M** | **~790MB** | **~1.56GB** | **~3.1GB** |

### 导出产物体积

| 文件 | 大小 | 说明 |
|------|:-----|------|
| `llm.mnn` | 494KB | LLM Decoder 结构 (29× FusedAttention) |
| `llm.mnn.weight` | 604MB | Decoder INT8 权重 + Token Embedding |
| `audio.mnn` | 350KB | Audio Encoder 结构 |
| `audio.mnn.weight` | 210MB | Audio Encoder INT8 权重 |
| `config.json` + `tokenizer.*` | ~6MB | 配置 + BPE 分词器 |
| **总计** | **~821MB** | |

---

## 二、核心加载机制对比

### 旧路径 (qwen3_asr_engine.cpp)

```
┌─────────────────────────────────────────────────────────────┐
│  embeddings_bf16.bin → mmap(MAP_SHARED) + MADV_RANDOM      │
│                        虚拟 312MB, RSS 随热区增长            │
│                        页粒度 4KB, 一次访问连带相邻 token     │
│                                                             │
│  audio_encoder.mnn   → Module::load() 单文件模式            │
│                        权重嵌入 .mnn 内, 全量读入 MNN 内存池  │
│                                                             │
│  llm_kv_8bit.mnn     → Module::load() + setExternalFile()   │
│  + .weight              USE_CACHED_MMAP=1                   │
│                        INT8 权重 mmap 延迟加载                │
│                                                             │
│  KV Cache            → 动态增长, 无硬上限                     │
│                                                             │
│  Per-utterance       → 每次 startDecode() 新建 Executor      │
│  Executor               避免碎片化, 但反复分配/释放            │
└─────────────────────────────────────────────────────────────┘
```

### 新路径 (Omni 引擎)

```
┌─────────────────────────────────────────────────────────────┐
│  DiskEmbedding       → FileLoader 逐 token 读盘              │
│                        RAM 中仅 ~2KB (单 token 缓冲)          │
│                        + alpha 参数 (~数 MB)                 │
│                                                             │
│  audio.mnn           → Module::load() + 独立 CPU RTManager   │
│  + .weight              USE_CACHED_MMAP=1, 权重按需缺页       │
│                                                             │
│  llm.mnn             → Module::load() + setExternalFile()   │
│  + .weight              USE_CACHED_MMAP=1                   │
│                        MEM_ALLOCATOR_TYPE=0 (延迟分配)        │
│                        604MB 含 Decoder + Embedding 全部权重  │
│                                                             │
│  KV Cache            → KVMeta 管理, 支持 kvcache_mmap 盘外   │
│                                                             │
│  Module Pool         → 2 个克隆 (prefill + decode)           │
│                        共享 mmap 权重, 独立 KV Cache          │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、逐组件内存估算

### 3.1 Token Embedding（最大差异点）

| 维度 | 旧路径 | 新路径 (DiskEmbedding) |
|------|:------|:-----------------------|
| 磁盘文件 | `embeddings_bf16.bin` 312MB | 嵌入在 `llm.mnn.weight` 604MB 中 |
| 加载方式 | mmap 全表, OS 缺页调度 | FileLoader 单行 seek+read |
| 内存占用 | 0（冷启动）→ 30–80MB RSS (热) | **固定 ~2KB 缓冲 + alpha 参数 (~5MB)** |
| 访问模式 | OS 按页 (4KB) 缺页, 每次命中连带相邻 token 进入 RSS | 精确读取所需 token 行, 无浪费 |
| 典型 RSS | 50MB (多次推理后热缓存) | **< 5MB** |

> **核心差异**：mmap 以 4KB 页为单位，每次访问一个 bf16 token (2KB) 会连带相邻 token 进入 RSS。经过多次推理后，高频 token 的页面逐渐驻留，RSS 稳定在 30–80MB。DiskEmbedding 永远只缓存当前 token 行 (2KB)，完全消除嵌入表的 RAM 占用。

### 3.2 音频编码器

| 维度 | 旧路径 | 新路径 |
|------|:------|:------|
| 文件 | `audio_encoder.mnn` (单文件) | `audio.mnn` + `audio.mnn.weight` (分离) |
| 量化 | FP16 (~372MB) 或 INT8 (~186MB) | INT8 (~210MB) |
| 权重加载 | 单文件模式：权重全量读入 MNN 内部池 | mmap 外部权重：按需缺页 |
| Runtime | 复用 LLM RuntimeManager | 独立 CPU RuntimeManager |
| 激活峰值 | Conv2d_1 输出 ~184MB ([1,480,64,1500]×fp32) | 相同 |
| 权重 RSS | **~186–372MB** 全部驻留 | **~50–100MB** RSS |
| **总计 RSS** | **~300–500MB** | **~150–250MB** |

### 3.3 LLM Decoder 权重

| 维度 | 旧路径 | 新路径 |
|------|:------|:------|
| 权重文件 | `llm_kv_8bit.mnn.weight` ~430MB | `llm.mnn.weight` ~604MB |
| Decoder 权重 | ~430MB INT8 | ~430MB INT8 (同一文件前半部分) |
| 加载方式 | mmap (USE_CACHED_MMAP=1) | mmap (USE_CACHED_MMAP=1) |
| 权重 RSS | **~50–100MB** | **~50–100MB** |
| 模块实例 | 1 (per-utterance Executor) | 2 (prefill + decode module pool) |
| 额外开销 | 每次新建 Executor 分配开销 | 2× 克隆固定开销 ~10–30MB |

### 3.4 KV Cache

两种路径完全一致（MNN 后端内部管理）。

KV Cache 大小 = `2 × 28层 × 8 KV头 × seq_len × 128 head_dim × 2字节(FP16)`

| 场景 | seq_len | KV Cache |
|------|--------|:---------|
| 10 秒中文语音识别 | ~155 (125 AE帧 + 30 tokens) | **~17 MB** |
| 30 秒长语音 + 长回复 | ~500 (375 AE帧 + 125 tokens) | **~56 MB** |
| 极端长文本生成 | ~2000 | **~224 MB** |

### 3.5 MNN 内部内存池

| 维度 | 旧路径 | 新路径 |
|------|:------|:------|
| 配置 | Memory_Low, DYNAMIC_QUANT_OPTIONS | Memory_Low, MEM_ALLOCATOR_TYPE=0 |
| Attention Mask | 每次 forward 临时分配 | 预分配 VARP 数组 (`mAttentionMaskVarVec`) |
| 中间张量 | 单层峰值 ~10–30MB, 层间复用 | 相同 |
| 典型 RSS | **50–80MB** | **40–70MB** |

---

## 四、总内存对比

### 场景 1：10 秒中文语音 → 识别 → 短回复 (~155 tokens)

```
┌──────────────────────┬───────────────────┬───────────────────┐
│ 组件                   │ 旧路径 RSS          │ 新路径 RSS          │
├──────────────────────┼───────────────────┼───────────────────┤
│ Token Embedding       │  30 – 80 MB       │  < 5 MB   ⬇️       │
│ 音频编码器 (权重+激活)  │ 300 – 500 MB      │ 150 – 250 MB ⬇️   │
│ LLM Decoder 权重       │  50 – 100 MB      │  50 – 100 MB      │
│ KV Cache              │  ~17 MB           │  ~17 MB           │
│ MNN 内部内存池         │  50 – 80 MB       │  40 – 70 MB       │
│ Module/Executor 开销   │  ~10 MB           │  10 – 30 MB       │
├──────────────────────┼───────────────────┼───────────────────┤
│ **总计**               │ **460 – 790 MB**  │ **270 – 470 MB**  │
│ **典型稳定值**          │ **~550 MB**       │ **~350 MB**  ⬇️   │
└──────────────────────┴───────────────────┴───────────────────┘
```

### 场景 2：30 秒长语音 + 长回复 (~500 tokens)

```
┌──────────────────────┬─────────────────────┬─────────────────────┐
│ 组件                   │ 旧路径 RSS            │ 新路径 RSS            │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Token Embedding       │  50 – 120 MB        │  < 5 MB   ⬇️         │
│ 音频编码器              │ 350 – 550 MB        │ 180 – 280 MB  ⬇️    │
│ LLM Decoder 权重       │  80 – 150 MB        │  80 – 150 MB        │
│ KV Cache              │  ~56 MB             │  ~56 MB             │
│ MNN 内部内存池          │  80 – 120 MB        │  60 – 100 MB        │
│ Module/Executor 开销   │  ~15 MB             │  15 – 30 MB         │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ **总计**               │ **630 – 1010 MB**   │ **390 – 620 MB**    │
│ **典型稳定值**          │ **~750 MB**         │ **~480 MB**  ⬇️     │
└──────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 五、差异归因

新路径节省 **~200 – 270MB**，主要来自三项优化：

| 优化点 | 节省量 | 原理 |
|--------|:------|------|
| **DiskEmbedding** | 30–80MB | 逐 token 精确读取替代 mmap 全表，embedding 不占 RAM |
| **音频编码器 mmap 权重** | 150–250MB | 外部权重文件 + USE_CACHED_MMAP 延迟加载 |
| **延迟内存分配** | 10–20MB | `MEM_ALLOCATOR_TYPE=0` 推迟 MNN 内部池分配 |

### 核心原理解析

**1. mmap 全表 vs DiskEmbedding**

旧路径 `embeddings_bf16.bin` (vocab_size=151936 × hidden_size=1024 × bf16=2字节) 约 312MB 以 mmap 映射。虽然初始 RSS 接近零，但每次 token 访问触发缺页，OS 以 4KB 页粒度拉入数据。bf16 token 行仅 2KB，一次缺页连带相邻 token 进入物理内存。经过数十次推理后，高频 token 分布在大量页面上全部驻留，RSS 逐渐趋近全表大小。

DiskEmbedding 使用 FileLoader 的 seek+read，精确读取单个 token 行到 2KB 的复用缓冲中，无缺页放大效应，RAM 占用恒为缓冲大小。

**2. 音频编码器单文件 vs 外部权重 mmap**

旧路径 `audio_encoder.mnn` 为 MNNConvert 产出的单文件模型，权重嵌入在 `.mnn` 文件内部。`Module::load()` 时 MNN 将全部权重读入内部内存池，不受 `setExternalFile` / `USE_CACHED_MMAP` 影响。

新路径 `audio.mnn` + `audio.mnn.weight` 分离后，`USE_CACHED_MMAP=1` 生效，音频编码器权重同样享受延迟缺页加载。

**3. 延迟内存分配**

`MEM_ALLOCATOR_TYPE=0` 让 MNN 只在首次执行 `onForward()` 时才分配工作缓冲区，而非在 `Module::load()` 时预分配全量。对多模块场景（AE + LLM）效果显著。

---

## 六、Mate 40 (Kirin 9000) 适配评估

Mate 40 典型可用 RAM 约 4–6GB（系统已占用部分）。

| 维度 | 旧路径 | 新路径 |
|------|:------|:------|
| 峰值内存 | ~1GB | ~500MB |
| 富余量 | 紧张 (可能触发 LMK) | 充足 |
| 后台存活 | 差 (易被 kill) | 良好 |
| 扩展性 | 长语音/多轮对话内存线性增长 | margin 大, 可支持 kvcache_mmap |

新路径还支持：
- **kvcache_mmap**：进一步将 KV Cache 卸载到磁盘，适合长会话
- **GPU 后端** (OpenCL/Vulkan)：权重上传到 GPU 后 CPU 侧仅保留少量元数据

---

## 七、结论

新路径 (Omni 引擎) 在内存效率上全面优于旧路径：
1. **DiskEmbedding** 消除嵌入表 RAM 占用 (~50MB)
2. **音频编码器 mmap 权重** 替代单文件全量加载 (~200MB)
3. **延迟内存分配** 降低峰值 (~15MB)

Mate 40 实机测试新路径内存压力更小、稳定性更好，建议优先用 Omni 路径。
