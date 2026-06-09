# Qwen3-ASR → llmexport.py 迁移计划

> 创建：2026-06-09
> 基于：MNN 框架源码验证（HiAI/OpenCL/Vulkan 后端 + llmexport.py/omni.cpp/llm.cpp）
> 关联文档：[[Qwen3-ASR-STREAMING-PLAN]] [[Qwen3-ASR-MNN-PROGRESS]]
>
> **状态：已实施 (2026-06-09)**

---

## 一、迁移动机

当前 Qwen3-ASR 使用 `export_qwen3_asr.py` → ONNX → MNNConvert 导出路径，存在两个结构性问题：

1. **分解算子阻塞 GPU 加速**：模型中的 Attention 被分解为 MatMul + Add + Softmax 逐 op，无法利用 MNN OpenCL/Vulkan 后端已有的 `OpType_Attention` 融合 Attention kernel。即使 GPU 后端已实现该 kernel，模型格式不支持。

2. **未集成进 MNN Omni 引擎**：手写 decode 循环无法使用引擎内置的采样策略（top-k/top-p/temperature）、prefix caching、以及 `llm.cpp` 的多模态推理管线。

### 收益预估

| 维度 | 当前路径（ONNX 分解） | 迁移后（llmexport.py 融合） | 提升 |
|------|------|------|:--:|
| **CPU 推理** | ~20 tok/s (Phase 3 已验证) | ~22-24 tok/s（内存布局优化） | +10-20% |
| **GPU (Vulkan) 推理** | 不可用（分解算子阻塞） | 预期 ~40-60 tok/s | 2-3x |
| **GPU (OpenCL) 推理** | 不可用（分解算子阻塞） | 预期 ~30-50 tok/s | 1.5-2.5x |
| **采样策略** | 仅 argmax | top-k/top-p/temperature/min_p/... | 质量提升 |
| **Prefill 延迟** | ~500-700ms | ~200-400ms (FusedAttention + GPU) | 2-3x |
| **权重体积** | 575 MB (MNNConvert 8-bit) | ~300-400 MB (BF16 + MNN 内置量化) | -30-50% |
| **代码维护** | C++ 手写 decode loop (~300 行) | 引擎内置（0 行） | 大幅简化 |

---

## 二、前置条件验证

基于 2026-06-09 对 MNN 源码的实际验证，以下关键基础设施**已经存在**，无需从零开发：

| 基础设施 | 位置 | 状态 |
|----------|------|:--:|
| MNN OpenCL `OpType_Attention` fused kernel | `source/backend/opencl/execution/buffer/AttentionBufExecution.cpp:1854` | ✅ 已注册为 TRANSFORMER |
| MNN Vulkan `VulkanAttention` fused kernel | `source/backend/vulkan/buffer/execution/VulkanAttention.cpp` | ✅ 含 KV cache + FP16 |
| MNN OpenCL `RMSNorm` (LayerNorm with RMSNorm flag) | `source/backend/opencl/execution/buffer/LayerNormBufExecution.cpp:26,165` | ✅ |
| llmexport.py `FusedAttention` custom ONNX op | `transformers/llm/export/utils/custom_op.py:36-68` | ✅ |
| llmexport.py `FakeLinear` 权重卸载 | `transformers/llm/export/utils/custom_op.py:5-34` | ✅ |
| llm.cpp `is_audio()` → 自动检测 `inputs_embeds` | `transformers/llm/engine/src/llm.cpp:316-317` | ✅ |
| omni.cpp `qwen3_asr` audio_type 分支（whisper_fbank） | `transformers/llm/engine/src/omni.cpp:876-878` | ✅ |
| omni.cpp 音频嵌入注入（mAudioEmbeddings） | `transformers/llm/engine/src/omni.cpp:1176-1185` | ✅ |
| llmconfig.hpp `is_audio()` / `audio_type()` | `transformers/llm/engine/src/llmconfig.hpp:260-266` | ✅ |

> **核心洞察**：MNN 框架的 Omni 引擎已经为 `qwen3_asr` 开了分支，fbank 处理已实现。缺失的是 Python 侧的模型注册、导出适配，以及 C++ 侧的文本 embedding 注入逻辑。工程量比原预期小得多。

---

## 三、工作包总览

```
WP1: 模型注册 (model_mapper.py)          ~0.5 day
  └─→ WP2: 模型加载 (model.py)           ~0.5 day
       └─→ WP3: 模型适配 (transformers.py)  ~0.5 day
            └─→ WP4: 导出适配 (llmexport.py) ~0.5 day
                 └─→ WP5: C++ 引擎集成 (omni.cpp) ~1 day
                      └─→ WP6: 端到端验证            ~1 day
─────────────────────────────────────────────────────────
                        总计:                       ~4-5 days
```

### 涉及文件清单

| 文件 | 工作包 | 变更类型 |
|------|:--|------|
| `transformers/llm/export/utils/model_mapper.py` | WP1 | 新增 `regist_qwen3asr()` 方法 |
| `transformers/llm/export/utils/model.py` | WP2 | `MODEL_CLASS_MAPPING` 新增条目 |
| `transformers/llm/export/llmexport.py` | WP2, WP4 | 加载分支 + 导出适配 + export_audio() |
| `transformers/llm/export/utils/transformers.py` | WP3 | 验证 Q/K-Norm 兼容性（预计无需修改） |
| `transformers/llm/engine/src/omni.cpp` | WP5 | 完善 qwen3_asr 文本 embedding 注入 |
| `transformers/llm/engine/src/llmconfig.hpp` | — | 无需修改（已通用） |
| 上述全部 | WP6 | 端到端测试验证 |

---

## 四、WP1: 模型注册 — model_mapper.py

**修改文件**：`transformers/llm/export/utils/model_mapper.py`

在 `ModelMapper.__init__()` 中调用新方法 `regist_qwen3asr()`，注册 Qwen3-ASR 的模型字段映射。

```python
def regist_qwen3asr(self):
    # Qwen3-ASR 模型结构：HuggingFace 中 audio_encoder + text_decoder 在同一个 model 下
    # ModelScope 路径: Qwen/Qwen3-ASR-0.6B
    # config.json 包含: text_config + audio_config
    qwen3asr_config = {
        'hidden_size':    'text_config.hidden_size',
        'head_dim':       'text_config.head_dim',
        'num_attention_heads':   'text_config.num_attention_heads',
        'num_hidden_layers':     'text_config.num_hidden_layers',
        'num_key_value_heads':   'text_config.num_key_value_heads',
        'rope_theta':     'text_config.rope_theta',
        'rope_scaling':   'text_config.rope_scaling',
        'max_position_embeddings': 'text_config.max_position_embeddings',
        'attention_type': 'text_config.attention_type',  # qwen3 用 GQA
        # Audio-specific configs (写入 llm_config.json 供 C++ 读取)
        'audio_type':     'qwen3_asr',
        'is_audio':       True,
        'audio_pad':      151676,   # <|audio_pad|>
        'audio_start':    151669,   # <|audio_start|>
        'audio_end':      151670,   # <|audio_end|>
    }
    qwen3asr_model = {
        'lm':     'lm_head',               # LM head (token prediction)
        'embed':  'model.embed_tokens',    # Text embedding layer
        'blocks': 'model.layers',          # 28 × Decoder layers
        'final_layernorm': 'model.norm',   # Final RMSNorm
        'audio':  'audio_encoder',         # Audio encoder (3xConv2d + 18xTransformer)
    }
    qwen3asr_attention = {
        'q_proj': 'self_attn.q_proj',
        'k_proj': 'self_attn.k_proj',
        'v_proj': 'self_attn.v_proj',
        'o_proj': 'self_attn.o_proj',
        'q_norm': 'self_attn.attention.q_norm',  # Qwen3 特有: Q/K-Norm
        'k_norm': 'self_attn.attention.k_norm',
    }
    qwen3asr_map = {
        'config':    qwen3asr_config,
        'model':     qwen3asr_model,
        'decoder':   self.default_decoder,
        'attention': qwen3asr_attention,
    }
    self.regist('qwen3_asr', qwen3asr_map)
```

**关键决策**：
- `attention_type`：Qwen3-ASR 使用 GQA（28 heads Q, 4 heads KV），`num_key_value_groups = 28//4 = 7`
- `audio` 字段映射到 `audio_encoder`（不是 `audio_tower`）→ 后续 `is_audio()` 时通过 `audio_model()` 加载
- `is_audio: True` 写入 llm_config → C++ 引擎自动走 `inputs_embeds` 路径

**验证标准**：
- [ ] `LlmModel.from_pretrained('Qwen/Qwen3-ASR-0.6B')` 不报错
- [ ] `model.audio` 正确指向 audio_encoder
- [ ] `model.blocks` 包含 28 层 decoder
- [ ] `model.embed` 正确引用 `model.embed_tokens`

---

## 五、WP2: 模型加载 — model.py + llmexport.py

**修改文件**：
1. `transformers/llm/export/utils/model.py` — `get_model_class()`
2. `transformers/llm/export/llmexport.py` — `load_model()`

Qwen3-ASR 的 HF model class 不在标准 transformers 库中（属于自定义模型），需要类似 `lfm2_audio` 的处理方式，用 `trust_remote_code=True` 加载。

**model.py**：
```python
MODEL_CLASS_MAPPING = {
    # ... 现有条目保持不变 ...
    'qwen3_asr': None,   # Sentinel: 自定义加载路径
}
```

**llmexport.py load_model() 扩展**：
```python
if self.model_type == 'qwen3_asr':
    from transformers import AutoModelForCausalLM
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto'
    )
```

**验证标准**：
- [ ] 无需修改 transformers 源码即可加载模型
- [ ] `config.model_type = 'qwen3_asr'`
- [ ] `config.text_config.hidden_size = 1024`
- [ ] `config.text_config.num_hidden_layers = 28`
- [ ] audio encoder 权重正确加载到 `model.audio`

---

## 六、WP3: 模型适配 — transformers.py

**修改文件**：`transformers/llm/export/utils/transformers.py`

Qwen3-ASR Decoder Layer 结构：

```
Qwen3 Decoder Layer:
├── input_layernorm (RMSNorm)
├── self_attn
│   ├── q_proj, k_proj, v_proj  (GQA: 28 Q-heads, 4 KV-heads)
│   ├── q_norm, k_norm          (Q/K RMSNorm — Qwen3 特有)
│   ├── RoPE (通过 Rotary 类)
│   ├── FusedAttention           (llmexport.py 融合路径)
│   └── o_proj
├── post_attention_layernorm (RMSNorm)
└── mlp
    ├── gate_proj, up_proj       (SwiGLU)
    └── down_proj
```

**当前 Attention 类已支持**（无需修改）：
- ✅ GQA (`num_key_value_groups`)
- ✅ `q_norm`/`k_norm` 在 RoPE 前应用（`qk_norm_after_rope = False`）
- ✅ `FusedAttention` 融合路径 (`export_fused_attn = True`)
- ✅ `FakeLinear` 权重卸载

**可能需要验证/调整**：
1. **Q/K-Norm 类型**：确认是 `RMSNorm` 还是标准 `LayerNorm`。Qwen3 使用 `RMSNorm`
2. **Position IDs**：Audio frames 需要正确的 position IDs，由 C++ 端在推理时计算
3. **Embedding 输入格式**：`Embedding.forward()` 返回 `view(-1, 1, hidden_size)`，兼容 `inputs_embeds` 格式

**验证标准**：
- [ ] `model.blocks[i].self_attn.export_fused_attn = True` 正常执行
- [ ] FusedAttention 在 ONNX trace 中正确生成 `LlmExporter::FusedAttention` op
- [ ] Q/K-Norm 在 RoPE 之前正确应用

---

## 七、WP4: 导出适配 — llmexport.py

**修改文件**：`transformers/llm/export/llmexport.py`

### 7.1 dynamic_axes 适配

```python
# qwen3_asr 使用 inputs_embeds 而非 input_ids 作为第一输入
if self.model_type == 'qwen3_asr':
    self.model_dynamic_axes = {
        "inputs_embeds" : { 0: "seq_len" },
        "attention_mask" : { 2: "seq_len", 3: "seq_len" },
        "position_ids" : { 1: "seq_len" },
    }
```

### 7.2 ONNX 导出适配

当前代码 `export_onnx()` 第 488 行已做 `input_ids = model.embedding(input_ids)`（token → embedding），对 qwen3_asr 可直接复用，仅需将 input name 改为 `inputs_embeds`。C++ 端 `inputNames = {}`（auto-detect）已支持自动识别 float32 输入。

### 7.3 config.json 生成

```python
if self.model_type == 'qwen3_asr':
    config.update({
        'is_audio': True,
        'audio_type': 'qwen3_asr',
        'audio_model': 'audio_encoder.mnn',
        'audio_pad': 151676,
        'audio_start': 151669,
        'audio_end': 151670,
        'system_prompt': 'You are a helpful assistant.',
    })
```

### 7.4 音频编码器导出（新增 `export_audio()`）

```python
def export_audio(self):
    if self.audio is None:
        return
    audio_onnx = self.audio.export(self.onnx_path)
    if self.mnn_converter:
        MNNConverter(self, None).export(audio_onnx,
                                         quant_bit=0,           # AE 不量化（精度关键）
                                         transformer_fuse=True)  # AE 内部 Attention 可融合
```

**导出产物**：

| 文件 | 说明 |
|------|------|
| `llm.mnn` + `llm.mnn.weight` | LLM Decoder with `OpType_Attention` |
| `audio_encoder.mnn` | Audio Encoder (3xConv2d + 18xTransformer) |
| `embeddings_bf16.bin` | Text embeddings (151936 × 1024, BF16) |
| `tokenizer.mtok` | MNN 标准 Tokenizer |
| `llm_config.json` | 含 `is_audio:true, audio_type:qwen3_asr` |
| `config.json` | 含 `system_prompt, audio_pad/start/end` |

---

## 八、WP5: C++ 引擎集成 — omni.cpp

**修改文件**：`transformers/llm/engine/src/omni.cpp`

### 8.1 已实现部分（无需修改）

```
omni.cpp:876-878  → whisper_fbank()  (audio_type == "qwen3_asr")
omni.cpp:1176     → AUDIO_PAD 位置注入 mAudioEmbeddings
omni.cpp:152      → mAudioModule 加载 (audio_encoder.mnn)
```

### 8.2 需要实现：Text Prompt Token 嵌入注入

将当前手写代码的 prompt 构建逻辑（`qwen3_asr_engine.cpp:253-285`）迁移到 omni.cpp 的 `prefillForGraph()`：

```cpp
// 构建完整的 token 序列
std::vector<int> tokens;
// 1. System prefix: <|im_start|>system\n + tokenize("You are a helpful assistant.") + <|im_end|>\n<|im_start|>user\n
tokens.insert(tokens.end(), mPrefixTokens.begin(), mPrefixTokens.end());
// 2. Audio: AUDIO_START + AUDIO_PAD × T + AUDIO_END
tokens.push_back(mAudioStart);
tokens.insert(tokens.end(), T, mAudioPad);
tokens.push_back(mAudioEnd);
tokens.push_back(mImEnd);
// 3. Assistant: <|im_start|>assistant\n
tokens.push_back(mImStart);
tokens.push_back(mAssistantId);
```

然后将文本 token 通过 `Llm::embedding()` → embeddings，在 AUDIO_PAD 位置替换为 audio encoder 输出，生成 merged embeddings 传入 prefill。

### 8.3 Executor/Module 加载

Qwen3-ASR 的 Audio Encoder 和 LLM Decoder 使用不同的 MNN Module。omni.cpp 已支持此模式（`mAudioModule` + `mModule`），无需额外修改。

**验证标准**：
- [ ] Audio encoder 加载成功
- [ ] `whisper_fbank()` 正确生成 mel spectrogram
- [ ] Audio encoder 输出 shape = `[T_audio_frames, 1, 1024]`
- [ ] Token 序列正确构建（含 system prompt + audio tokens）
- [ ] Text embeddings 正确注入到 audio pad 位置
- [ ] Merged embeddings → prefill → first token 正确
- [ ] Decode loop 使用引擎内置采样策略

---

## 九、WP6: 端到端验证

### 9.1 x86 服务器验证

```bash
# Step 1: 导出 Qwen3-ASR 模型
cd transformers/llm/export
python llmexport.py \
    --path /path/to/Qwen3-ASR-0.6B \
    --export mnn \
    --dst_path ./Qwen3-ASR-MNN \
    --quant_bit 8 \
    --transformer_fuse

# Step 2: 验证导出产物
ls ./Qwen3-ASR-MNN/
# 期望: llm.mnn + llm.mnn.weight + audio_encoder.mnn
#       + embeddings_bf16.bin + tokenizer.mtok
#       + llm_config.json + config.json

# Step 3: 运行 LLM demo
cd build
./llm_demo ./Qwen3-ASR-MNN/config.json test.wav
```

### 9.2 Android 实机验证（Mate 40, Kirin 9000）

```bash
# 推送模型
adb push Qwen3-ASR-MNN /data/local/tmp/mnn_models/Qwen3-ASR-MNN/

# 编译安装
cd apps/Android/MnnLlmChat
./gradlew assembleDebug
adb install app/build/outputs/apk/googleplay/debug/app-googleplay-debug.apk

# 性能对比期望:
# CPU (Phase 3 baseline): ~20 tok/s
# Vulkan GPU (目标):       ~40-60 tok/s
# CPU FusedAttention:       ~22-24 tok/s
```

### 9.3 验证清单

| 验证项 | 环境 | 标准 |
|--------|:--|------|
| 导出成功，无报错 | x86 | 6 个文件生成 |
| llm_demo 端到端推理 | x86 | 正确转写中文 |
| Audio encoder 输出一致 | x86 | cosim > 0.999 vs 当前路径 |
| Decoder 首 token 一致 | x86 | token ID 匹配当前路径 |
| FusedAttention op 存在 | x86 | MNN 模型不含分解 MatMul |
| APK 编译成功 | Android | assembleDebug 无报错 |
| CPU 模式正常 | Mate 40 | 中文识别正确，~22 tok/s |
| Vulkan GPU 模式正常 | Mate 40 | 中文识别正确，~40-60 tok/s |
| OpenCL GPU 模式正常 | Mate 40 | 中文识别正确（可接受偶尔 fallback）|
| 多轮 utterance 无崩溃 | Mate 40 | 5 轮以上稳定 |
| 内存 < 1.5GB RSS | Mate 40 | 无 lmkd kill |

---

## 十、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|------|------|
| Qwen3-ASR 的 HF 格式不兼容 llmexport.py 加载器 | 中 | 阻塞 WP2 | 参考 lfm2_audio 自定义加载模式；备选：复用 `export_qwen3_asr.py` 的加载逻辑 |
| FusedAttention 在 Qwen3-ASR 上精度异常 | 低 | 识别质量下降 | 对比 FusedAttention vs 分解 MatMul 的 cosine similarity |
| Audio encoder MNN 转换失败 | 低 | 阻塞 WP4 | AE 导出已在 `export_qwen3_asr.py` 中验证通过，可复用 |
| Mali Vulkan 驱动处理 28 层模型时崩溃 | 中 | GPU 不可用 | 回退 CPU 路径；Phase 3 CPU 已充分优化 |
| `inputs_embeds` auto-detect 失败 | 低 | Decoder 加载失败 | 显式指定 inputNames（已有 fallback 机制） |

---

## 十一、回滚策略

采用**双轨并行**策略，新旧路径互不干扰：

1. 保留当前 `qwen3_asr_engine.cpp` 全部代码不动
2. 新路径在 `omni.cpp` 中独立实现
3. Android 端通过 config flag 切换：
   - `use_omni_engine=false` → 走当前手写路径（已充分验证）
   - `use_omni_engine=true` → 走 llmexport.py 融合路径（新功能）

出问题立即可回退，零风险。
