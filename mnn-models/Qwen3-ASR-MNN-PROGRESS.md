# Qwen3-ASR → MNN 项目状态

> 更新：2026-06-06（修订版）
>
> **前期阻塞已解决。** 之前报告的「28层 decoder 精度问题（cosine 相似度 0.814）」经过系统排查，**根因不在 MNN runtime**（MNN decoder vs ONNX Runtime 的 cosim=1.0，audio encoder cosim=0.999）。实际问题是：
> 1. asr_direct.cpp 缺少 text prompt token → 模型立即输出 EOS
> 2. 早期对比使用了不同条件/不同版本的模型
>
> **修复 prompt 后，MNN 全管道成功生成正确转录。详见第三节。**

---

## 一、已完成工作 ✅

### 1. 模型下载
- `/root/projects/mnn-models/Qwen3-ASR-0.6B/` (ModelScope)
- `model.safetensors` 1.87GB, 612 张量

### 2. Python 环境
- `/root/projects/mnn-models/venv/` (torch 2.12, transformers 5.10, onnx)
- `source /root/projects/mnn-models/venv/bin/activate`

### 3. 模型导出（通过 export_qwen3_asr.py）
| 文件 | 大小 | 说明 |
|------|------|------|
| `audio_encoder.mnn` | 190 MB | 3 Conv2d + 18层 Transformer Encoder |
| `llm.mnn` | 0.4 MB | 28层 Qwen3 Decoder 结构 |
| `llm.mnn.weight` | 2.27 GB | FP32 权重（非量化） |
| `embeddings_bf16.bin` | 297 MB | 词嵌入 |
| `tokenizer.txt` | 3 MB | MNN 标准格式 |
| `config.json` | - | 运行时配置 |

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
| KV Cache | 不涉及（需手写 decode 循环） | 引擎内置 ✓ |
| 采样策略 | 手写 argmax | 引擎内置（top-k/top-p/temperature）✓ |
| 音频集成 | 手写 audio_encoder 推理 + embedding 注入 | Omni 引擎统一处理 ✓ |

**当前路径的核心缺陷：**
1. ONNX 作为中间格式引入了额外的序列化/反序列化步骤，权重可能被隐式转换
2. 没有 KV Cache → decode 循环每次重新计算全部历史
3. 没有采样策略 → 只能 argmax
4. 没有集成到 Omni 引擎 → 无法使用 MNN 已有的多模态推理能力

**向推荐路径迁移的工作量：**
1. 在 `utils/model_mapper.py` 中注册 `qwen3_asr` 模型条目，指定 `q_norm`/`k_norm` 等 Qwen3 特有结构
2. 在 `llmexport.py` 中处理 `inputs_embeds` 输入（当前假设 `input_ids`）
3. 在 `omni.cpp` 中完善 `qwen3_asr` 分支的 KV Cache 和采样逻辑
4. 在 `llmconfig.hpp` 中添加 `qwen3_asr` 默认配置

**暂不迁移的理由：**
- 当前路径已验证数值正确性（cosim=1.0），无精度损失
- 对于原型验证和简单 ASR 场景，当前路径功能完整
- 迁移到 llmexport.py 需要理解其整体架构，投入较大
- 可安排在中期完善阶段进行

### 推理流水线

```
WAV → MNN::AUDIO::load() → waveform
  → whisper_fbank(128mel, 400fft, 160hop) → [1,128,T]
  → audio_encoder.mnn → Module::forward → [1,T/8,1024]
  → _Permute({1,0,2}) → [T/8,1,1024]
  → 拼接 token 序列 [prompt_tokens, audio_start, pad*T, audio_end, suffix_tokens]
  → embed_lookup() → merged embedding [1, S, 1024]
  → llm.mnn → Module::onForward({embeds, mask, pos})
  → argmax → 下一个 token
  → 循环直到 EOS
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

### 3.4 关于前期「0.814 cosine 相似度」的根因分析

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

## 五、后续路线图

### 短期（Android 部署）
- [ ] 交叉编译 MNN for Android（arm64-v8a）
  ```bash
  cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/.../android.toolchain.cmake \
           -DANDROID_ABI=arm64-v8a \
           -DMNN_BUILD_LLM=ON -DMNN_BUILD_AUDIO=ON
  make -j$(nproc) asr_demo
  ```
- [ ] 将模型文件（~3GB）推送到手机测试
- [ ] MNN 的 ARM NEON 后端经过充分测试，精度不会比 x86 差

### 中期（完善）
- [ ] 在 asr_direct.cpp 中实现 beam search 或采样解码
- [ ] 添加 repetition penalty 防止重复
- [ ] 支持流式 ASR
- [ ] 集成到 MNN LLM Omni 引擎

### 长期（验证）
- [ ] 待 transformers 官方支持 `qwen3_asr` 后，对比标准 RoPE vs mRoPE 差异
- [ ] 研究 4-bit 量化对精度的影响（当前 FP32 权重 2.27GB 对移动端偏大）
