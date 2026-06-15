---
date: 2026-06-14
status: superseded
tags: [qwen3-asr, plan, sherpa-onnx, mnnconvert, omni-engine, audio-encoder]
category: plan
aliases: [Sherpa AE 集成方案, MNNConvert 替换 AE]
related: [[root-cause-analysis]], [[fbank-numerical-analysis]], [[progress]], [[replicate-onnx-export-plan]]
---

# Sherpa-ONNX AE → MNN 集成方案

> 目标：用 MNNConvert 转换 sherpa-onnx 的 conv_frontend.onnx + encoder.int8.onnx 替换 llmexport.py 手写 forward() 导出的 audio.mnn，使 MNN AE 与 sherpa-onnx **完全架构等价**。
>
> **状态更新（2026-06-15）**：此方案曾成功部署（MNNConvert 转换 Wasser1462 ONNX → MNN 双模型），是早期快速验证和生产过渡的关键步骤。现已**被 llmexport.py 自控双模型导出替代**（[[replicate-onnx-export-plan]]），不再依赖第三方 ONNX 文件。Wasser1462 ONNX 文件仅保留用于调试对照。

## 背景

[[root-cause-analysis]] 确认 MNN llmexport.py 的 `Qwen3ASRAudioEncoder.forward()` 与 sherpa-onnx 的 conv_frontend + encoder **图结构不等价**（cosim ~0.30）。深入追踪后发现差异远超 Pad/Slice —— Transpose perm 序列、tensor layout、Conv2d 作用维度均不同。手写修复需要逆向 130 节点 ONNX 动态图，投入巨大且不保证正确。

**方案三**直接使用 MNNConvert 转换 sherpa 的 ONNX 模型，已验证数值一致性：

| 模型 | ONNX → MNN | vs ONNX cosim | 判定 |
|------|:----------:|:-------------:|:----:|
| `conv_frontend.mnn` | 42 MB | **1.0000** | ✅ 完全一致 |
| `encoder.mnn` | 176 MB | **0.9968** | ✅ 高度一致 |
| **合计** | **218 MB** | | 对比手写 audio.mnn 的 0.30 |

转换命令：
```bash
MNNConvert -f ONNX --modelFile conv_frontend.onnx --MNNModel conv_frontend.mnn
MNNConvert -f ONNX --modelFile encoder.int8.onnx   --MNNModel encoder.mnn
```

## 架构对比

### 当前架构（llmexport audio.mnn）

```
FBank [1, 128, T]  ──→  audio.mnn (手写 forward, 单文件)  ──→  [1, T', 1024]  ──Permute──→  Decoder
                                ↑
                          Conv2d×3 + 18×Transformer
                          (Pad/Slice 缺失, subsample 8×)
```

### 目标架构（sherpa MNN AE）

```
FBank [1, 128, T]
    │
    │ _Permute({0,2,1})   ← 【改动 1】维度转置: [1,128,T] → [1,T,128]
    ▼
conv_frontend.mnn         ← 【改动 2】新模型 (Pad→Conv×3→Slice, subsample ~6.5×)
    │
    │ [1, T', 896]
    ├──→ 提取 seq_len = T'
    │    创建 mask [1, T'] (all 1)   ← 【改动 3】1D feature mask
    ▼
encoder.mnn               ← 【改动 2】新模型 (18×Transformer, INT8)
    │
    │ [1, T', 1024]
    │ _Permute({1,0,2})   ← 【已有】omni.cpp:968
    ▼
Decoder (保持不变)
```

### I/O 规格

| 模型 | 输入 | 输出 |
|------|------|------|
| `conv_frontend.mnn` | `input_features` [1, T, 128] float32 | `conv_output` [1, T', 896] float32 |
| `encoder.mnn` | `input_features` [1, T', 896] float32<br>`feature_attention_mask` [1, T'] bool (1D 特征掩码) | `audio_features` [1, T', 1024] float32 |
| 最终送入 Decoder | | [T', 1, 1024]（permute 后） |

## Omni 引擎改动清单

### 文件：`transformers/llm/engine/src/omni.hpp`

#### 改动 H1：新增成员变量

```cpp
// 替换: std::shared_ptr<Module> mVisionModule, mAudioModule;
// 改为:
std::shared_ptr<Module> mVisionModule, mAudioModule, mAudioEncoder;
// mAudioModule → conv_frontend.mnn (CNN subsampling)
// mAudioEncoder → encoder.mnn      (Transformer encoder)
```

#### 改动 H2：析构函数补充

```cpp
~Omni() {
    mVisionModule.reset();
    mAudioModule.reset();
    mAudioEncoder.reset();  // 新增
}
```

### 文件：`transformers/llm/engine/src/omni.cpp`

#### 改动 C1：模型加载（~ 行 151-163）

**当前**：
```cpp
if (mConfig->is_audio()) {
    auto audio_model_path = mConfig->audio_model();
    mAudioModule.reset(Module::load(..., audio_model_path, ...));
}
```

**改为**：
```cpp
if (mConfig->is_audio()) {
    auto audio_model_path = mConfig->audio_model();
    // Load conv_frontend
    mAudioModule.reset(Module::load(..., audio_model_path, ...));
    if (!mAudioModule.get()) { return false; }

    // Load encoder (new)
    auto audio_encoder_path = mConfig->audio_encoder();
    mAudioEncoder.reset(Module::load(..., audio_encoder_path, ...));
    if (!mAudioEncoder.get()) { return false; }
}
```

#### 改动 C2：推理逻辑（~ 行 910-977）

**当前**（简化）：
```
fbank → input_features [1, 128, T]
if nInputs == 1: audio_embedding = mAudioModule->forward(input_features)
else:            audio_embedding = mAudioModule->onForward({input_features, mask_2d})
audio_embedding = _Permute(audio_embedding, {1, 0, 2})
```

**改为**：
```cpp
// Step 1: FBank → Transpose for conv_frontend
input_features = _Permute(input_features, {0, 2, 1});  // [1,128,T] → [1,T,128]

// Step 2: conv_frontend (1 input, 1 output)
auto conv_output = mAudioModule->forward(input_features);

// Step 3: Create feature mask for encoder
int enc_seq_len = conv_output->getInfo()->dim[1];  // T'
VARP feature_mask = _Input({1, enc_seq_len}, NCHW, halide_type_of<float>());
auto* mask_ptr = feature_mask->writeMap<float>();
std::fill(mask_ptr, mask_ptr + enc_seq_len, 1.0f);  // all-1 = no padding

// Step 4: encoder (2 inputs: features + mask)
auto audio_embedding = mAudioEncoder->onForward({conv_output, feature_mask})[0];

// Step 5: Permute for Decoder (已有)
audio_embedding = _Permute(audio_embedding, {1, 0, 2});  // [1,T',1024] → [T',1,1024]
```

> **关键差异**：feature_mask 是 **1D** [1, seq_len]（特征有效性），不是 Omni 现有 2-input 路径的 **2D** [1, seqlen, seqlen]（attention 矩阵）。Sherpa encoder 内部自己创建 attention mask。

#### 改动 C3：qwen3_asr 类型跳过旧 2-input 分支

qwen3_asr 的 `audio_type` 进入 `audioProcess()` 时，应走新的 conv_frontend → encoder 路径，而不是旧的 2-input attention mask 路径。最简单的方式：在 `nInputs` 检查前，用 `audio_type == "qwen3_asr"` 提前分流。

### 文件：`config.json`

```json
{
    "audio_model": "conv_frontend.mnn",     // 原 "audio.mnn"
    "audio_encoder": "encoder.mnn",         // 新增
    // ... 其余不变
}
```

> 字段名 `audio_encoder` 需在 `llmconfig.hpp` 中新增读取方法。

### 文件：`transformers/llm/engine/src/llmconfig.hpp`

新增：
```cpp
std::string audio_encoder() const {
    return base_dir_ + config_.value("audio_encoder", "encoder.mnn");
}
```

## 改动量估计

| 文件 | 行数 | 说明 |
|------|:----:|------|
| `omni.hpp` | ~3 | 新增成员变量 + 析构 |
| `omni.cpp` | ~40 | 加载 + 推理逻辑改写 |
| `llmconfig.hpp` | ~3 | 新增 `audio_encoder()` 方法 |
| `config.json` | ~1 | 新增/修改字段 |
| **合计** | **~47** | |

## 验证结果 (2026-06-14)

### Step 3: Express API 双模型串联

| 模型 | API | cosim vs ONNX | 判定 |
|------|-----|:------------:|:----:|
| conv_frontend.mnn | Module::onForward (NCHW) | **1.000000** | ✅ 完全一致 |
| encoder.mnn (MNN cf 输入) | Module::onForward (NCHW) | **0.996771** | ✅ 与 Session API 一致 |
| 串联 (cf → encoder) | Module::onForward | **0.996771** | ✅ INT8 量化极限 |

> **关键发现**：原 test_mnn_models.cpp 失败是因为链接了旧版 MNN 库。用当前 build 重新编译后，Express API 完全正常。输入格式统一使用 NCHW (`_Input({1, T, C}, NCHW, ...)`)。

### Step 4: 端到端 First Token

| 指标 | 结果 |
|------|:----:|
| AE cosim (T=300 真实长度) | **0.988756** |
| Per-position cosim 范围 | 0.938 – 0.998 |
| ONNX AE → Decoder 1st token | **15** |
| MNN AE → Decoder 1st token | **15** |
| Top-5 token 交集 | **5/5** |
| Logit cosim | **1.000000** |

> **关键发现**：尽管长序列时 AE cosim 降至 0.989（个别位置 0.938），Decoder 的 cross-attention 机制通过聚合所有位置信息进行了补偿，**first token 和 top-5 完全一致**。INT8 风险 (R1) 已排除。

### 代码实现

| 文件 | 改动 | 状态 |
|------|:----:|:----:|
| `llmconfig.hpp` | +3 行：新增 `audio_encoder()` 方法 | ✅ |
| `omni.hpp` | +2 行：新增 `mAudioEncoder` 成员 + 析构 | ✅ |
| `omni.cpp` (加载) | +10 行：encoder.mnn 加载（非致命失败） | ✅ |
| `omni.cpp` (推理) | +30 行：双模型推理路径 (qwen3_asr) | ✅ |
| `config.json` | 修改：`audio_model`→`conv_frontend.mnn`，新增 `audio_encoder` | ✅ |

**向后兼容**：`audio_model` 加载失败仍报错（必需），`audio_encoder` 加载失败仅警告（可选，回退到旧单模型路径）。

### 桌面集成测试

```
✅ conv_frontend.mnn loaded (42 MB)
✅ encoder.mnn loaded (176 MB)
✅ llm_demo 启动成功，推理正常
✅ 旧模型路径 fallback 正常
```

## 成功率评估

### 有利因素

| # | 因素 | 权重 |
|---|------|:----:|
| 1 | conv_frontend.mnn 与 ONNX **完全等价**（cosim=1.0），CNN 前处理零风险 | 🔴 |
| 2 | encoder.mnn 与 ONNX **高度一致**（cosim=0.997），远好于手写版 0.30 | 🔴 |
| 3 | 输出维度兼容：AE 输出 [T', 1024]，Decoder 期望 1024-dim，**完全匹配** | 🔴 |
| 4 | Omni 引擎已有 Permute [1,T,H]→[T,1,H] 逻辑（omni.cpp:968），**不需改** | 🟡 |
| 5 | 两个模型可独立调试，问题隔离清晰 | 🟡 |
| 6 | 内存增量微小：218 MB vs 当前 210 MB（+4%） | 🟢 |

### 风险点

| # | 风险 | 可能性 | 影响 | 缓解措施 |
|---|------|:------:|:----:|----------|
| R1 | encoder.mnn 的 INT8 精度偏差（cosim 0.997 vs 0.998）导致 first token 变化 | 低 | 高 | 桌面对比验证：sherpa MNN AE → ONNX Dec 的 first token 是否与纯 ONNX 一致 |
| R2 | feature_mask 类型不匹配（ONNX bool → MNN float）导致 encoder 输出异常 | 低 | 高 | Session API 测试已验证 float mask 可用；或改为 uint8 mask |
| R3 | Omni 引擎 Module::load 双模型失败（如路径错误、格式不兼容） | 低 | 中 | 添加 fallback 日志；Session API 已验证两个文件可加载 |
| R4 | Decoder 对帧数变化敏感（AE 帧数从 53→65 变化可能导致注意力模式不同） | 极低 | 低 | 这是**纠正**，不是**改变**——sherpa AE 的帧数才是正确的 |
| R5 | 手机端 CPU 推理时两个小模型比一个大模型慢 | 极低 | 低 | conv_frontend 仅 3 层 Conv2d，< 1ms 额外开销 |
| R6 | FBank Permute 后内存布局可能影响性能 | 极低 | 低 | MNN 的 Permute 是 lazy 操作，不产生额外拷贝 |

### 综合评估

```
成功率: 85% – 90%

主要不确定性:
├── INT8 精度 (R1):      当前 0.997 cosim vs 工作基线 0.998，差距极小
├── feature_mask 类型 (R2): Session API 测试成功，Express API 待验证
└── 端到端 first token:   需桌面对比实验确认

关键验证步骤（5 步，预计 1-2 小时）:
1. ✅ MNNConvert 转换成功                    [已完成]
2. ✅ Session API 数值验证 (cosim=1.0/0.997) [已完成]
3. ✅ Express API (Module::onForward) 双模型串联验证 [已完成 2026-06-14]
4. ✅ 桌面对比：MNN AE → ONNX Dec first token [已完成 2026-06-14]
5. ⏳ 手机实机部署 + A/B 测试
```

## 替代方案 & 降级策略

### 如果 R1 命中（INT8 精度不够）

1. **FP16 encoder**：请求 sherpa 提供 FP16 encoder.onnx（而非 INT8），重新 MNNConvert → cosim 预期 > 0.999
2. **仅替换 conv_frontend**：conv_frontend.mnn（结构正确）→ 手写 18×Transformer → decoder。复杂度高，不推荐
3. **当前手写 audio.mnn 已可用**：手机 FP16 + 无 AEC/NS 的精度已接近可用状态，可作为 fallback

### 如果集成失败（Omni 引擎兼容性）

1. 直接使用 MNN Session API 编写独立的 `audio_encode()` 函数，绕过 Module API
2. 或先保持现状，等待 MNN 官方修复 llmexport.py 的 Qwen3-ASR 支持

## 验证脚本

### Step 3: Express API 双模型串联

```python
# 需配合 C++ 测试：用 Module::onForward 分别跑 conv_frontend + encoder
# 记录中间输出并与 ONNX 对比
```

### Step 4: 端到端 first token 对比

```
同一段音频 → kaldi-native-fbank (HTK mel, preemphasis=0.97)
    ├─→ ONNX conv_frontend → ONNX encoder → ONNX Decoder → Token A
    └─→ MNN conv_frontend  → MNN encoder  → ONNX Decoder → Token B
                                                          A == B ?
```

## 参考

- Sherpa 模型位置: `/Users/bxy/Documents/sherpa-onnx/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/`
- 转换验证代码: `/tmp/sherpa_mnn_test/`
- 根因分析: [[root-cause-analysis]]
- FBank 对齐: [[fbank-numerical-analysis]]
- 项目进度: [[progress]]
