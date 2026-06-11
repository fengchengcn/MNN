# Qwen3-ASR → llmexport.py 迁移计划

> 创建：2026-06-09 | 完成：2026-06-11
> **状态：WP1-WP6 全部完成 ✅**
>
> 模型已通过 llmexport.py 成功导出（INT8 814MB / FP16 1.3GB）。
> Android VoiceChatPresenter Omni 引擎集成完成，旧引擎 Qwen3AsrEngine 已移除。

## 迁移动机

旧路径 `export_qwen3_asr.py` → ONNX → MNNConvert 的问题：
1. **分解算子阻塞 GPU 加速**：Attention 被分解为 MatMul+Add+Softmax，无法利用 FusedAttention kernel
2. **未集成进 Omni 引擎**：手写 decode 循环无采样策略、无 prefix caching

## 收益（实际达成）

| 维度 | 旧路径 | 新路径 (Omni) |
|------|--------|--------------|
| FusedAttention | ❌ 分解 MatMul | ✅ 29 个 FusedAttention |
| 采样策略 | 仅 argmax | top-k/top-p/temperature/min_p |
| 推理引擎 | 手写 ~300 行 | Omni 引擎内置 |
| Decode 速度 | ~20 tok/s | 18-22 tok/s (FP16 权重大) |
| AE 权重加载 | 单文件全量 | mmap 延迟缺页 |

## 工作包（全部完成）

```
WP1: 模型注册 (model_mapper.py)          ✅ 2026-06-09
WP2: 模型加载 (model.py)                 ✅ 2026-06-09 (含修正)
WP3: 模型适配 (transformers.py)          ✅ 无需修改（已验证兼容）
WP4: 导出适配 (llmexport.py)             ✅ 2026-06-09
WP5: C++ 引擎集成 (omni.cpp)             ✅ 基本完成
WP6: 端到端验证                           ✅ x86 + Android 验证通过
```

## 导出产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `llm.mnn` + `.weight` | 494K + 604 MB (INT8) / 1.1 GB (FP16) | 29× FusedAttention |
| `audio.mnn` + `.weight` | 350K + 210 MB | AE INT8 (transformer_fuse=False) |
| `config.json` | ~1 KB | `is_audio`, `audio_type`, `jinja` template |
| `tokenizer.txt` / `.mtok` | ~3 MB | BPE tokenizer |

## 关键问题与修复

### 1. Qwen3-ASR 无法通过 transformers 标准路径加载
- **症状**: `AutoConfig.from_pretrained()` 失败（`qwen3_asr` 未注册）
- **修复**: 新增 `utils/qwen3_asr_model.py`，直接从 safetensors 加载权重到自定义 PyTorch 模块

### 2. Audio encoder segfault
- **症状**: Omni 加载 audio.mnn 后 response() 崩溃
- **根因**: `--transformerFuse` 融合了 AE 内部 18 层 encoder-only Transformer
- **修复**: `export_audio(transformer_fuse=False)`

### 3. EOS-only 输出
- **症状**: 模型仅输出 `<|im_end|>` (1 token)
- **根因**: 缺少 Jinja chat template
- **修复**: config.json 添加 Qwen 格式 jinja template

### 4. ConfigObj 缺少 dict 兼容方法
- **症状**: `rope_scaling` 不支持 `values()`、`in` 等 dict 操作
- **修复**: 新增 `ConfigObj` 类支持属性+dict 双访问

### 5. Qwen3AsrAudio 属性初始化顺序
- **症状**: `self.audio_pad_id` 在 `super().__init__()` 之后赋值，但父类调用了 `self.load()`
- **修复**: 赋值移到 `super().__init__()` 之前

## Android 集成：模型检测优先级

```
audio.mnn + config.json(is_audio=true) → QWEN3_OMNI  ← 优先
audio_encoder.mnn 存在                  → QWEN3_OLD  (已移除)
默认                                    → SHERPA
```

### Omni 路径数据流
```
AudioRecord → omniAudioBuffer → writeWavFile() → <audio> tag
→ ChatPresenter.sendMessage() → LlmSession.Response()
→ Omni 引擎自动: fbank → AE → 嵌入注入 → 推理
```

## 回滚策略

双轨并行，零风险：
- 保留旧引擎代码不动（后续已移除）
- 新路径在 omni.cpp 独立运行
- 通过 `AsrMode` 枚举切换

## Git 提交

```
b95e6eb2 [LLM:Feature] Qwen3-ASR Omni engine integration — VoiceChatPresenter Plan B
2ce26418 [LLM:Feature] Qwen3-ASR llmexport.py migration — complete WP1-WP6
c750aa41 [LLM:Bugfix] Register qwen3_asr_audio_encoder in Audio.get_audio()
b4aa1602 [LLM:Bugfix] Fix Qwen3-ASR mapper paths — use thinker. prefix
6324c0f1 [LLM:Feature] Add Qwen3-ASR llmexport.py migration (WP1-WP5)
```
