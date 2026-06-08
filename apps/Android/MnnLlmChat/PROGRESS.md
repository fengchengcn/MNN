# Qwen3-ASR Test Page — 开发进度

## 已完成

### UI 改造
- 暗色主题 Material 风格布局 (`activity_qwen3_asr_test.xml`)
- Batch / Streaming 模式切换 chip
- 120dp 圆形录制按钮，三态切换（idle 蓝/active 红/processing 灰）
- 音频电平实时条（基于 raw PCM RMS）
- 卡片式结果展示（序号 + 时间戳 + 识别文本）
- Clear 清空按钮 + 空状态提示
- ripple 触摸反馈、文字可选复制、对比度合规

### 批处理模式 (Batch)
- 点击 REC → 录音 → 点击 STOP → `endAudio()` → `getResultText()` → 显示卡片
- 行为与原版一致，UI 升级

### 流模式 (Streaming)
- 持续录音 + RMS-based VAD 自动分段
- 阈值: SPEECH_RMS=400, SILENCE_RMS=100 (raw PCM int16 尺度)
- 1.5s 静音 → 自动触发解码 → 显示结果 → 300ms 后自动恢复监听
- 最大单段 30s 保护
- `stoppedByUser` 标志防止双重处理
- `onStop()` 暂停录音保活

## 已修复的 Bug

| 发现渠道 | 问题 | 修复 |
|---------|------|------|
| Code Review | Handler 回调在 Activity 销毁后触发 → 崩溃 | 加 `isDestroyed/isFinishing` 检查 |
| Code Review | `CoroutineScope` 裸用 → 内存泄漏 | 改用 `lifecycleScope` |
| Code Review | `currentMode` 跨线程无同步 | `@Volatile` |
| 真机测试 | **RMS 尺度不匹配** — 对归一化 float 算 RMS（值域 ~0.3），阈值却是 raw int16 尺度（值 400/100）→ 永远不触发端点 | 改为对 raw short 算 RMS |
| 编译 | `text` 参数与 `TextView.text` 属性名冲突 | 改用 `setText()` |
| 编译 | `scrollbarThumbVertical` 不接受裸色值 | 移除该属性 |

## 当前流式实现的性能限制（非 Bug，架构层面）

### 延迟模型

```
用户说话 → 1.5s 静音 → [AudioEncoder 加载-infer-释放] + [Decoder 解码] → 显示结果
                          ←———————— 2~5 秒 ————————→
```

### 具体瓶颈

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| Audio Encoder 反复加载 | `ae_mod.reset()` 每次释放 ~500MB 模型，下一段重新加载 | 每段额外 1-2s |
| 无增量解码 | `runDecoder()` 是原子操作，不支持 `decodeStep()` + `getPartialResult()` | 无法边说边出字 |
| VAD 死时间 | 1.5s 静音窗口 + 2-5s 解码期间无法接收新音频 | 用户说话可能丢失 |
| 双模型交替驻留 | AE 和 Decoder 不同时驻留以控制内存 <1GB | 牺牲延迟换内存 |

### 与生产级实时流的差距

| 维度 | 当前 | 生产标准 |
|------|------|---------|
| 首字延迟 | 2-5s | <500ms |
| 部分结果 | 不支持 | 边说边出 |
| 解码中录音 | 停止（丢失音频） | 持续缓冲 |
| Audio Encoder | 每次重载 | 常驻 |
| 增量解码 | 无 | 逐 token 产出 |

### 要达到生产标准，需在 Native 层 (qwen3_asr_engine.cpp) 做

1. **Audio Encoder 常驻** — 不释放 `ae_mod`，与 Decoder 同时持有（内存 +~500MB）
2. **流式解码 API** — 暴露 `startStream()`, `pushAudio()`, `decodeStep()`, `getPartialResult()` 给上层
3. **音频双缓冲** — 解码期间继续录音到独立 buffer，解码完成后立即处理下一段

## 文件清单

| 文件 | 状态 |
|------|------|
| `res/layout/activity_qwen3_asr_test.xml` | 已改写 |
| `res/drawable/bg_asr_result_card.xml` | 新增 |
| `res/drawable/bg_rec_button_idle.xml` | 新增 |
| `res/drawable/bg_rec_button_active.xml` | 新增 |
| `res/drawable/bg_rec_button_processing.xml` | 新增 |
| `res/drawable/bg_mode_chip_selected.xml` | 新增 |
| `res/drawable/bg_mode_chip_normal.xml` | 新增 |
| `res/drawable/bg_audio_level_bar.xml` | 新增 |
| `asr/Qwen3AsrTestActivity.kt` | 已改写 |
