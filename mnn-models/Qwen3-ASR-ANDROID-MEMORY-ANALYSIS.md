# Qwen3-ASR Android 运行时内存与推理性能分析

> **日期:** 2026-06-11 | **状态:** 实机测试完成
> **设备:** 华为 TAS-AL00 (Kirin 990, 5.4GB RAM, Android 12)
> **应用:** MnnLlmChat | **页面:** `Qwen3AsrTestActivity` — OMNI-VAD 模式
> **后端:** CPU (MNN)
>
> ⚠️ 此数据来自旧引擎 FP16 模型。Omni 引擎预计内存 ~30-40% 更低（DiskEmbedding + mmap AE 权重）。
> 理论分析见 [[Qwen3-ASR-MEMORY-ANALYSIS]]。

---

## 1. 设备概况

| 参数 | 值 |
|------|-----|
| SoC | Kirin 990 (4×A76 + 4×A55) |
| GPU | Mali-G76 MP16 (未使用) |
| RAM 总量 | 5,679,556 KB (~5.4 GB) |
| 加载模型后可用 RAM | ~346 MB |
| Swap 总量 / 已用 | 2.2 GB / 1.06 GB |
| 电池 | 93%, 32°C |

---

## 2. 模型配置

应用从 `/data/local/tmp/mnn_models/` 扫描模型目录，优先加载 FP16（打分规则：FP16=2 > INT8=1）。

**当前加载: Qwen3-ASR-MNN-FP16**

### 模型文件大小

| 组件 | FP16 | INT8 | 说明 |
|------|------|------|------|
| audio.mnn | 346 KB | 359 KB | 音频编码器图结构 |
| audio.mnn.weight | 210 MB | 210 MB | 音频编码器权重（两者相同） |
| llm.mnn | 506 KB | 506 KB | LLM 解码器图结构 |
| llm.mnn.weight | **1,192 MB** | **634 MB** | LLM 解码器权重 |
| 分词器 | 3.2 MB | 3.2 MB | 词表文件 |
| **权重合计** | **1,402 MB** | **844 MB** | |
| **全部文件** | **1,410 MB** | **860 MB** | |

> INT8 比 FP16 节省 **558 MB (39.8%)** 的权重文件，LLM 解码器权重从 1,192 MB 降至 634 MB（降幅 46.8%）。

---

## 3. 内存分析

### 3.1 内存构成 (dumpsys meminfo)

| 类别 | PSS (MB) | 占比 | 说明 |
|------|----------|------|------|
| **Native Heap** | **1,580.4** | **91.6%** | 模型权重 + 运行时 buffer |
| 代码 (.so/.dex/.art) | 99.2 | 5.8% | 原生库 + DEX |
| Java Heap | 13.5 | 0.8% | UI、Android 框架 |
| System | 19.5 | 1.1% | 系统分配 |
| 其他 (Graphics/Stack) | 12.1 | 0.7% | GL 缓冲 + 线程栈 |
| **总计 PSS** | **1,724.7** | **100%** | |
| **总计 RSS** | **1,841.1** | | |

```
Native Heap  ██████████████████████████████████████████████ 91.6% (1,580 MB)
代码         ███                                             5.8% (  99 MB)
Java Heap    ▏                                               0.8% (  14 MB)
其他         ▏                                               1.8% (  32 MB)
```

### 3.2 Native Heap 详情

| 指标 | 值 |
|------|-----|
| 堆大小 | 1,647.4 MB |
| 已分配 | 1,646.2 MB |
| 空闲 | 11.0 MB |
| **利用率** | **99.3%** |

Native Heap 构成估算：
- 模型权重：~1,402 MB
- KV Cache + 中间 tensor + 图执行 buffer：~244 MB
- 运行时开销比：权重 × 1.17 = 实际内存

### 3.3 进程级指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 稳态 RSS | 1,843 MB | 模型加载完成后的常驻内存 |
| 峰值 RSS (VmHWM) | **2,353 MB** | 加载期间峰值，比稳态高 510 MB |
| 虚拟内存 (VmSize) | 10,577 MB | 含 mmap 映射的虚拟地址空间 |
| 线程数 | 47 | 含 9 个 Mali GPU 空闲线程、20+ 框架线程 |

> 模型加载期间 RSS 冲高 510 MB，是文件 I/O buffer 和权重反序列化临时内存叠加导致的。

### 3.4 系统内存压力

| 指标 | 值 |
|------|-----|
| 系统空闲 RAM | ~346 MB (6.1%) |
| 系统 Swap 使用 | 1.06 GB (48.5%) |
| App 占物理内存比例 | ~33% |

---

## 4. FP16 vs INT8 内存对比（预估）

| 指标 | FP16 (当前) | INT8 (预估) | 节省 |
|------|------------|------------|------|
| 权重文件 | 1,402 MB | 844 MB | 558 MB (39.8%) |
| Native Heap | 1,646 MB | ~1,000 MB | ~646 MB (39.3%) |
| 总 RSS | 1,885 MB | ~1,239 MB | ~646 MB (34.3%) |
| 占设备 RAM | 33% | ~22% | |

> **结论：** INT8 模型将内存占用降低约 40%，对 4GB 以下设备至关重要。

---

## 5. 模型加载时间（估算）

| 组件 | 权重大小 | 估算加载时间 |
|------|---------|-------------|
| 音频编码器 | 210 MB | ~1–2s |
| LLM 解码器 | 1,192 MB | ~6–12s |
| 分词器 + 图编译 | 3 MB | ~1–2s |
| **合计** | **1,405 MB** | **~8–16s** |

---

## 6. 推理性能

> 数据来源：MNN 引擎内建 `MNN_DEBUG PERF` 日志。每条 PERF 对应一次 OMNI-VAD 语音段推理。

### 6.1 实际推理记录

| # | Prefill | Decode |
|---|---------|--------|
| 1 | 69 tok / 0.43s (**161 t/s**) | 9 tok / 0.41s (**22 t/s**) |
| 2 | 90 tok / 0.56s (**161 t/s**) | 13 tok / 0.64s (**20 t/s**) |
| 3 | 110 tok / 0.54s (**205 t/s**) | 21 tok / 1.03s (**20 t/s**) |
| 4 | 45 tok / 0.29s (**155 t/s**) | 6 tok / 0.25s (**24 t/s**) |
| 5 | 53 tok / 0.27s (**193 t/s**) | 12 tok / 0.57s (**21 t/s**) |
| 6 | 60 tok / 0.35s (**173 t/s**) | 12 tok / 0.68s (**18 t/s**) |

### 6.2 统计汇总

| 阶段 | 说明 | Token 数 | 耗时 | 平均速度 |
|------|------|----------|------|----------|
| **Prefill** | 音频编码 + embedding 注入 + 首次前向 | 45–110 | 0.27–0.56s | **174.8 t/s** |
| **Decode** | 逐 token 自回归生成 | 6–21 | 0.25–1.03s | **20.8 t/s** |
| **单段总计** | 每个 VAD 语音段 | 51–131 | 0.54–1.57s | — |

- Prefill 速度与输入 token 数相关（音频越长 token 越多）
- Decode 速度稳定在 ~21 t/s，约 **48ms/token**，与上下文长度无关

### 6.3 实时率 (RTF)

| 音频时长 | 总推理时间 | RTF | 倍速 |
|----------|-----------|-----|------|
| 7.1s (113,600 采样点) | 1.57s | 0.22 | **4.5× 实时** |
| 1.9s (30,400 采样点) | 0.54s | 0.28 | **3.5× 实时** |
| **平均** | | **~0.25** | **~4× 实时** |

> RTF < 1 即满足实时流式 ASR 需求，当前 4× 实时速度有充足余量。

### 6.4 推理时内存波动

| 状态 | Native Heap 已分配 | 变化 |
|------|-------------------|------|
| 空闲（模型加载完毕） | 1,680 MB | 基线 |
| 推理后 #1 | 1,686 MB | +6 MB |
| 推理后 #2 | 1,635 MB | -45 MB (GC?) |

> 推理本身内存压力极小（+5–10 MB），KV Cache 已预分配。多轮推理未发现内存泄漏。

---

## 7. 总结与建议

### 主要发现

1. **内存绝对主导者：模型权重。** Native Heap 占 91.6%，Java/UI 开销可忽略。
2. **峰值内存 2.35 GB**，比稳态高 510 MB，是 OOM 风险点（4GB 设备）。
3. **Prefill 平均 175 t/s，Decode 平均 21 t/s**，总体 4× 实时速度。
4. **INT8 可省 ~40% 内存**，是低内存设备的必选项。
5. **仅 CPU 推理**，GPU 未参与（GL mtrack 仅 4.3 MB）。
6. **无内存泄漏**，多轮推理后内存回归基线。

### 建议

| 优先级 | 建议 | 预期收益 |
|--------|------|----------|
| 高 | 生产环境使用 INT8 模型 | 节省 558 MB 磁盘 + 646 MB 内存 |
| 中 | 实现 mmap 权重加载 | 消除 510 MB 加载峰，加速冷启动 |
| 中 | 加载前做内存预算检查 | 避免低内存设备 OOM |
| 低 | 调小 KV Cache / max_length | 进一步压缩 ~64–128 MB |

---

## 8. GPU 后端启用分析

### 8.1 可选后端对比

MNN 支持的 Android GPU 后端：

| 后端 | MNNForwardType | 设备支持 | 当前 APK 编译 | KVMeta | shapeMutable | 注意 |
|------|---------------|----------|-------------|--------|-------------|------|
| CPU | 0 | ✅ | ✅ | ✅ | ✅ (动态) | 当前使用，Decode 20.8 t/s |
| **OpenCL** | 3 | ✅ (Mali-G76) | ✅ (`libMNN_CL.so`) | ❌ 禁用 | ❌ (false) | 代码已有后端选择 UI |
| **Vulkan** | 7 | ✅ (Mali-G76) | ❌ 需重新编译 | ✅ 未禁用 | ❌ (false) | KVMeta 正常 |
| NPU (NN) | 5 | ✅ (Kirin 990) | ❌ 未编译 | — | ❌ (false) | 需 DaVinci 适配 |

### 8.2 OpenCL vs Vulkan 关键代码差异

在 `llm.cpp` 中，两个 GPU 后端的关键差异：

```cpp
// 1. 模型加载 — 两者相同，均禁用动态形状
if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
    module_config.shapeMutable = false;  // GPU 后端不支持动态 shape 优化
}

// 2. KVMeta 调优 — 仅 OpenCL 被禁用！
// FIXME: Currently OpenCL Don't support KVMeta
if (mConfig->backend_type() == "opencl") {
    return;  // ⚠️ OpenCL 跳过 KVMeta 优化
}

// 3. Record Queue 批处理 — 仅 OpenCL 有此优化
if (mConfig->backend_type() == "opencl") {
    mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, 512);
}

// 4. Attention 下三角 mask 优化 — 仅 CPU
if (mConfig->backend_type() == "cpu" && mValidBlockSize.empty()) {
    // 使用标量 mask（零内存），GPU 后端使用显式 4D mask
}
```

### 8.3 Vulkan 优势

| 维度 | OpenCL | Vulkan |
|------|--------|--------|
| KVMeta 块大小调优 | ❌ 禁用 | ✅ 正常工作 |
| Record Queue 批处理 | ✅ 512-op 合并 | ❌ 无（单 op 提交） |
| shapeMutable 动态形状 | ❌ | ❌ |
| Attention mask 优化 | ❌ 4D mask | ❌ 4D mask |
| Mali GPU 驱动成熟度 | 一般 | **更好**（Vulkan 是 Android 一等公民） |
| 业界趋势 | 逐渐被替代 | Google/ARM 主推 |

**核心结论：Vulkan 保留了 KVMeta 优化**，这对 Decode 阶段性能至关重要（KVMeta 自动选择最优 KV Cache 计算块大小）。而 OpenCL 跳过 KVMeta 可能导致 Decode 性能不升反降。

### 8.4 启用 OpenCL（最小改动）

只需修改 1 行代码：

```kotlin
// Qwen3AsrTestActivity.kt 第 262 行
- "cpu"          // backendType
+ "opencl"       // backendType
```

或在 `config.json` 中设置：
```json
{ "backend_type": "opencl" }
```

**风险：**
- KVMeta 被禁用，Decode 可能比 CPU 慢
- `libMNN_CL.so` 已在 APK 中，无需重新编译
- Mali OpenCL 驱动可能有稳定性问题

### 8.5 启用 Vulkan（需重新编译）

**步骤 1：编译 MNN 时开启 Vulkan**

```bash
cd MNN
mkdir build_vulkan && cd build_vulkan
cmake .. \
  -DMNN_VULKAN=ON \
  -DMNN_OPENCL=OFF \
  -DMNN_BUILD_LLM=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**步骤 2：修改 JNI CMakeLists.txt**

```cmake
# 新增 Vulkan 库引用
add_library(MNN_Vulkan SHARED IMPORTED)
set_target_properties(MNN_Vulkan PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libMNN_Vulkan.so")

target_link_libraries(${CMAKE_PROJECT_NAME}
    ...
    MNN_Vulkan   # 新增
    ...
)
```

**步骤 3：修改 Activity 代码**

```kotlin
// Qwen3AsrTestActivity.kt 第 262 行
- "cpu"          // backendType
+ "vulkan"       // backendType
```

**步骤 4：更新 Settings UI（可选）**

```kotlin
// SettingsBottomSheetFragment.kt 第 145 行
- val backendOptions = listOf("cpu", "opencl")
+ val backendOptions = listOf("cpu", "opencl", "vulkan")
```

### 8.6 实测：OpenCL 性能（失败）

已将 `Qwen3AsrTestActivity.kt` 改为 `"opencl"` 并在同一设备上实测：

| 阶段 | CPU (基线) | OpenCL (实测) | 劣化倍数 |
|------|-----------|--------------|----------|
| Prefill | 175 t/s | **2.9–11.0 t/s** | 15–60× 慢 |
| Decode | 21 t/s | **1.7–6.8 t/s** | 3–12× 慢 |

原始 PERF 日志：
```
# 第一次增量推理 (2.5s 音频)
PERF | prefill: 53 tok in 18.00s (2.9 t/s) | decode: 8 tok in 4.62s (1.7 t/s)

# 第二次增量推理 (5.5s 音频)
PERF | prefill: 90 tok in 8.19s (11.0 t/s) | decode: 15 tok in 2.19s (6.8 t/s)
```

**失败原因分析：**

1. **Mali-G76 OpenCL 驱动 kernel 启动开销大** — LLM 推理有大量小 op，每次都要 GPU 调度
2. **KVMeta 被禁用** — 无法自动调优 KV Cache 块大小
3. **Decode 是 memory-bound** — 每 token 只需少量矩阵乘法，GPU 并行优势无法发挥
4. **OpenCL Record Queue 批处理不够激进** — 512-op 批处理在 Mali 上仍有可见延迟

### 8.7 Vulkan 实测（同样失败）

编译了 `MNN_VULKAN=ON`，重新打包 APK，在同一设备实测：

```
MNN_DEBUG load begin modelId: omni_test backend: vulkan
LIFECYCLE: LlmSession CREATED ... load_success=1    ← 模型加载成功
vulkan: searching for layers in '.../lib/arm64'       ← Vulkan 驱动初始化
vulkan: failed to load layer library 'libiGraphicsCore.huawei.so' ← 华为 Vulkan 层加载失败（非致命）
...
Segment #1 — transcribing... 42.5s                     ← 推理卡死，无 PERF 输出
```

结果：**模型加载成功，但推理无法完成。** 一个 5 秒的语音段处理 42 秒仍无输出。

### 8.8 最终结论

| 后端 | 实测结果 | Prefill | Decode | 判断 |
|------|---------|---------|--------|------|
| **CPU** | ✅ 正常 | **175 t/s** | **21 t/s** | **唯一可用选项** |
| OpenCL | ❌ | 2.9 t/s (60× 慢) | 1.7 t/s (12× 慢) | 不可用 |
| Vulkan | ❌ | 卡死 | 卡死 | 不可用 |

**移动端 LLM Decode 是 memory-bound 任务，GPU 不适合。** 每 token 只需少数矩阵乘法，GPU kernel 启动/同步的开销远大于实际计算。CPU 反而凭借低延迟和高效缓存成为最优解。

**当前最优配置：CPU + 4 线程 + FP16 + Low Precision + Low Memory。**

### 8.7 最终建议

移动端 LLM Decode 是 memory-bound 任务，CPU 是最优解。
**当前最优配置：CPU + 4 线程 + FP16 + Low Precision + Low Memory。**

---

## 附录：采集方法

```bash
# 设备信息
adb shell getprop
adb shell cat /proc/meminfo

# 进程内存
adb shell dumpsys meminfo com.alibaba.mnnllm.android
adb shell cat /proc/<pid>/status

# 推理性能（MNN 内建日志）
adb logcat -s MNN_DEBUG | grep PERF

# 模型文件
adb shell ls -laR /data/local/tmp/mnn_models/
```
