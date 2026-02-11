# MNN MultiModal Audio Demo (MNN 多模态语音交互 Demo)

这是一个基于 MNN Framework 开发的 Android 应用程序，演示了如何在端侧设备上运行 Qwen2.5-Omni-7B 多模态大模型。该应用支持语音输入（录音）和图像输入，并能通过语音（TTS）和文本实时反馈 AI 的回答。

对于前端开发者来说，你可以将这个项目理解为一个移动端的全栈应用：
- **UI 层 (Kotlin/XML)**：类似于前端的 React/Vue 组件与 HTML/CSS。
- **逻辑层 (Kotlin)**：处理用户交互、数据流转。
- **Native 层 (C++)**：类似于后端服务或 WebAssembly 模块，负责核心的高性能计算（模型推理）。
- **JNI (Java Native Interface)**：连接 Java/Kotlin 与 C++ 的桥梁。

---

## 1. 项目目录结构概览

以下是项目的核心目录结构及其作用说明：

```
MnnMultiModalAudioDemo/
├── app/                           # 主应用模块 (类似于前端项目中的 src)
│   ├── src/
│   │   └── main/
│   │       ├── java/             # Java/Kotlin 源代码 (业务逻辑)
│   │       │   └── com/alibaba/mnnllm/multimodal/audio/
│   │       │       ├── MainActivity.kt        # 主入口页面 (UI交互、权限、流程控制)
│   │       │       ├── AudioHandler.kt        # 录音逻辑封装
│   │       │       ├── SimpleWaveRecorder.kt  # 具体的录音实现 (PCM转WAV)
│   │       │       └── TtsManager.kt          # 文字转语音 (TTS) 管理
│   │       │
│   │       │       ├── asr/
│   │       │       │   └── RecognizeService.kt    # 流式语音识别 (ASR, sherpa-mnn)
│   │       │
│   │       │   └── com/k2fsa/sherpa/mnn/          # sherpa-mnn 的 Kotlin 封装 (JNI 交互)
│   │       │       ├── OnlineRecognizer.kt
│   │       │       ├── FeatureConfig.kt
│   │       │       └── OnlineStream.kt
│   │       │
│   │       ├── cpp/              # C++ 源代码 (核心推理引擎)
│   │       │   ├── CMakeLists.txt             # C++ 构建脚本 (定义编译规则)
│   │       │   └── multimodal_audio_jni.cpp   # JNI 接口实现 (Java与MNN的桥梁)
│   │       │
│   │       ├── res/              # 资源文件 (布局、图片、字符串等)
│   │       │   ├── layout/
│   │       │   │   └── activity_main.xml      # 主界面布局文件 (新增 ASR 悬浮按钮与录音指示层)
│   │       │   └── values/                    # 常量定义 (颜色、字符串)
│   │       │
│   │       └── AndroidManifest.xml # 应用清单文件 (声明权限、Activity、应用元数据)
│   │
│   └── build.gradle              # 模块级构建配置 (依赖管理、SDK版本)
│
├── build.gradle                   # 项目级构建配置
├── settings.gradle                # 项目设置 (包含的模块)
└── gradle/                        # Gradle 包装器 (保证构建环境一致)
```

---

## 2. 核心文件与功能详解

### 2.1 Java/Kotlin 层 (UI 与 交互)

*   **`MainActivity.kt`**: 应用的“大脑”。
    *   **职责**:
        *   **UI 初始化**: 加载布局，设置按钮点击事件。
        *   **权限管理**: 申请录音和存储权限。
        *   **模型管理**: 检查本地是否有 `Qwen2.5-Omni` 模型。如果没有，调用 `ModelDownloadManager` 从 ModelScope 下载。
        *   **用户交互**:
            *   **选择图片**: 调用系统相册，将选中的图片复制到私有缓存目录，保存路径供 C++ 层读取。
            *   **长按录音**: 监听 `MotionEvent`，按下时开始录音，松开时停止录音并保存为 `.wav` 文件。
        *   **调用推理**: 获取图片路径和音频路径，拼接成 Prompt，调用 JNI 函数 `nativeChat()`。
        *   **接收回调**: 定义了 `onChatStreamUpdate(chunk: String)` 方法，供 C++ 层调用，实时更新 UI 和播放语音。

*   **`AudioHandler.kt` / `SimpleWaveRecorder.kt`**:
    *   负责调用 Android 的 `AudioRecord` API 录制音频。
    *   将原始的 PCM 音频数据加上 WAV 文件头，保存为标准的 `.wav` 文件。这是因为 MNN 的音频处理模块通常需要标准格式的音频文件。

*   **`TtsManager.kt`**:
    *   简单的 `TextToSpeech` 封装，用于将 AI 生成的文本转换为语音播放。

### 2.3 语音识别（ASR）

*   **`RecognizeService.kt`**（流式 ASR 服务）：
    *   负责从麦克风读取 16kHz 单声道 PCM 数据，按 100ms chunk 送入在线识别器。
    *   基于 sherpa-mnn 的在线 Zipformer Transducer 模型进行解码，内部自动进行端点检测（Endpointing）。
    *   端点触发时（一句话结束），通过回调 `onRecognizeText(text)` 返回识别文本，主界面随后将该文本发送给大模型进行问答。
    *   关键方法与流程：
        *   初始化识别器：`initRecognizer(asrModelDir, int8 = true)`，加载 encoder/decoder/joiner 三个 `.mnn` 文件与 `tokens.txt`，并加载 `with-state-epoch-99-avg-1.int8.onnx` 语言模型。
        *   采集与解码：`startRecord()` 启动录音线程，循环 `acceptWaveform()` 并在 `isReady()` 时调用 `decode()`；端点判定后获取 `getResult(stream).text`。
        *   停止：`stopRecord()` 停止录音并释放资源。
    *   代码参考：
        *   [RecognizeService.kt](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/asr/RecognizeService.kt)
*   **MainActivity 集成**：
    *   下载与准备 ASR 模型：根据系统语言自动选择中英双语或英文模型并下载缓存，完成后调用 `initAsr(path)`。
    *   悬浮按钮控制：点击右下角的麦克风悬浮按钮可开始/停止 ASR 录音，识别完成后会自动停止并将文本发到对话流。
    *   识别结果处理：`handleAsrText(text)` 将识别文本作为用户消息加入聊天，并与当前选中图片一起发送给 LLM。
    *   代码参考：
        *   [setupAsrFloatingButton](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L257-L280)
        *   [ensureAsrModelAndInit](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L290-L301)
        *   [initAsr](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L307-L350)
        *   [handleAsrText](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L352-L363)

### 2.2 Native C++ 层 (模型推理)

*   **`multimodal_audio_jni.cpp`**: 核心逻辑所在。
    *   **`JNI_OnLoad`**: 初始化 JVM 环境，以便后续能在 C++ 线程中回调 Java 方法。
    *   **`Java_..._loadModel`**: 
        *   接收 Java 传入的模型路径。
        *   使用 `MNN::Transformer::Llm::createLLM` loading 模型 (加载 `config.json` 等配置)。
    *   **`Java_..._nativeChat`**:
        *   **输入解析**: 接收包含 `<img>path</img>` 和 `<audio>path</audio>` 标签的 Prompt 字符串。
        *   **多模态构建**: 解析标签，将本地图片和音频文件路径加载到 `MNN::Transformer::MultimodalPrompt` 结构体中。
        *   **执行推理**: 调用 `g_llm->response()` 开始生成回答。
    *   **`LlmStreamBuffer`**: 自定义的流缓冲区，用于捕获 LLM 的实时输出字符流。
    *   **`notifyJava`**: 将捕获到的字符流通过 JNI 回调给 Java 层的 `onChatStreamUpdate`。

*   **`CMakeLists.txt`**:
    *   配置 C++ 编译选项。
    *   链接 MNN 的核心库 (`libMNN.so`, `libMNN_LLM.so`, `libMNNAudio.so` 等)。
    *   这些库文件通常由外部框架提供（在 `settings.gradle` 中可以看到引入了 `mnn_tts` 等模块）。

---

## 3. 项目运行机制与流程

### 阶段一：初始化
1.  **App 启动**: `MainActivity` `onCreate` 执行。
2.  **组件准备**: 初始化 `AudioHandler` (录音机) 和 `TtsManager` (朗读机)。
3.  **模型检查**:
    *   检查本地目录 `.mnnmodels/MNN/Qwen2.5-Omni-7B-MNN` 是否存在且完整。
    *   **不存在**: 显示下载进度，调用 `ModelDownloadManager` 下载模型 (约 8GB-10GB)。
    *   **存在**: 调用 Native 方法 `loadModel()` 加载模型进入内存。
4.  **ASR 模型检查与初始化**：
    *   根据系统语言自动选择：
        *   中文环境：`ModelScope/MNN/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20`
        *   英文环境：`ModelScope/MNN/sherpa-mnn-streaming-zipformer-en-2023-02-21`
    *   检查是否已下载并完整（必须包含 `encoder-epoch-99-avg-1.int8.mnn`, `decoder-epoch-99-avg-1.int8.mnn`, `joiner-epoch-99-avg-1.int8.mnn`, `tokens.txt`, `with-state-epoch-99-avg-1.int8.onnx`）。
    *   完成后调用 `initAsr(path)` 初始化识别服务。

### 阶段二：用户交互 (Input)
1.  **图片输入 (可选)**:
    *   用户点击 "选择图片"。
    *   系统相册选择图片 -> `copyUriToCache` 将图片复制到 App 缓存目录 (例如 `/data/user/0/.../cache/temp_image.jpg`)。
    *   记录图片路径 `currentImagePath`。
2.  **语音输入**:
    *   用户 **按住** "Record" 按钮。
    *   `SimpleWaveRecorder` 开始录制 16kHz 单声道音频。
    *   用户 **松开** 按钮。
    *   录音停止，保存为 `.wav` 文件 (例如 `/data/user/0/.../cache/record_12345.wav`)。
    *   记录音频路径 `wavPath`。
3.  **ASR 输入（实时语音转文本）**:
    *   点击右下角的 **麦克风悬浮按钮**（`btn_asr_floating`）开始录音，界面显示 **录音指示层**（`recording_indicator`）与 **波形视图**（`WaveformView`）。
    *   端点检测触发后自动停止录音，识别文本通过回调传回主界面并作为用户消息发送。

### 阶段三：推理与反馈 (Inference & Output)
1.  **构建 Prompt**:
    *   Kotlin 层将路径拼接成伪 XML 格式的 Prompt：
        ```xml
        <img>/path/to/image.jpg</img><audio>/path/to/audio.wav</audio>
        ```
2.  **Native 调用**:
    *   调用 `nativeChat(prompt)`。
3.  **C++ 解析与执行**:
    *   C++ 解析 `<img>` 和 `<audio>` 标签，加载实际文件数据。
    *   `MNN Loop`: LLM 模型根据输入开始逐字生成回答。
4.  **流式回调**:
    *   每生成一段文本，C++ 通过 `notifyJava` -> JNI -> `MainActivity.onChatStreamUpdate` 将文本传回。
5.  **UI 更新与 TTS**:
    *   Kotlin 更新 `TextView` 显示回答。
    *   同时调用 `ttsManager.speak(text)` 朗读出来的文字。

---

## 4. 给前端开发者的类比总结

| Android 概念 | 前端 Web 概念 | 说明 |
| :--- | :--- | :--- |
| **Activity** | **Page / Component** | 一个屏幕/页面，包含 UI 和交互逻辑。 |
| **layout.xml** | **HTML / DOM** | 定义界面的结构和元素 (Button, TextView/Div, ImageView/Img)。 |
| **Gradle** | **WebPack / Vite + package.json** | 构建工具，管理依赖和打包流程。 |
| **JNI (Native)** | **WebAssembly (Wasm)** | 允许调用底层高性能代码 (C/C++)。MNN 在这里相当于一个高性能的 Wasm 模块。 |
| **ViewModel/Handler** | **Store / Service** | 处理业务逻辑，如录音、数据转换。 |
| **Manifest.xml** | **manifest.json / PWA Config** | 应用的全局配置，声明入口和权限。 |

## 5. 如何开始调试

1.  **连接真机**: 此应用涉及录音和高性能计算，建议使用真机调试。
2.  **Sync Gradle**: 在 Android Studio 中点击右上角的 "Sync Project with Gradle Files" 图标，确保所有依赖下载完成。
3.  **Run**: 点击绿色的三角形 "Run" 按钮。
4.  **Logcat**: 在底部的 Logcat 面板中，你可以输入 `MnnMultiModalAudio_JNI` 来查看 C++ 层的日志，或者 `AudioHandler` 查看 Java 层的日志。
5.  **ASR 日志**: 过滤 `ASR_RecognizeService` 查看 ASR 录音与解码流程日志；查看 `MainActivity` 中与 ASR 相关的状态日志（下载、初始化、按钮状态）。



## 6. 问题记录
最初的问题是解决 Android 应用加载 libsherpa-mnn-jni.so 时因为找不到 MNN 符号而导致的闪退（UnsatisfiedLinkError）。
1. 最初要解决的问题
•
现象：应用启动或调用 ASR（语音识别）功能时直接崩溃。
•
报错：java.lang.UnsatisfiedLinkError: dlopen failed: cannot locate symbol "...MNN::Express::Module::load..."。
•
原因：你项目中使用的 libsherpa-mnn-jni.so 是预编译的（可能是从 CDN 下载的），而它链接的 MNN 函数签名与你本地项目中实际运行的 libMNN.so 不匹配（版本、NDK 版本或编译配置不一致）。
2. 解决方案总结
为了彻底解决“版本不匹配”的问题，我们的核心思路是：用你本地正在使用的 MNN 库，重新手动编译一遍 sherpa-mnn。
方案 A：手动编译 sherpa-mnn（目前正在进行的步骤）
这是最可靠的方案，确保所有的 .so 文件“同根同源”。
1.
环境配置：
◦
使用 Android SDK 自带的 CMake 3.22.1 和 Ninja，避免系统全局 CMake 4.2 的兼容性问题。
◦
指定本地 MNN 库的路径（-DMNN_LIB_DIR）。
2.
执行编译命令（在 sherpa-mnn/build-android-arm64-v8a 目录下）：
gitbash 中执行：
```bash
/d/AndroidSdk/cmake/3.22.1/bin/cmake.exe -G "Ninja" \
  -DCMAKE_MAKE_PROGRAM="/d/AndroidSdk/cmake/3.22.1/bin/ninja.exe" \
  -DCMAKE_TOOLCHAIN_FILE="/d/AndroidSdk/ndk/25.2.9519653/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-21" \
  -DSHERPA_MNN_ENABLE_JNI="ON" \
  -DSHERPA_MNN_ENABLE_TTS="ON" \
  -DSHERPA_MNN_ENABLE_SPEAKER_DIARIZATION="ON" \
  -DSHERPA_MNN_ENABLE_BINARY="OFF" \
  -DSHERPA_MNN_ENABLE_C_API="OFF" \
  -DBUILD_SHARED_LIBS="ON" \
  -DMNN_LIB_DIR="/d/mojing/MNN/project/android/build_64" \
  -DCMAKE_INSTALL_PREFIX="./install" \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES="ON" \
  ..
```

之后再执行：
```bash
/d/AndroidSdk/cmake/3.22.1/bin/ninja.exe
/d/AndroidSdk/cmake/3.22.1/bin/ninja.exe install
```

4.
替换库文件（最关键的一步）： 编译完成后，不能只替换一个文件，必须将以下 3 个文件同时拷贝 到 Android 项目的 app/src/main/jniLibs/arm64-v8a/ 目录：
◦
libsherpa-mnn-jni.so（新编出来的）
◦
libMNN.so（本地 build_64 目录下的）
◦
libMNN_Express.so（本地 build_64 目录下的）

---

## 7. ASR 使用与集成要点
*   **模型下载与缓存路径**：应用会优先请求外部存储的“所有文件访问”权限以将模型持久化到 `/MnnModels`，否则退回到应用私有缓存目录（`.mnnmodels`）。见 [setupDownloaderAndStart](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L124-L132)。
*   **语言自适应**：根据系统语言自动选择中英双语/英文 ASR 模型，见 [asrModelId](file:///d:/mojing/MNN/apps/Android/MnnMultiModalAudioDemo/app/src/main/java/com/alibaba/mnnllm/multimodal/audio/MainActivity.kt#L28-L36)。
*   **权限**：首次点击 ASR 按钮会申请录音权限；权限允许后再次点击即可开始录音。
*   **UI 元素**：`activity_main.xml` 新增 `btn_asr_floating` 悬浮按钮与 `recording_indicator` 录音指示层、`waveform_view` 波形控件，便于录音状态可视化。
*   **数据流转**：ASR 识别完成后将文本加入聊天并作为 Prompt 发送至 LLM，形成“语音 → 文本 → LLM”的闭环。
