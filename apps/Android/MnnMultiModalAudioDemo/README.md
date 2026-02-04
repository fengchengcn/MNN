# MNN MultiModal Audio Demo (MNN 多模态语音交互 Demo)

这是一个基于 MNN 框架开发的 Android 应用程序，演示了如何在端侧设备上运行 Qwen2.5-Omni 多模态大模型。该应用支持语音输入（录音）和图像输入，并能通过语音（TTS）和文本实时反馈 AI 的回答。

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
│   │       ├── cpp/              # C++ 源代码 (核心推理引擎)
│   │       │   ├── CMakeLists.txt             # C++ 构建脚本 (定义编译规则)
│   │       │   └── multimodal_audio_jni.cpp   # JNI 接口实现 (Java与MNN的桥梁)
│   │       │
│   │       ├── res/              # 资源文件 (布局、图片、字符串等)
│   │       │   ├── layout/
│   │       │   │   └── activity_main.xml      # 主界面布局文件 (类似于 HTML)
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
    *   检查本地目录 `.mnnmodels/MNN/Qwen2.5-Omni-3B-MNN` 是否存在且完整。
    *   **不存在**: 显示下载进度，调用 `ModelDownloadManager` 下载模型 (约 4GB)。
    *   **存在**: 调用 Native 方法 `loadModel()` 加载模型进入内存。

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
