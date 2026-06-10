@echo off
REM ============================================================
REM Push Qwen3-ASR model (old-engine) to Android device
REM Target location auto-detected by the app's VoiceModelPathUtils
REM ============================================================

setlocal enabledelayedexpansion

set MODEL_SRC=D:\mojing\MNN\mnn-models\Qwen3-ASR-sherpa-onnx-old-engine
set DEVICE_DIR=/data/local/tmp/mnn_models/Qwen3-ASR-sherpa-onnx-old-engine

echo ============================================================
echo Pushing Qwen3-ASR model (old-engine) to Android device...
echo ============================================================

REM Check adb
where adb >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] adb not found in PATH. Please install Android SDK platform-tools.
    exit /b 1
)

REM Check device
adb get-state >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] No Android device connected. Please connect a device and enable USB debugging.
    echo.
    echo Expected device path: %DEVICE_DIR%
    echo Files to push (5 files, total ~1.15 GB^):
    echo   audio_encoder.mnn    210.5 MB
    echo   llm_kv_8bit.mnn        463.9 KB
    echo   llm_kv_8bit.mnn.weight 641.9 MB
    echo   tokenizer.txt          930.1 KB
    echo   embeddings_bf16.bin   296.8 MB
    exit /b 1
)

echo Device connected. Creating target directory...
adb shell mkdir -p %DEVICE_DIR% 2>nul

echo.
echo Pushing model files...

echo [1/5] audio_encoder.mnn (210.5 MB^)...
adb push "%MODEL_SRC%\audio_encoder.mnn" "%DEVICE_DIR%/audio_encoder.mnn"

echo [2/5] llm_kv_8bit.mnn (463.9 KB^)...
adb push "%MODEL_SRC%\llm_kv_8bit.mnn" "%DEVICE_DIR%/llm_kv_8bit.mnn"

echo [3/5] llm_kv_8bit.mnn.weight (641.9 MB^)...
adb push "%MODEL_SRC%\llm_kv_8bit.mnn.weight" "%DEVICE_DIR%/llm_kv_8bit.mnn.weight"

echo [4/5] tokenizer.txt (930.1 KB^)...
adb push "%MODEL_SRC%\tokenizer.txt" "%DEVICE_DIR%/tokenizer.txt"

echo [5/5] embeddings_bf16.bin (296.8 MB^)...
adb push "%MODEL_SRC%\embeddings_bf16.bin" "%DEVICE_DIR%/embeddings_bf16.bin"

echo.
echo ============================================================
echo Verifying deployed files...
echo ============================================================
adb shell ls -la %DEVICE_DIR%/

echo.
echo ============================================================
echo Done! The model should be auto-detected by MnnLlmChat.
echo Restart the app and check Voice Chat settings.
echo ============================================================
