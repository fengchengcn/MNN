@echo off
REM ============================================================
REM Push Qwen3-ASR-0.6B-INT8-MNN model to Android device
REM Uses new llmexport.py naming: audio.mnn, llm.mnn, llm.mnn.weight
REM Target location auto-detected by the app.
REM ============================================================

setlocal enabledelayedexpansion

set MODEL_SRC=D:\mojing\MNN\mnn-models\Qwen3-ASR-0.6B-INT8-MNN
set DEVICE_DIR=/data/local/tmp/mnn_models/Qwen3-ASR-0.6B-INT8-MNN

echo ============================================================
echo Pushing Qwen3-ASR-0.6B-INT8-MNN model to Android device...
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
    exit /b 1
)

echo Device connected. Creating target directory...
adb shell mkdir -p %DEVICE_DIR% 2>nul

echo.
echo Pushing model files...

echo [1/5] audio.mnn (2.85 MB^)...
adb push "%MODEL_SRC%\audio.mnn" "%DEVICE_DIR%/audio.mnn"

echo [2/5] llm.mnn (494 KB^)...
adb push "%MODEL_SRC%\llm.mnn" "%DEVICE_DIR%/llm.mnn"

echo [3/5] llm.mnn.weight (604.4 MB^)...
adb push "%MODEL_SRC%\llm.mnn.weight" "%DEVICE_DIR%/llm.mnn.weight"

echo [4/5] embeddings_bf16.bin (296.8 MB^)...
adb push "%MODEL_SRC%\embeddings_bf16.bin" "%DEVICE_DIR%/embeddings_bf16.bin"

echo [5/5] tokenizer.txt (3.05 MB^)...
adb push "%MODEL_SRC%\tokenizer.txt" "%DEVICE_DIR%/tokenizer.txt"

echo.
echo ============================================================
echo Pushing optional config files (not required by engine)...
echo ============================================================
adb push "%MODEL_SRC%\llm.mnn.json" "%DEVICE_DIR%/llm.mnn.json" 2>nul
adb push "%MODEL_SRC%\llm_config.json" "%DEVICE_DIR%/llm_config.json" 2>nul
adb push "%MODEL_SRC%\config.json" "%DEVICE_DIR%/config.json" 2>nul

echo.
echo ============================================================
echo Verifying deployed files...
echo ============================================================
adb shell ls -la %DEVICE_DIR%/

echo.
echo ============================================================
echo Done! The model should be auto-detected by MnnLlmChat.
echo Run the app and check Qwen3AsrTestActivity or Voice Chat.
echo ============================================================
