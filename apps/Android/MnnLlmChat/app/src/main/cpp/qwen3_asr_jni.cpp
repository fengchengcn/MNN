// Qwen3-ASR JNI Bridge for Android
// Provides native methods for Kotlin/Java to call Qwen3AsrEngine
#include <jni.h>
#include <string>
#include <android/log.h>
#include "qwen3_asr_engine.h"

#define LOG_TAG "Qwen3AsrJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

// Native pointer storage
static jfieldID getNativePtrField(JNIEnv* env, jobject thiz) {
    jclass clazz = env->GetObjectClass(thiz);
    return env->GetFieldID(clazz, "mNativePtr", "J");
}

static Qwen3AsrEngine* getNativePtr(JNIEnv* env, jobject thiz) {
    jfieldID field = getNativePtrField(env, thiz);
    if (!field) return nullptr;
    return reinterpret_cast<Qwen3AsrEngine*>(env->GetLongField(thiz, field));
}

static void setNativePtr(JNIEnv* env, jobject thiz, Qwen3AsrEngine* ptr) {
    jfieldID field = getNativePtrField(env, thiz);
    if (field) env->SetLongField(thiz, field, reinterpret_cast<jlong>(ptr));
}

/**
 * Initialize the engine.
 * @param model_dir Path to model directory
 * @param num_threads Number of inference threads
 * @return true on success
 */
JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeInit(
    JNIEnv* env, jobject thiz, jstring model_dir, jstring cache_dir, jint num_threads) {

    const char* dir_chars = env->GetStringUTFChars(model_dir, nullptr);
    std::string dir(dir_chars);
    env->ReleaseStringUTFChars(model_dir, dir_chars);

    const char* cache_chars = env->GetStringUTFChars(cache_dir, nullptr);
    std::string cache(cache_chars);
    env->ReleaseStringUTFChars(cache_dir, cache_chars);

    LOGI("Initializing engine with model dir: %s, cache: %s, threads: %d",
         dir.c_str(), cache.c_str(), (int)num_threads);

    auto* engine = new Qwen3AsrEngine();
    if (!engine->init(dir, cache, (int)num_threads)) {
        LOGE("Engine initialization failed");
        delete engine;
        return JNI_FALSE;
    }

    setNativePtr(env, thiz, engine);
    LOGI("Engine initialized OK");
    return JNI_TRUE;
}

/**
 * Push PCM audio samples (16kHz, mono, float [-1.0, 1.0]).
 */
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativePushAudio(
    JNIEnv* env, jobject thiz, jfloatArray pcm_data) {

    auto* engine = getNativePtr(env, thiz);
    if (!engine) {
        LOGE("pushAudio: engine not initialized");
        return;
    }

    jsize len = env->GetArrayLength(pcm_data);
    jfloat* buf = env->GetFloatArrayElements(pcm_data, nullptr);
    engine->pushAudio(buf, (int)len);
    env->ReleaseFloatArrayElements(pcm_data, buf, JNI_ABORT);
}

/**
 * Signal end of audio and run decoder.
 */
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeEndAudio(
    JNIEnv* env, jobject thiz) {

    auto* engine = getNativePtr(env, thiz);
    if (!engine) {
        LOGE("endAudio: engine not initialized");
        return;
    }
    engine->endAudio();
    LOGI("endAudio completed");
}

/**
 * Get current transcription result (token IDs as space-separated string).
 */
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeGetResult(
    JNIEnv* env, jobject thiz) {

    auto* engine = getNativePtr(env, thiz);
    if (!engine) {
        return env->NewStringUTF("");
    }
    std::string result = engine->getResult();
    return env->NewStringUTF(result.c_str());
}

/**
 * Get decoded text result (requires tokenizer.txt in model directory).
 */
JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeGetResultText(
    JNIEnv* env, jobject thiz) {

    auto* engine = getNativePtr(env, thiz);
    if (!engine) {
        return env->NewStringUTF("");
    }
    std::string result = engine->getResultText();
    return env->NewStringUTF(result.c_str());
}

/**
 * Reset engine state for new utterance (keeps models loaded).
 */
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeReset(
    JNIEnv* env, jobject thiz) {

    auto* engine = getNativePtr(env, thiz);
    if (engine) {
        engine->reset();
        LOGI("Engine reset");
    }
}

/**
 * Release all resources and delete engine.
 */
JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_android_asr_Qwen3AsrEngine_nativeRelease(
    JNIEnv* env, jobject thiz) {

    auto* engine = getNativePtr(env, thiz);
    if (engine) {
        engine->release();
        delete engine;
        setNativePtr(env, thiz, nullptr);
        LOGI("Engine released");
    }
}

} // extern "C"
