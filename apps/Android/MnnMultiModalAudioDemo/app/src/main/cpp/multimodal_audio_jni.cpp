#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llm/llm.hpp"
#include "audio/audio.hpp"

#define TAG "MnnMultiModalAudio_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
