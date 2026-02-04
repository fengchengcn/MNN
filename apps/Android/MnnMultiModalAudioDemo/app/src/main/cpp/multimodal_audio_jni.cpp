#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include <android/log.h>
#include <mutex>
#include <sstream>
#include <functional>
#include "llm/llm.hpp"

#define TAG "MnnMultiModalAudio_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Custom streambuf to capture streaming output
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char *str, size_t len)>;
    explicit LlmStreamBuffer(CallBack callback) : callback_(std::move(callback)) {}

protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override {
        if (callback_ && n > 0) {
            callback_(s, n);
        }
        return n;
    }

    int overflow(int c) override {
        if (c != EOF && callback_) {
            char ch = static_cast<char>(c);
            callback_(&ch, 1);
        }
        return c;
    }

private:
    CallBack callback_ = nullptr;
};

// Global model pointer with thread safety
static std::unique_ptr<MNN::Transformer::Llm> g_llm;
static std::mutex g_mutex;

static JavaVM* g_jvm = nullptr;
static jobject g_callback_obj = nullptr;
static jmethodID g_callback_method = nullptr;

// Helper to notify Java callback
void notifyJava(const std::string& chunk) {
    if (g_jvm == nullptr || g_callback_obj == nullptr || g_callback_method == nullptr) {
        return;
    }
    JNIEnv* env;
    bool attached = false;
    int status = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (status < 0) {
        status = g_jvm->AttachCurrentThread(&env, nullptr);
        if (status < 0) return;
        attached = true;
    }
    
    jstring jChunk = env->NewStringUTF(chunk.c_str());
    env->CallVoidMethod(g_callback_obj, g_callback_method, jChunk);
    env->DeleteLocalRef(jChunk);
    
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    LOGI("JNI_OnLoad called");
    return JNI_VERSION_1_6;
}

#include <dirent.h>

extern "C" JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeInit(
        JNIEnv* env,
        jobject thiz,
        jstring model_dir) {
    
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    std::string modelDirStr(c_model_dir);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);

    std::lock_guard<std::mutex> lock(g_mutex);
    
    LOGI("nativeInit: Loading Omni Model from path: %s", modelDirStr.c_str());

    // Debug: List files in the directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(modelDirStr.c_str())) != NULL) {
        LOGI("nativeInit: Contents of directory %s:", modelDirStr.c_str());
        while ((ent = readdir(dir)) != NULL) {
            LOGI("  - %s", ent->d_name);
        }
        closedir(dir);
    } else {
        LOGE("nativeInit: Could not open directory %s", modelDirStr.c_str());
    }
    
    try {
        g_llm.reset(MNN::Transformer::Llm::createLLM(modelDirStr.c_str()));
        if (!g_llm) {
            LOGE("nativeInit: Failed to create LLM instance (createLLM returned null)");
            return JNI_FALSE;
        }
        
        LOGI("nativeInit: Calling g_llm->load()...");
        bool load_res = g_llm->load();
        LOGI("nativeInit: g_llm->load() result: %d", load_res);
        
        if (!load_res) {
            LOGE("nativeInit: Failed to load model weights/assets");
            return JNI_FALSE;
        }
    } catch (const std::exception& e) {
        LOGE("nativeInit: Exception during model loading: %s", e.what());
        return JNI_FALSE;
    }
    
    LOGI("nativeInit: Omni Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeChat(
        JNIEnv* env,
        jobject thiz,
        jstring prompt) {
    
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_llm) {
        LOGE("nativeChat: LLM model not initialized or creation failed");
        return;
    }
    
    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string promptStr(c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);
    
    LOGI("nativeChat: Start generation. Prompt length: %zu. Prompt: %s", promptStr.size(), promptStr.c_str());
    
    // Cache callback object
    if (g_callback_obj == nullptr) {
        g_callback_obj = env->NewGlobalRef(thiz);
        jclass clazz = env->GetObjectClass(thiz);
        g_callback_method = env->GetMethodID(clazz, "onChatStreamUpdate", "(Ljava/lang/String;)V");
        LOGI("nativeChat: Registered Java callback");
    }
    
    // Create custom stream buffer to capture output
    LlmStreamBuffer stream_buffer([](const char* str, size_t len) {
        std::string chunk(str, len);
        LOGI("nativeChat: Received chunk: %s", chunk.c_str());
        notifyJava(chunk);
    });
    std::ostream output_stream(&stream_buffer);
    
    // Build MultimodalPrompt if we have <img> or <audio> tags
    MNN::Transformer::MultimodalPrompt multimodal_prompt;
    bool has_multimodal = false;
    std::string processed_prompt = promptStr;
    
    // Simple <img> parser (for demo purpose)
    size_t img_start = processed_prompt.find("<img>");
    while (img_start != std::string::npos) {
        size_t img_end = processed_prompt.find("</img>", img_start);
        if (img_end != std::string::npos) {
            size_t path_start = img_start + 5;
            std::string img_path = processed_prompt.substr(path_start, img_end - path_start);
            std::string placeholder = "<image_" + std::to_string(multimodal_prompt.images.size()) + ">";
            
            MNN::Transformer::PromptImagePart image_part;
            image_part.width = 0;
            image_part.height = 0;
            multimodal_prompt.images[placeholder] = image_part;
            
            processed_prompt.replace(img_start, img_end - img_start + 6, placeholder);
            has_multimodal = true;
            LOGI("nativeChat: Found image: %s -> %s", img_path.c_str(), placeholder.c_str());
        }
        img_start = processed_prompt.find("<img>", img_start + 1);
    }
    
    // Simple <audio> parser
    size_t audio_start = processed_prompt.find("<audio>");
    while (audio_start != std::string::npos) {
        size_t audio_end = processed_prompt.find("</audio>", audio_start);
        if (audio_end != std::string::npos) {
            size_t path_start = audio_start + 7;
            std::string audio_path = processed_prompt.substr(path_start, audio_end - path_start);
            std::string placeholder = "<audio_" + std::to_string(multimodal_prompt.audios.size()) + ">";
            
            MNN::Transformer::PromptAudioPart audio_part;
            audio_part.file_path = audio_path;
            multimodal_prompt.audios[placeholder] = audio_part;
            
            processed_prompt.replace(audio_start, audio_end - audio_start + 8, placeholder);
            has_multimodal = true;
            LOGI("nativeChat: Found audio: %s -> %s", audio_path.c_str(), placeholder.c_str());
        }
        audio_start = processed_prompt.find("<audio>", audio_start + 1);
    }
    
    try {
        if (has_multimodal) {
            multimodal_prompt.prompt_template = processed_prompt;
            LOGI("nativeChat: Using multimodal API with template: %s", processed_prompt.c_str());
            g_llm->response(multimodal_prompt, &output_stream, nullptr, -1);
        } else {
            LOGI("nativeChat: Using text API");
            g_llm->response(promptStr, &output_stream, nullptr, -1);
        }
    } catch (const std::exception& e) {
        LOGE("nativeChat: Exception during response generation: %s", e.what());
    }
    
    LOGI("nativeChat: Generation finished.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeRelease(JNIEnv *env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    LOGI("nativeRelease called");
    if (g_llm) {
        g_llm.reset();
    }
    if (g_callback_obj) {
        env->DeleteGlobalRef(g_callback_obj);
        g_callback_obj = nullptr;
        g_callback_method = nullptr;
    }
}
