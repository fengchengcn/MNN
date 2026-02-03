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
        if (callback_) {
            callback_(s, n);
        }
        return n;
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
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_loadModel(
        JNIEnv* env,
        jobject thiz,
        jstring model_dir) {
    
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    std::string modelDirStr(c_model_dir);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);

    std::lock_guard<std::mutex> lock(g_mutex);
    
    // Initialize LLM (Qwen2.5-Omni)
    LOGI("Loading Omni Model from: %s", modelDirStr.c_str());
    
    // MNN Llm::createLLM automatically detects config.json and handles multimodal (audio/vision)
    g_llm.reset(MNN::Transformer::Llm::createLLM(modelDirStr.c_str()));
    
    if (!g_llm) {
        LOGE("Failed to create LLM");
        return JNI_FALSE;
    }
    
    g_llm->load(); 
    LOGI("Omni Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeChat(
        JNIEnv* env,
        jobject thiz,
        jstring prompt) {
    if (!g_llm) {
        LOGE("LLM model not initialized");
        return JNI_FALSE;
    }
    
    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string promptStr(c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);
    
    LOGI("nativeChat prompt: %s", promptStr.c_str());
    
    // Cache callback object
    if (g_callback_obj == nullptr) {
        g_callback_obj = env->NewGlobalRef(thiz);
        jclass clazz = env->GetObjectClass(thiz);
        g_callback_method = env->GetMethodID(clazz, "onChatStreamUpdate", "(Ljava/lang/String;)V");
    }
    
    std::thread([promptStr]() {
        if (!g_llm) {
            LOGE("LLM became null in thread");
            return;
        }
        
        // Create custom stream buffer to capture output
        std::string accumulated_output;
        LlmStreamBuffer stream_buffer([&accumulated_output](const char* str, size_t len) {
            std::string chunk(str, len);
            accumulated_output += chunk;
            // Notify Java with each chunk
            notifyJava(chunk);
        });
        std::ostream output_stream(&stream_buffer);
        
        // Build MultimodalPrompt if we have <img> or <audio> tags
        MNN::Transformer::MultimodalPrompt multimodal_prompt;
        bool has_multimodal = false;
        std::string processed_prompt = promptStr;
        
        // Parse <img>...</img> tags
        size_t img_start = processed_prompt.find("<img>");
        while (img_start != std::string::npos) {
            size_t img_end = processed_prompt.find("</img>", img_start);
            if (img_end != std::string::npos) {
                size_t path_start = img_start + 5; // length of "<img>"
                std::string img_path = processed_prompt.substr(path_start, img_end - path_start);
                
                // Generate placeholder key
                std::string placeholder = "<image_" + std::to_string(multimodal_prompt.images.size()) + ">";
                
                // Load image
                MNN::Transformer::PromptImagePart image_part;
                // Note: For image loading, you would typically use MNN's image loading utilities
                // For now, we'll just store the path info
                image_part.width = 0;
                image_part.height = 0;
                multimodal_prompt.images[placeholder] = image_part;
                
                // Replace in processed prompt
                processed_prompt.replace(img_start, img_end - img_start + 6, placeholder);
                has_multimodal = true;
                LOGI("Found image: %s -> %s", img_path.c_str(), placeholder.c_str());
            }
            img_start = processed_prompt.find("<img>", img_start + 1);
        }
        
        // Parse <audio>...</audio> tags
        size_t audio_start = processed_prompt.find("<audio>");
        while (audio_start != std::string::npos) {
            size_t audio_end = processed_prompt.find("</audio>", audio_start);
            if (audio_end != std::string::npos) {
                size_t path_start = audio_start + 7; // length of "<audio>"
                std::string audio_path = processed_prompt.substr(path_start, audio_end - path_start);
                
                // Generate placeholder key
                std::string placeholder = "<audio_" + std::to_string(multimodal_prompt.audios.size()) + ">";
                
                // Create audio part with file path
                MNN::Transformer::PromptAudioPart audio_part;
                audio_part.file_path = audio_path;
                multimodal_prompt.audios[placeholder] = audio_part;
                
                // Replace in processed prompt  
                processed_prompt.replace(audio_start, audio_end - audio_start + 8, placeholder);
                has_multimodal = true;
                LOGI("Found audio: %s -> %s", audio_path.c_str(), placeholder.c_str());
            }
            audio_start = processed_prompt.find("<audio>", audio_start + 1);
        }
        
        if (has_multimodal) {
            multimodal_prompt.prompt_template = processed_prompt;
            LOGI("Using multimodal API with template: %s", processed_prompt.c_str());
            g_llm->response(multimodal_prompt, &output_stream, nullptr, -1);
        } else {
            // Simple text response using ChatMessages format
            std::vector<std::pair<std::string, std::string>> history;
            history.emplace_back("user", promptStr);
            LOGI("Using text API");
            g_llm->response(history, &output_stream, nullptr, -1);
        }
        
        LOGI("Chat generation finished. Output: %s", accumulated_output.c_str());
    }).detach();

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeReset(JNIEnv *env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_llm) {
        g_llm->reset();
    }
}
