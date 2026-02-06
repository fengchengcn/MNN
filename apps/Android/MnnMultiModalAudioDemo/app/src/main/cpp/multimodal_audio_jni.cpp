#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include <android/log.h>
#include <mutex>
#include <sstream>
#include <functional>
#include <dirent.h>
#include <sys/types.h>
#include <llm/llm.hpp>
#include "omni.hpp"
#include "llmconfig.hpp"

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
static std::shared_ptr<MNN::Transformer::Llm> g_llm;
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
    

    std::string configPath = modelDirStr;
    if (modelDirStr.back() != '/') {
        configPath += "/";
    }
    
    // Attempt to find specific config file to ensure LlmConfig loads it correctly
    // Priority: llm_config.json > config.json > directory
    std::string llmConfigPath = configPath + "llm_config.json";
    std::string finalConfigArg = modelDirStr;

    // Simple file existence check using FILE* as dirent approach is already doing listing
    FILE* fp = fopen(llmConfigPath.c_str(), "r");
    if (fp) {
        fclose(fp);
        finalConfigArg = llmConfigPath;
        LOGI("nativeInit: Found llm_config.json, using specific config path: %s", finalConfigArg.c_str());
    } else {
        std::string wrapperConfigPath = configPath + "config.json";
        fp = fopen(wrapperConfigPath.c_str(), "r");
        if (fp) {
            fclose(fp);
            finalConfigArg = wrapperConfigPath;
             LOGI("nativeInit: Found config.json, using specific config path: %s", finalConfigArg.c_str());
        } else {
             LOGI("nativeInit: No specific config file found, using directory path: %s", finalConfigArg.c_str());
        }
    }
    
    try {
        // FORCE Creation of Omni instance using manually loaded config
        // This ensures valid Audio support regardless of precompiled libllm.so settings
        std::shared_ptr<MNN::Transformer::LlmConfig> config(new MNN::Transformer::LlmConfig(finalConfigArg));
        g_llm.reset(new MNN::Transformer::Omni(config));

        if (!g_llm) {
            LOGE("nativeInit: Failed to create Omni instance (new Omni returned null)");
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
    
    // Fix for repetition loop: Apply robust sampling parameters
    // penalty (presence penalty), n_gram & ngram_factor (penalize repeating sequences)
    const char* sampler_config = "{\"sampler_type\": \"mixed\", "
                                 "\"temperature\": 0.8, "
                                 "\"topK\": 40, "
                                 "\"topP\": 0.8, "
                                 "\"penalty\": 1.2, "
                                 "\"n_gram\": 3, "
                                 "\"ngram_factor\": 1.5}";
    g_llm->set_config(sampler_config);
    LOGI("nativeInit: Applied sampler config: %s", sampler_config);

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeChat(
        JNIEnv* env,
        jobject thiz,
        jobjectArray history) {
    
    std::shared_ptr<MNN::Transformer::Llm> llm;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        llm = g_llm;
    }
    
    if (!llm) {
        LOGE("nativeChat: LLM model not initialized or creation failed");
        return;
    }

    // Convert Java Array to ChatMessages
    MNN::Transformer::ChatMessages chat_messages;
    jsize len = env->GetArrayLength(history);
    for (jsize i = 0; i < len; i += 2) {
        jstring jRole = (jstring)env->GetObjectArrayElement(history, i);
        jstring jContent = (jstring)env->GetObjectArrayElement(history, i + 1);
        
        const char* cRole = env->GetStringUTFChars(jRole, nullptr);
        const char* cContent = env->GetStringUTFChars(jContent, nullptr);
        
        chat_messages.push_back({std::string(cRole), std::string(cContent)});
        
        env->ReleaseStringUTFChars(jRole, cRole);
        env->ReleaseStringUTFChars(jContent, cContent);
        env->DeleteLocalRef(jRole);
        env->DeleteLocalRef(jContent);
    }

    // Inject System Prompt if not present
    bool has_system = false;
    if (!chat_messages.empty() && chat_messages[0].first == "system") {
        has_system = true;
    }
    
    if (!has_system) {
        std::string system_prompt = "You are a helpful assistant. Please provide concise and direct answers. "
                                    "Avoid repeating the same sentences or phrases in your response. "
                                    "If you have finished your thought, stop immediately without circular talk.";

        chat_messages.insert(chat_messages.begin(), {"system", system_prompt});
    }
    
    LOGI("nativeChat: Start multi-turn generation. History size: %zu", chat_messages.size());
    
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_callback_obj == nullptr) {
            g_callback_obj = env->NewGlobalRef(thiz);
            jclass clazz = env->GetObjectClass(thiz);
            g_callback_method = env->GetMethodID(clazz, "onChatStreamUpdate", "(Ljava/lang/String;)V");
        }
    }
    
    std::string cumulative_response;
    LlmStreamBuffer stream_buffer([&cumulative_response](const char* str, size_t len) {
        std::string chunk(str, len);
        cumulative_response += chunk;
        notifyJava(chunk);
    });
    std::ostream output_stream(&stream_buffer);
    
    try {
        std::string full_prompt = llm->apply_chat_template(chat_messages);
        LOGI("nativeChat: Full prompt (first 500 chars): %.500s", full_prompt.c_str());

        std::vector<int> input_ids = llm->tokenizer_encode(full_prompt);
        LOGI("nativeChat: Tokenized %zu tokens from prompt", input_ids.size());
        
        llm->response(input_ids, &output_stream, nullptr, 512);
    } catch (const std::exception& e) {
        LOGE("nativeChat: Exception: %s", e.what());
    }

    // Notify finished
    jclass clazz = env->GetObjectClass(thiz);
    jmethodID finish_method = env->GetMethodID(clazz, "onChatFinished", "(Ljava/lang/String;)V");
    if (finish_method) {
        jstring jFullResponse = env->NewStringUTF(cumulative_response.c_str());
        env->CallVoidMethod(thiz, finish_method, jFullResponse);
        env->DeleteLocalRef(jFullResponse);
    }
    
    LOGI("nativeChat: Generation finished.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeReset(JNIEnv *env, jobject thiz) {
    LOGI("nativeReset called");
    std::shared_ptr<MNN::Transformer::Llm> llm;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        llm = g_llm;
    }
    
    if (llm) {
        // Cancel first to interrupt any ongoing generation in other threads
        static_cast<MNN::Transformer::Omni*>(llm.get())->cancel();
        
        // Then reset history
        std::lock_guard<std::mutex> lock(g_mutex);
        llm->reset();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_alibaba_mnnllm_multimodal_audio_MainActivity_nativeRelease(JNIEnv *env, jobject thiz) {
    LOGI("nativeRelease called");
    std::shared_ptr<MNN::Transformer::Llm> llm;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        llm = g_llm;
        g_llm.reset();
    }
    
    if (llm) {
        static_cast<MNN::Transformer::Omni*>(llm.get())->cancel();
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_callback_obj) {
        env->DeleteGlobalRef(g_callback_obj);
        g_callback_obj = nullptr;
        g_callback_method = nullptr;
    }
}
