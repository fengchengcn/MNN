// Qwen3-ASR Engine — C++ inference class for Android JNI
// Encapsulates audio encoder + decoder with KV cache + repetition penalty
// Memory-optimized: mmap embeddings, lazy model loading, on-demand audio encoder
#ifndef QWEN3_ASR_ENGINE_H
#define QWEN3_ASR_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

// POSIX mmap for embedding file (Android/Linux)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

class Qwen3AsrEngine {
public:
    Qwen3AsrEngine();
    ~Qwen3AsrEngine();

    // Initialize engine: load tokenizer + mmap embeddings, models on-demand
    // cache_dir: app's internal cache dir for temp files (e.g. /data/data/.../cache)
    bool init(const std::string& model_dir, const std::string& cache_dir, int num_threads = 2);

    // Feed PCM float samples (16kHz mono, normalized to [-1, 1])
    bool pushAudio(const float* samples, int num_samples);

    // Signal end of audio input, run decoder to get final result
    void endAudio();

    // Get current transcription result
    std::string getResult() const;
    std::string getResultText() const;

    bool isInitialized() const { return m_initialized; }

    // Reset engine state for new utterance (keeps decoder loaded, clears audio+KV cache)
    void reset();

    // Release all resources
    void release();

private:
    std::string runDecoder();

    // Embedding file management (mmap-based)
    bool openEmbeddingFile(const std::string& path);
    void closeEmbeddingFile();
    void embedLookup(const std::vector<int>& ids, float* dst);

    // Model loading (on-demand, serial: AE loaded → used → freed, then decoder loaded)
    std::shared_ptr<MNN::Express::Module> loadAudioEncoder();
    bool ensureDecoderLoaded();

    // LLM decoder (loaded once, kept across utterances)
    std::shared_ptr<MNN::Express::Module> m_llm_mod;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> m_rt;
    bool m_decoder_loaded;

    // Decoder state (KV cache)
    MNN::Express::VARP m_k_cache;
    MNN::Express::VARP m_v_cache;

    // Token history
    std::vector<int> m_token_ids;
    int m_prefill_token_count;

    // Audio buffer
    std::vector<float> m_audio_buffer;

    // Tokenizer (MNN SentencePiece/Tiktoken proper decoder)
    void* m_tokenizer = nullptr;  // MNN::Transformer::Tokenizer* (opaque to avoid header dep)
    // Prompt token cache (built in init() using tokenizer->encode())
    std::vector<int> m_prefix_tokens;   // system + user prefix
    std::vector<int> m_suffix_tokens;   // audio_end + assistant suffix
    void buildPromptTokens();

    // Config
    std::string m_model_dir;
    std::string m_cache_dir;
    int m_num_threads;
    bool m_initialized;
    bool m_decoder_ran;

    // Embedding via mmap
    int m_embed_fd = -1;
    void* m_embed_mmap = MAP_FAILED;
    size_t m_embed_file_size = 0;

    // Reusable buffers
    std::vector<float> m_penalty_buf;

    // Constants
    static constexpr int HIDDEN = 1024;
    static constexpr int VOCAB = 151936;
    static constexpr int EOS_TOKEN = 151643;
    static constexpr int IM_END_TOKEN = 151645;
    static constexpr int AUDIO_START = 151669;
    static constexpr int AUDIO_PAD = 151676;
    static constexpr int LAYERS = 28;
    static constexpr int KV_HEADS = 8;
    static constexpr int HEAD_DIM = 128;
    static constexpr float REP_PENALTY = 1.15f;
    static constexpr int MAX_NEW_TOKENS = 100;

    // Helpers
    static float bf16_to_f32(uint16_t v);
    static int argmaxPenalized(const float* logits, float* penalized_buf, const std::vector<int>& history, float penalty);
    MNN::Express::VARP createCausalMask(int S_new, int past_len);
    std::vector<MNN::Express::VARP> createEmptyCache();
};

#endif // QWEN3_ASR_ENGINE_H
