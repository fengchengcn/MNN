// Qwen3-ASR Engine — C++ inference class for Android JNI
// Encapsulates audio encoder + decoder with KV cache + repetition penalty
#ifndef QWEN3_ASR_ENGINE_H
#define QWEN3_ASR_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

class Qwen3AsrEngine {
public:
    Qwen3AsrEngine();
    ~Qwen3AsrEngine();

    // Initialize engine: load models from directory
    // model_dir must contain: audio_encoder.mnn, llm_kv_8bit.mnn,
    //   llm_kv_8bit.mnn.weight, embeddings_bf16.bin
    bool init(const std::string& model_dir, int num_threads = 4);

    // Feed PCM float samples (16kHz mono, normalized to [-1, 1])
    // Returns true if audio was accepted
    bool pushAudio(const float* samples, int num_samples);

    // Signal end of audio input, run decoder to get final result
    void endAudio();

    // Get current transcription result (space-separated token IDs)
    std::string getResult() const;

    // Get decoded text result (requires tokenizer.txt in model directory)
    std::string getResultText() const;

    // Returns true if the engine has been initialized
    bool isInitialized() const { return m_initialized; }

    // Reset engine state for new utterance (keeps models loaded)
    void reset();

    // Release all resources
    void release();

private:
    // Internal: run decoder inference on accumulated audio
    std::string runDecoder();

    // Model handles
    std::shared_ptr<MNN::Express::Module> m_audio_mod;     // audio_encoder.mnn
    std::shared_ptr<MNN::Express::Module> m_llm_mod;       // llm_kv_8bit.mnn
    MNN::Express::VARP m_embed_tbl;                         // embeddings_bf16.bin

    // Decoder state (KV cache)
    MNN::Express::VARP m_k_cache;
    MNN::Express::VARP m_v_cache;

    // Token history
    std::vector<int> m_token_ids;
    int m_prefill_token_count;

    // Audio buffer (accumulated across pushAudio calls)
    std::vector<float> m_audio_buffer;

    // Runtime manager for external weights
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> m_rt;

    // Tokenizer: id → token string (loaded from tokenizer.txt)
    std::vector<std::string> m_token_table;

    // Config
    std::string m_model_dir;
    int m_num_threads;
    bool m_initialized;
    bool m_decoder_ran;

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

    // Helper methods
    static float bf16_to_f32(uint16_t v);
    MNN::Express::VARP loadEmbedding(const std::string& path);
    MNN::Express::VARP embedLookup(MNN::Express::VARP tbl, const std::vector<int>& ids);
    static int argmaxPenalized(const float* logits, const std::vector<int>& history, float penalty);
    MNN::Express::VARP createCausalMask(int S_new, int past_len);
    std::vector<MNN::Express::VARP> createEmptyCache();
};

#endif // QWEN3_ASR_ENGINE_H
