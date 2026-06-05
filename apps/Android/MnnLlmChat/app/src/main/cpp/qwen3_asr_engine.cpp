#include "qwen3_asr_engine.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <android/log.h>

#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif

#define LOG_TAG "Qwen3AsrEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace MNN::Express;

Qwen3AsrEngine::Qwen3AsrEngine()
    : m_num_threads(4)
    , m_initialized(false)
    , m_decoder_ran(false)
    , m_prefill_token_count(0) {
}

Qwen3AsrEngine::~Qwen3AsrEngine() {
    release();
}

float Qwen3AsrEngine::bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

VARP Qwen3AsrEngine::loadEmbedding(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return nullptr;
    size_t n = (size_t)VOCAB * HIDDEN;
    std::vector<uint16_t> buf(n);
    f.read((char*)buf.data(), n * 2);
    f.close();
    auto t = _Input({VOCAB, HIDDEN}, NCHW, halide_type_of<float>());
    float* d = t->writeMap<float>();
    for (size_t i = 0; i < n; i++) d[i] = bf16_to_f32(buf[i]);
    return t;
}

VARP Qwen3AsrEngine::embedLookup(VARP tbl, const std::vector<int>& ids) {
    int S = (int)ids.size();
    auto r = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* dst = r->writeMap<float>();
    const float* src = tbl->readMap<float>();
    for (int i = 0; i < S; i++) {
        int id = ids[i];
        if (id < 0 || id >= VOCAB) id = 0;
        memcpy(dst + i * HIDDEN, src + id * HIDDEN, HIDDEN * sizeof(float));
    }
    return r;
}

int Qwen3AsrEngine::argmaxPenalized(const float* logits, const std::vector<int>& history, float penalty) {
    if (penalty <= 1.0f || history.empty()) {
        int idx = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[idx]) idx = i;
        return idx;
    }
    std::vector<float> penalized(VOCAB);
    memcpy(penalized.data(), logits, VOCAB * sizeof(float));
    for (int id : history) {
        if (id < 0 || id >= VOCAB) continue;
        if (penalized[id] < 0)
            penalized[id] *= penalty;
        else
            penalized[id] /= penalty;
    }
    int idx = 0;
    for (int i = 1; i < VOCAB; i++) if (penalized[i] > penalized[idx]) idx = i;
    return idx;
}

VARP Qwen3AsrEngine::createCausalMask(int S_new, int past_len) {
    int S_total = past_len + S_new;
    auto mask = _Input({1, 1, S_new, S_total}, NCHW, halide_type_of<float>());
    float* mp = mask->writeMap<float>();
    for (int i = 0; i < S_new; i++) {
        for (int j = 0; j < S_total; j++) {
            mp[i * S_total + j] = (j <= past_len + i) ? 0.0f : -1e9f;
        }
    }
    return mask;
}

std::vector<VARP> Qwen3AsrEngine::createEmptyCache() {
    auto k = _Input({LAYERS, 1, KV_HEADS, 0, HEAD_DIM}, NCHW, halide_type_of<float>());
    auto v = _Input({LAYERS, 1, KV_HEADS, 0, HEAD_DIM}, NCHW, halide_type_of<float>());
    return {k, v};
}

bool Qwen3AsrEngine::init(const std::string& model_dir, int num_threads) {
    if (m_initialized) release();

    m_model_dir = model_dir;
    m_num_threads = num_threads;

    // Create executor with specified thread count
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, bc, num_threads);
    ExecutorScope scope(executor);

    // Load audio encoder
    LOGI("Loading audio encoder...");
    m_audio_mod = Module::load({}, {}, (model_dir + "/audio_encoder.mnn").c_str());
    if (!m_audio_mod) {
        LOGE("Failed to load audio encoder");
        return false;
    }
    LOGI("Audio encoder loaded");

    // Load decoder with KV cache
    LOGI("Loading LLM decoder (KV Cache)...");
    MNN::ScheduleConfig sched;
    sched.backendConfig = &bc;
    m_rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    m_rt->setExternalFile(model_dir + "/llm_kv_8bit.mnn.weight");
    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    m_llm_mod = Module::load({}, {}, (model_dir + "/llm_kv_8bit.mnn").c_str(), m_rt, &mc);
    if (!m_llm_mod) {
        LOGE("Failed to load LLM decoder");
        return false;
    }
    LOGI("LLM decoder loaded");

    // Load tokenizer
    LOGI("Loading tokenizer...");
    {
        std::ifstream tok_file(model_dir + "/tokenizer.txt");
        if (tok_file.is_open()) {
            std::string line;
            while (std::getline(tok_file, line)) {
                // Remove trailing newline/carriage return
                if (!line.empty() && line.back() == '\r') line.pop_back();
                m_token_table.push_back(line);
            }
            LOGI("Tokenizer loaded: %zu tokens", m_token_table.size());
        } else {
            LOGW("tokenizer.txt not found, text decoding disabled");
        }
    }

    // Load embeddings
    LOGI("Loading embeddings...");
    m_embed_tbl = loadEmbedding(model_dir + "/embeddings_bf16.bin");
    if (m_embed_tbl.get() == nullptr) {
        LOGE("Failed to load embeddings");
        return false;
    }
    LOGI("Embeddings loaded");

    m_initialized = true;
    LOGI("Qwen3-ASR Engine initialized successfully");
    return true;
}

void Qwen3AsrEngine::reset() {
    m_audio_buffer.clear();
    m_token_ids.clear();
    m_prefill_token_count = 0;
    m_decoder_ran = false;
    m_k_cache = VARP();
    m_v_cache = VARP();
}

void Qwen3AsrEngine::release() {
    reset();
    m_audio_mod.reset();
    m_llm_mod.reset();
    m_embed_tbl = VARP();
    m_rt.reset();
    m_initialized = false;
}

bool Qwen3AsrEngine::pushAudio(const float* samples, int num_samples) {
    if (!m_initialized) {
        LOGE("Engine not initialized");
        return false;
    }
    // Accumulate audio samples
    m_audio_buffer.insert(m_audio_buffer.end(), samples, samples + num_samples);
    return true;
}

void Qwen3AsrEngine::endAudio() {
    if (!m_initialized || m_audio_buffer.empty()) {
        LOGI("No audio to process");
        return;
    }
    runDecoder();
}

std::string Qwen3AsrEngine::getResult() const {
    // Decode token IDs to text using the embedding table
    // Note: full BPE decoding requires the tokenizer, which is in Java
    // Return space-separated token IDs for Java-side decoding
    if (m_token_ids.empty()) return "";
    std::string result;
    for (size_t i = 0; i < m_token_ids.size(); i++) {
        if (i > 0) result += " ";
        result += std::to_string(m_token_ids[i]);
    }
    return result;
}

std::string Qwen3AsrEngine::getResultText() const {
    if (m_token_ids.empty() || m_token_table.empty()) return getResult();
    std::string text;
    for (int id : m_token_ids) {
        if (id >= 0 && id < (int)m_token_table.size()) {
            text += m_token_table[id];
        } else {
            text += "[UNK:" + std::to_string(id) + "]";
        }
    }
    return text;
}

std::string Qwen3AsrEngine::runDecoder() {
    if (!m_initialized) return "";
    LOGI("Starting decoder with %zu audio samples", m_audio_buffer.size());

    // Recreate executor scope
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, bc, m_num_threads);
    ExecutorScope scope(executor);
    (void)scope;

    // ====== Audio processing ======
#ifdef LLM_SUPPORT_AUDIO
    // Load waveform from buffer
    int nsamples = (int)m_audio_buffer.size();
    // MNN AUDIO::load expects a file path - we need direct buffer processing
    // For now, write to temp file and load
    // TODO: use direct buffer API when available
    std::string tmp_wav = m_model_dir + "/_tmp_asr_input.wav";
    {
        // Write WAV header + PCM data
        std::ofstream wav_file(tmp_wav, std::ios::binary);
        if (!wav_file.is_open()) {
            LOGE("Failed to write temp WAV");
            return "";
        }
        // WAV header (44 bytes)
        int16_t fmt = 1;  // PCM
        int16_t channels = 1;
        int sample_rate = 16000;
        int16_t bits_per_sample = 16;
        int data_size = nsamples * 2;
        int16_t block_align = channels * bits_per_sample / 8;
        int byte_rate = sample_rate * block_align;

        wav_file.write("RIFF", 4);
        int32_t chunk_size = 36 + data_size;
        wav_file.write((char*)&chunk_size, 4);
        wav_file.write("WAVE", 4);
        wav_file.write("fmt ", 4);
        int32_t subchunk1_size = 16;
        wav_file.write((char*)&subchunk1_size, 4);
        wav_file.write((char*)&fmt, 2);
        wav_file.write((char*)&channels, 2);
        wav_file.write((char*)&sample_rate, 4);
        wav_file.write((char*)&byte_rate, 4);
        wav_file.write((char*)&block_align, 2);
        wav_file.write((char*)&bits_per_sample, 2);
        wav_file.write("data", 4);
        wav_file.write((char*)&data_size, 4);

        // Convert float PCM to int16
        for (int i = 0; i < nsamples; i++) {
            int16_t val = (int16_t)(m_audio_buffer[i] * 32767.0f);
            if (val > 32767) val = 32767;
            if (val < -32768) val = -32768;
            wav_file.write((char*)&val, 2);
        }
    }

    auto lr = MNN::AUDIO::load(tmp_wav, 16000);
    auto wf = lr.first;
    if (wf.get() == nullptr) {
        LOGE("Failed to load audio for fbank");
        std::remove(tmp_wav.c_str());
        return "";
    }
    std::remove(tmp_wav.c_str());

    auto feat = MNN::AUDIO::whisper_fbank(wf);
    if (feat.get() == nullptr || feat->getInfo() == nullptr) {
        LOGE("whisper_fbank failed");
        return "";
    }

    // Materialize fbank tensor
    {
        auto info = feat->getInfo();
        auto p = feat->readMap<float>();
        auto f = _Input(info->dim, NCHW, halide_type_of<float>());
        memcpy(f->writeMap<float>(), p, info->size * sizeof(float));
        feat = f;
    }

    // Warmup encoder
    m_audio_mod->onForward({feat});
    m_audio_mod->onForward({feat});

    // Run audio encoder
    auto aout = m_audio_mod->onForward({feat});
    if (aout.empty()) {
        LOGE("Audio encoder failed");
        return "";
    }

    // Permute to [T', 1, HIDDEN]
    auto audio_emb = _Permute(aout[0], {1, 0, 2});
    int T = audio_emb->getInfo()->dim[0];
    LOGI("Audio frames: %d", T);

    // ====== Build token sequence ======
    std::vector<int> prefix_tokens = {151644, 8948, 198, 151645, 198,  // system
                                      151644, 872, 198};               // user
    std::vector<int> suffix_tokens = {151670,                          // audio_end
                                      151645, 198,                     // im_end + newline
                                      151644, 77091, 198};             // assistant
    std::vector<int> tokens;
    tokens.insert(tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
    tokens.push_back(AUDIO_START);
    tokens.insert(tokens.end(), T, AUDIO_PAD);
    tokens.insert(tokens.end(), suffix_tokens.begin(), suffix_tokens.end());

    // Build merged embeddings
    auto txt_emb = embedLookup(m_embed_tbl, tokens);
    int S = (int)tokens.size();
    auto merged = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* md = merged->writeMap<float>();
    const float* td = txt_emb->readMap<float>();
    const float* ad = audio_emb->readMap<float>();
    int ai = 0;
    for (int i = 0; i < S; i++) {
        if (tokens[i] == AUDIO_PAD && ai < T) {
            memcpy(md + i * HIDDEN, ad + ai * HIDDEN, HIDDEN * sizeof(float));
            ai++;
        } else {
            memcpy(md + i * HIDDEN, td + i * HIDDEN, HIDDEN * sizeof(float));
        }
    }

    // ====== Prefill ======
    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
    auto pp = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pp[i] = i;

    auto mask = createCausalMask(S, 0);
    auto cache = createEmptyCache();
    m_k_cache = cache[0];
    m_v_cache = cache[1];

    auto out = m_llm_mod->onForward({merged, pos, mask, m_k_cache, m_v_cache});
    if (out.size() < 3) {
        LOGE("Prefill failed");
        return "";
    }

    auto logits = out[0];
    m_k_cache = out[1];
    m_v_cache = out[2];

    // First token
    int current_token = argmaxPenalized(logits->readMap<float>() + (S - 1) * VOCAB, {}, 1.0f);
    m_token_ids.clear();
    m_token_ids.push_back(current_token);
    m_prefill_token_count = S;
    int gen_len = 1;

    // ====== Decode loop ======
    while (gen_len < MAX_NEW_TOKENS && current_token != EOS_TOKEN && current_token != IM_END_TOKEN) {
        // Embedding for current token
        auto tok_emb = embedLookup(m_embed_tbl, {current_token});

        int cache_len = S;
        S = cache_len + 1;

        auto pos_decode = _Input({1, 1}, NCHW, halide_type_of<int32_t>());
        auto ppd = pos_decode->writeMap<int32_t>();
        ppd[0] = cache_len;

        auto mask_decode = createCausalMask(1, cache_len);

        out = m_llm_mod->onForward({tok_emb, pos_decode, mask_decode, m_k_cache, m_v_cache});
        if (out.size() < 3) break;

        logits = out[0];
        m_k_cache = out[1];
        m_v_cache = out[2];

        current_token = argmaxPenalized(logits->readMap<float>(), m_token_ids, REP_PENALTY);
        m_token_ids.push_back(current_token);
        gen_len++;
    }

    LOGI("Generated %d tokens", gen_len);
    m_decoder_ran = true;

#endif
    return getResult();
}
