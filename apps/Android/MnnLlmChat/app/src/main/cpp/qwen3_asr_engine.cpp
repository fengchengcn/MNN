#include "qwen3_asr_engine.h"
#include <MNN/Interpreter.hpp>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <android/log.h>
#include "tokenizer.hpp"

#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif

#define LOG_TAG "Qwen3AsrEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace MNN::Express;

// ====== Constructor / Destructor ======

Qwen3AsrEngine::Qwen3AsrEngine()
    : m_decoder_loaded(false)
    , m_ae_loaded(false)
    , m_prefill_token_count(0)
    , m_num_threads(4)
    , m_initialized(false)
    , m_decoder_ran(false) {
}

Qwen3AsrEngine::~Qwen3AsrEngine() {
    release();
}

// ====== bf16 conversion ======

float Qwen3AsrEngine::bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

// ====== Embedding via mmap (zero-RAM, on-demand paging) ======

bool Qwen3AsrEngine::openEmbeddingFile(const std::string& path) {
    m_embed_fd = open(path.c_str(), O_RDONLY);
    if (m_embed_fd < 0) {
        LOGE("Cannot open embedding file: %s (errno=%d)", path.c_str(), errno);
        return false;
    }

    struct stat st;
    if (fstat(m_embed_fd, &st) != 0) {
        LOGE("fstat failed on embedding file (errno=%d)", errno);
        close(m_embed_fd);
        m_embed_fd = -1;
        return false;
    }
    m_embed_file_size = st.st_size;

    size_t expected = (size_t)VOCAB * HIDDEN * 2;
    if (m_embed_file_size < expected) {
        LOGW("Embedding file size %zu < expected %zu", m_embed_file_size, expected);
    }

    m_embed_mmap = mmap(nullptr, m_embed_file_size, PROT_READ, MAP_SHARED, m_embed_fd, 0);
    if (m_embed_mmap == MAP_FAILED) {
        LOGE("mmap failed on embedding file (errno=%d)", errno);
        close(m_embed_fd);
        m_embed_fd = -1;
        return false;
    }

    madvise(m_embed_mmap, m_embed_file_size, MADV_RANDOM);
    LOGI("Embedding mmap'd: %zu MB (on-demand paging)", m_embed_file_size / (1024 * 1024));
    return true;
}

void Qwen3AsrEngine::closeEmbeddingFile() {
    if (m_embed_mmap != MAP_FAILED) {
        munmap(m_embed_mmap, m_embed_file_size);
        m_embed_mmap = MAP_FAILED;
    }
    if (m_embed_fd >= 0) {
        close(m_embed_fd);
        m_embed_fd = -1;
    }
    m_embed_file_size = 0;
}

void Qwen3AsrEngine::embedLookup(const std::vector<int>& ids, float* dst) {
    const uint16_t* src = static_cast<const uint16_t*>(m_embed_mmap);
    const size_t row_vals = HIDDEN;

    std::unordered_map<int, int> seen;
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        if (id < 0 || id >= VOCAB) id = 0;

        auto it = seen.find(id);
        if (it != seen.end()) {
            memcpy(dst + i * row_vals, dst + it->second * row_vals, row_vals * sizeof(float));
            continue;
        }
        seen[id] = (int)i;

        const uint16_t* row = src + id * row_vals;
        float* out = dst + i * row_vals;
        for (int j = 0; j < HIDDEN; j++) {
            out[j] = bf16_to_f32(row[j]);
        }
    }
}

// ====== argmax with repetition penalty ======

int Qwen3AsrEngine::argmaxPenalized(const float* logits, float* penalized_buf,
                                     const std::vector<int>& history, float penalty) {
    if (penalty <= 1.0f || history.empty()) {
        int idx = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[idx]) idx = i;
        return idx;
    }
    memcpy(penalized_buf, logits, VOCAB * sizeof(float));
    for (int id : history) {
        if (id < 0 || id >= VOCAB) continue;
        if (penalized_buf[id] < 0)
            penalized_buf[id] *= penalty;
        else
            penalized_buf[id] /= penalty;
    }
    int idx = 0;
    for (int i = 1; i < VOCAB; i++) if (penalized_buf[i] > penalized_buf[idx]) idx = i;
    return idx;
}

// ====== Causal mask & empty KV cache ======

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

// ====== Model loading (on-demand, serialized to avoid dual-model memory peak) ======

std::shared_ptr<Module> Qwen3AsrEngine::loadAudioEncoder() {
    LOGI("Loading audio encoder (on-demand)...");
    auto mod = std::shared_ptr<Module>(Module::load({}, {}, (m_model_dir + "/audio_encoder.mnn").c_str()));
    if (!mod) {
        LOGE("Failed to load audio encoder");
    } else {
        LOGI("Audio encoder loaded");
    }
    return mod;
}

bool Qwen3AsrEngine::ensureAudioEncoderLoaded() {
    if (m_ae_loaded) {
        LOGI("Audio encoder already loaded, reusing");
        return true;
    }

    LOGI("Loading audio encoder (first use, will keep resident)...");
    m_ae_mod = loadAudioEncoder();
    if (!m_ae_mod) {
        LOGE("Failed to load audio encoder");
        return false;
    }

    // Warmup: run twice with a synthetic input to trigger shape inference
    // and memory allocation once, not on every utterance.
    // Audio encoder input shape is [1, 128, T] where T varies per utterance,
    // so we warm up with T=50 (typical for ~3s audio at 16kHz with 160-hop fbank).
    // MNN will auto-recompile on first real input if T differs significantly;
    // this is still far cheaper than reloading the model.
    {
        // Create a minimal fbank-like input for warmup (128 mel bins × 50 frames)
        auto warmup_in = _Input({1, 128, 50}, NCHW, halide_type_of<float>());
        auto p = warmup_in->writeMap<float>();
        memset(p, 0, 1 * 128 * 50 * sizeof(float));
        // Pre-fill with tiny random values to avoid degenerate edge cases
        for (int i = 0; i < 128 * 50; i++) p[i] = (i % 255 - 127) / 12700.0f;
        m_ae_mod->onForward({warmup_in});
        m_ae_mod->onForward({warmup_in});
        LOGI("Audio encoder warmup complete (2× synthetic forward, T=50)");
    }

    m_ae_loaded = true;
    LOGI("Audio encoder resident (will not be released until engine shutdown)");
    return true;
}

bool Qwen3AsrEngine::ensureDecoderLoaded() {
    if (m_decoder_loaded) {
        LOGI("Decoder already loaded, reusing");
        return true;
    }

    LOGI("Loading LLM decoder (first use)...");
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    bc.memory = MNN::BackendConfig::Memory_Low;  // Reduce internal memory pool

    MNN::ScheduleConfig sched;
    sched.backendConfig = &bc;
    m_rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));

    // Memory optimization hints (same pattern as Llm::setRuntimeHint in llm.cpp)
    m_rt->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);       // Defer allocation → lower peak
    m_rt->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);          // mmap weights, don't read all
    m_rt->setHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);    // Minimal winograd memory
    m_rt->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, 1);    // Per-tensor dynamic quant

    m_rt->setExternalFile(m_model_dir + "/llm_kv_8bit.mnn.weight");

    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    m_llm_mod = std::shared_ptr<Module>(Module::load(
        {}, {}, (m_model_dir + "/llm_kv_8bit.mnn").c_str(), m_rt, &mc));

    if (!m_llm_mod) {
        LOGE("Failed to load LLM decoder");
        return false;
    }

    m_decoder_loaded = true;
    LOGI("LLM decoder loaded");
    return true;
}

// ====== Init (lightweight — no model loading) ======

void Qwen3AsrEngine::buildPromptTokens() {
    auto* tok = static_cast<MNN::Transformer::Tokenizer*>(m_tokenizer);
    if (!tok) {
        // Fallback: hardcoded empty-system prompt (Chinese-only, no language guidance)
        m_prefix_tokens = {151644, 8948, 198, 151645, 198,   // <|im_start|>system\n<|im_end|>\n
                           151644, 872, 198};                 // <|im_start|>user\n
        m_suffix_tokens = {151670,                            // <|audio_end|>
                           151645, 198,                       // <|im_end|>\n
                           151644, 77091, 198};               // <|im_start|>assistant\n
        return;
    }

    // Build prefix: <|im_start|>system\n{system_msg}\n<|im_end|>\n<|im_start|>user\n
    m_prefix_tokens = {151644, 8948, 198};  // <|im_start|>system\n

    // Encode system message using the tokenizer
    // A multilingual system prompt improves English/mixed-language recognition
    auto sys_msg = tok->encode("You are a helpful assistant.");
    m_prefix_tokens.insert(m_prefix_tokens.end(), sys_msg.begin(), sys_msg.end());
    m_prefix_tokens.push_back(198);   // \n
    m_prefix_tokens.push_back(151645); // <|im_end|>
    m_prefix_tokens.push_back(198);   // \n
    m_prefix_tokens.push_back(151644); // <|im_start|>
    m_prefix_tokens.push_back(872);    // user
    m_prefix_tokens.push_back(198);   // \n

    // Build suffix: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
    m_suffix_tokens = {151670,          // <|audio_end|>
                       151645, 198,     // <|im_end|>\n
                       151644, 77091, 198}; // <|im_start|>assistant\n

    LOGI("Prompt tokens built: prefix=%zu, suffix=%zu", m_prefix_tokens.size(), m_suffix_tokens.size());
}

bool Qwen3AsrEngine::init(const std::string& model_dir, const std::string& cache_dir, int num_threads) {
    if (m_initialized) release();

    m_model_dir = model_dir;
    m_cache_dir = cache_dir;
    m_num_threads = num_threads;

    // Pre-allocate penalty buffer (607 KB)
    m_penalty_buf.resize(VOCAB);

    // Load tokenizer via MNN's Tokenizer class (handles SentencePiece/Tiktoken binary formats)
    LOGI("Loading tokenizer...");
    {
        auto* tok = MNN::Transformer::Tokenizer::createTokenizer(model_dir + "/tokenizer.txt");
        if (tok) {
            m_tokenizer = tok;
            LOGI("Tokenizer loaded successfully");
        } else {
            LOGW("tokenizer.txt not found or invalid format, text decoding disabled");
        }
    }

    // Build prompt tokens using the tokenizer (encodes system message)
    buildPromptTokens();

    // mmap embedding file (virtually zero RSS)
    LOGI("Memory-mapping embedding file...");
    if (!openEmbeddingFile(model_dir + "/embeddings_bf16.bin")) {
        LOGE("Failed to mmap embedding file");
        return false;
    }

    // Phase 3: Create persistent CPU executor with FP16 + Power_High + 4 threads.
    // Reused across utterances by runDecoder() to avoid ~10ms per-call executor creation.
    // ARM v8.2 NEON FP16 (A76/A77 on Kirin 990/9000) doubles theoretical throughput at
    // Precision_Low. Power_High binds to big cores and raises CPU frequency.
    {
        MNN::BackendConfig bc;
        bc.precision = MNN::BackendConfig::Precision_Low;   // FP16 on ARM v8.2
        bc.power     = MNN::BackendConfig::Power_High;      // Bind big cores, raise freq
        bc.memory    = MNN::BackendConfig::Memory_Low;      // Reduce internal memory pool
        m_executor = Executor::newExecutor(MNN_FORWARD_CPU, bc, m_num_threads);
        LOGI("Phase 3: Persistent executor created (threads=%d, precision=FP16, power=High)", m_num_threads);
    }

    m_initialized = true;
    LOGI("Qwen3-ASR Engine initialized (models loaded on-demand, threads=%d)", m_num_threads);
    return true;
}

// ====== Audio input ======

bool Qwen3AsrEngine::pushAudio(const float* samples, int num_samples) {
    if (!m_initialized) {
        LOGE("Engine not initialized");
        return false;
    }
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

// ====== Result decoding ======

std::string Qwen3AsrEngine::getResult() const {
    if (m_token_ids.empty()) return "";
    std::string result;
    for (size_t i = 0; i < m_token_ids.size(); i++) {
        if (i > 0) result += " ";
        result += std::to_string(m_token_ids[i]);
    }
    return result;
}

std::string Qwen3AsrEngine::getResultText() const {
    if (m_token_ids.empty()) return "";
    auto* tok = static_cast<MNN::Transformer::Tokenizer*>(m_tokenizer);
    if (tok) {
        return tok->decode(m_token_ids);
    }
    // Fallback: return space-separated token IDs
    return getResult();
}

// ====== Phase 2: Streaming incremental decode (non-blocking, call from JNI) ======

bool Qwen3AsrEngine::startDecode() {
    // Guard against concurrent calls from multi-threaded JNI (same pattern as runDecoder)
    std::unique_lock<std::mutex> lock(m_decode_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        LOGW("startDecode skipped: another decode is already in progress");
        return false;
    }

    if (!m_initialized || m_audio_buffer.empty()) {
        LOGW("startDecode: not initialized or no audio");
        return false;
    }
    LOGI("startDecode: %zu audio samples, threads=%d, executor=per-utterance(FP16+Power_High)",
         m_audio_buffer.size(), m_num_threads);

    // Create executor (stored as member for lifecycle across decode steps)
    // Phase 3: FP16 precision + Power_High for streaming path too.
    // Executor is per-utterance (not reused) to avoid memory pool fragmentation (see D3).
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Low;   // FP16 on ARM v8.2
    bc.power     = MNN::BackendConfig::Power_High;      // Bind big cores, raise freq
    bc.memory    = MNN::BackendConfig::Memory_Low;
    m_decode_executor = Executor::newExecutor(MNN_FORWARD_CPU, bc, m_num_threads);
    ExecutorScope scope(m_decode_executor);
    (void)scope;

#ifdef LLM_SUPPORT_AUDIO
    int nsamples = (int)m_audio_buffer.size();

    // ====== Phase 1: Audio Encoder (kept resident across utterances) ======
    auto wf = _Input({nsamples}, NHWC, halide_type_of<float>());
    memcpy(wf->writeMap<float>(), m_audio_buffer.data(), nsamples * sizeof(float));

    auto feat = MNN::AUDIO::whisper_fbank(wf);
    if (feat.get() == nullptr || feat->getInfo() == nullptr) {
        LOGE("startDecode: whisper_fbank failed");
        m_decode_executor.reset();
        return false;
    }

    // Materialize fbank tensor (ensure data is in contiguous memory for AE)
    {
        auto info = feat->getInfo();
        auto p = feat->readMap<float>();
        auto f = _Input(info->dim, NCHW, halide_type_of<float>());
        memcpy(f->writeMap<float>(), p, info->size * sizeof(float));
        feat = f;
    }

    LOGI("=== Phase 1: Audio Encoder (streaming) ===");
    if (!ensureAudioEncoderLoaded()) { m_decode_executor.reset(); return false; }

    auto aout = m_ae_mod->onForward({feat});
    if (aout.empty()) {
        LOGE("startDecode: Audio encoder inference failed");
        m_decode_executor.reset();
        return false;
    }

    auto audio_emb = _Permute(aout[0], {1, 0, 2});
    m_decode_T = audio_emb->getInfo()->dim[0];
    LOGI("startDecode: Audio frames=%d", m_decode_T);

    // ====== Phase 2: Load LLM decoder ======
    LOGI("=== Phase 2: LLM Decoder (streaming) ===");
    if (!ensureDecoderLoaded()) { m_decode_executor.reset(); return false; }

    // ====== Build token sequence ======
    std::vector<int> tokens;
    tokens.insert(tokens.end(), m_prefix_tokens.begin(), m_prefix_tokens.end());
    tokens.push_back(AUDIO_START);
    tokens.insert(tokens.end(), m_decode_T, AUDIO_PAD);
    tokens.insert(tokens.end(), m_suffix_tokens.begin(), m_suffix_tokens.end());

    int S = (int)tokens.size();

    // Build merged embeddings
    auto merged = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* md = merged->writeMap<float>();
    embedLookup(tokens, md);

    // Replace AUDIO_PAD positions with audio encoder output
    const float* ad = audio_emb->readMap<float>();
    int ai = 0;
    for (int i = 0; i < S; i++) {
        if (tokens[i] == AUDIO_PAD && ai < m_decode_T) {
            memcpy(md + i * HIDDEN, ad + ai * HIDDEN, HIDDEN * sizeof(float));
            ai++;
        }
    }

    // ====== Prefill ======
    auto t_prefill_0 = std::chrono::steady_clock::now();

    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
    auto pp = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pp[i] = i;

    auto mask = createCausalMask(S, 0);
    auto cache = createEmptyCache();
    m_k_cache = cache[0];
    m_v_cache = cache[1];

    auto out = m_llm_mod->onForward({merged, pos, mask, m_k_cache, m_v_cache});
    if (out.size() < 3) {
        LOGE("startDecode: Prefill failed");
        m_decode_executor.reset();
        return false;
    }

    auto logits = out[0];
    m_k_cache = out[1];
    m_v_cache = out[2];

    // Extract first token from prefill output (last position logits)
    m_decode_current_token = argmaxPenalized(
        logits->readMap<float>() + (S - 1) * VOCAB,
        m_penalty_buf.data(), {}, 1.0f);
    m_token_ids.clear();
    m_token_ids.push_back(m_decode_current_token);
    m_prefill_token_count = S;
    m_decode_S = S;
    m_decode_gen_len = 1;
    m_decoder_ran = true;

    auto t_prefill_1 = std::chrono::steady_clock::now();
    auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_1 - t_prefill_0).count();
    m_decode_t0 = t_prefill_1;  // start decode loop timer (after prefill, incl. first token)

    // If the first token is already a stop token, no decode steps needed
    if (m_decode_current_token == EOS_TOKEN || m_decode_current_token == IM_END_TOKEN) {
        m_decoding_active = false;
        m_decode_executor.reset();
        LOGI("startDecode: Prefill done — S=%d, stop_token=%d, prefill=%lldms",
             S, m_decode_current_token, (long long)prefill_ms);
        return true;
    }

    m_decoding_active = true;
    LOGI("startDecode: Prefill done — S=%d, first_token=%d, audio_frames=%d, prefill=%lldms, ready for streaming",
         S, m_decode_current_token, m_decode_T, (long long)prefill_ms);
    return true;
#else
    LOGE("LLM_SUPPORT_AUDIO not defined at compile time — startDecode is a no-op!");
    m_decode_executor.reset();
    return false;
#endif
}

bool Qwen3AsrEngine::decodeStep(int* token_out) {
    if (!m_decoding_active) {
        LOGW("decodeStep: decode not active");
        return false;
    }

    auto log_perf = [&]() {
        auto t_now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - m_decode_t0).count();
        int decode_tokens = m_decode_gen_len - 1;  // exclude first token (from prefill)
        double tok_per_s = elapsed_ms > 0 ? (decode_tokens * 1000.0 / elapsed_ms) : 0;
        double ms_per_tok = decode_tokens > 0 ? (elapsed_ms / (double)decode_tokens) : 0;
        LOGI("Perf [streaming]: decode=%lldms, %d tokens (%.1f tok/s, %.1f ms/tok)",
             (long long)elapsed_ms, decode_tokens, tok_per_s, ms_per_tok);
    };

    // Check termination conditions (from previous step)
    if (m_decode_current_token == EOS_TOKEN || m_decode_current_token == IM_END_TOKEN) {
        m_decoding_active = false;
        m_decode_executor.reset();
        LOGI("decodeStep: stop token reached, decode complete");
        log_perf();
        return false;
    }
    if (m_decode_gen_len >= MAX_NEW_TOKENS) {
        m_decoding_active = false;
        m_decode_executor.reset();
        LOGI("decodeStep: max tokens reached (%d), decode complete", MAX_NEW_TOKENS);
        log_perf();
        return false;
    }

    // Re-enter executor context for this step
    if (!m_decode_executor) {
        LOGE("decodeStep: executor is null (should not happen while decoding_active)");
        m_decoding_active = false;
        return false;
    }
    ExecutorScope scope(m_decode_executor);
    (void)scope;

    std::vector<float> single_tok_emb(HIDDEN);
    embedLookup({m_decode_current_token}, single_tok_emb.data());

    int cache_len = m_decode_S;
    m_decode_S = cache_len + 1;

    auto tok_emb = _Input({1, 1, HIDDEN}, NCHW, halide_type_of<float>());
    memcpy(tok_emb->writeMap<float>(), single_tok_emb.data(), HIDDEN * sizeof(float));

    auto pos_decode = _Input({1, 1}, NCHW, halide_type_of<int32_t>());
    auto ppd = pos_decode->writeMap<int32_t>();
    ppd[0] = cache_len;

    auto mask_decode = createCausalMask(1, cache_len);

    auto out = m_llm_mod->onForward({tok_emb, pos_decode, mask_decode, m_k_cache, m_v_cache});
    if (out.size() < 3) {
        LOGE("decodeStep: forward failed");
        m_decoding_active = false;
        m_decode_executor.reset();
        log_perf();
        return false;
    }

    auto logits = out[0];
    m_k_cache = out[1];
    m_v_cache = out[2];

    m_decode_current_token = argmaxPenalized(logits->readMap<float>(), m_penalty_buf.data(),
                                              m_token_ids, REP_PENALTY);
    m_token_ids.push_back(m_decode_current_token);
    m_decode_gen_len++;

    if (token_out) *token_out = m_decode_current_token;

    LOGD("decodeStep: token=%d, gen=%d/%d", m_decode_current_token, m_decode_gen_len, MAX_NEW_TOKENS);

    // Check termination after this step
    if (m_decode_current_token == EOS_TOKEN || m_decode_current_token == IM_END_TOKEN) {
        m_decoding_active = false;
        m_decode_executor.reset();
        LOGI("decodeStep: stop token %d, decode complete (%d tokens total)",
             m_decode_current_token, m_decode_gen_len);
        log_perf();
        return false;
    }
    if (m_decode_gen_len >= MAX_NEW_TOKENS) {
        m_decoding_active = false;
        m_decode_executor.reset();
        LOGI("decodeStep: max tokens reached, decode complete (%d tokens total)", m_decode_gen_len);
        log_perf();
        return false;
    }

    return true;  // More tokens expected
}

bool Qwen3AsrEngine::isDecoding() const {
    return m_decoding_active;
}

std::string Qwen3AsrEngine::getPartialResult() const {
    return getResultText();
}

// ====== Reset (keep decoder loaded, clear transient state) ======

void Qwen3AsrEngine::reset() {
    m_audio_buffer.clear();
    m_token_ids.clear();
    m_prefill_token_count = 0;
    m_decoder_ran = false;
    m_k_cache = VARP();
    m_v_cache = VARP();
    // Phase 2 streaming state cleanup
    m_decoding_active = false;
    m_decode_gen_len = 0;
    m_decode_S = 0;
    m_decode_current_token = 0;
    m_decode_T = 0;
    m_decode_executor.reset();
}

void Qwen3AsrEngine::release() {
    reset();
    m_executor.reset();  // Phase 3: release persistent executor
    m_ae_mod.reset();
    m_ae_loaded = false;
    m_llm_mod.reset();
    m_rt.reset();
    m_decoder_loaded = false;
    if (m_tokenizer) {
        delete static_cast<MNN::Transformer::Tokenizer*>(m_tokenizer);
        m_tokenizer = nullptr;
    }
    m_penalty_buf.clear();
    m_penalty_buf.shrink_to_fit();
    closeEmbeddingFile();
    m_initialized = false;
}

// ====== Main decoder (models loaded on-demand) ======

std::string Qwen3AsrEngine::runDecoder() {
    // Guard against concurrent calls from multi-threaded JNI (e.g. duplicate endpoint detection).
    // If another thread is already decoding, this call returns immediately with an empty result.
    std::unique_lock<std::mutex> lock(m_decode_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        LOGW("runDecoder skipped: another decode is already in progress");
        return "";
    }

    if (!m_initialized) return "";
    LOGI("runDecoder: %zu audio samples, threads=%d, executor=persistent(FP16+Power_High)",
         m_audio_buffer.size(), m_num_threads);

    // Phase 3: Reuse persistent executor (FP16 + Power_High) created in init().
    // Avoids ~10ms per-utterance executor creation + benefits from FP16 acceleration.
    if (!m_executor) {
        LOGE("Persistent executor is null (should have been created in init)");
        return "";
    }
    ExecutorScope scope(m_executor);
    (void)scope;

#ifdef LLM_SUPPORT_AUDIO
    int nsamples = (int)m_audio_buffer.size();

    // ====== Phase 1: Audio Encoder (kept resident across utterances) ======
    // Build waveform VARP directly from float buffer — no WAV file roundtrip.
    // This eliminates: float→int16 quantization, disk I/O, RIFF header parsing,
    // and the SELinux risk of temp files. ~140 lines removed vs v5.
    auto wf = _Input({nsamples}, NHWC, halide_type_of<float>());
    memcpy(wf->writeMap<float>(), m_audio_buffer.data(), nsamples * sizeof(float));

    auto feat = MNN::AUDIO::whisper_fbank(wf);
    if (feat.get() == nullptr || feat->getInfo() == nullptr) {
        LOGE("whisper_fbank failed");
        return "";
    }

    // Materialize fbank tensor (ensure data is in contiguous memory for AE)
    {
        auto info = feat->getInfo();
        auto p = feat->readMap<float>();
        auto f = _Input(info->dim, NCHW, halide_type_of<float>());
        memcpy(f->writeMap<float>(), p, info->size * sizeof(float));
        feat = f;
    }

    LOGI("=== Phase 1: Audio Encoder ===");
    if (!ensureAudioEncoderLoaded()) return "";

    auto aout = m_ae_mod->onForward({feat});
    if (aout.empty()) {
        LOGE("Audio encoder inference failed");
        return "";
    }

    auto audio_emb = _Permute(aout[0], {1, 0, 2});
    int T = audio_emb->getInfo()->dim[0];
    LOGI("Audio frames: %d", T);

    // NOTE: m_ae_mod is kept alive — NOT reset. Next utterance reuses cached AE.

    // ====== Phase 2: Load LLM decoder (only model in memory now) ======
    LOGI("=== Phase 2: LLM Decoder ===");
    if (!ensureDecoderLoaded()) return "";

    // ====== Build token sequence (uses tokenizer-encoded prompt from init) ======
    std::vector<int> tokens;
    tokens.insert(tokens.end(), m_prefix_tokens.begin(), m_prefix_tokens.end());
    tokens.push_back(AUDIO_START);
    tokens.insert(tokens.end(), T, AUDIO_PAD);
    tokens.insert(tokens.end(), m_suffix_tokens.begin(), m_suffix_tokens.end());

    int S = (int)tokens.size();

    // Build merged embeddings
    auto merged = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* md = merged->writeMap<float>();
    embedLookup(tokens, md);

    // Replace AUDIO_PAD with audio encoder output
    const float* ad = audio_emb->readMap<float>();
    int ai = 0;
    for (int i = 0; i < S; i++) {
        if (tokens[i] == AUDIO_PAD && ai < T) {
            memcpy(md + i * HIDDEN, ad + ai * HIDDEN, HIDDEN * sizeof(float));
            ai++;
        }
    }

    // ====== Prefill ======
    auto t_prefill_0 = std::chrono::steady_clock::now();

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

    int current_token = argmaxPenalized(
        logits->readMap<float>() + (S - 1) * VOCAB,
        m_penalty_buf.data(), {}, 1.0f);
    m_token_ids.clear();
    m_token_ids.push_back(current_token);
    m_prefill_token_count = S;
    int gen_len = 1;

    auto t_prefill_1 = std::chrono::steady_clock::now();
    auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_1 - t_prefill_0).count();

    // ====== Decode loop ======
    std::vector<float> single_tok_emb(HIDDEN);
    auto t_decode_0 = std::chrono::steady_clock::now();

    while (gen_len < MAX_NEW_TOKENS && current_token != EOS_TOKEN && current_token != IM_END_TOKEN) {
        embedLookup({current_token}, single_tok_emb.data());

        int cache_len = S;
        S = cache_len + 1;

        auto tok_emb = _Input({1, 1, HIDDEN}, NCHW, halide_type_of<float>());
        memcpy(tok_emb->writeMap<float>(), single_tok_emb.data(), HIDDEN * sizeof(float));

        auto pos_decode = _Input({1, 1}, NCHW, halide_type_of<int32_t>());
        auto ppd = pos_decode->writeMap<int32_t>();
        ppd[0] = cache_len;

        auto mask_decode = createCausalMask(1, cache_len);

        out = m_llm_mod->onForward({tok_emb, pos_decode, mask_decode, m_k_cache, m_v_cache});
        if (out.size() < 3) break;

        logits = out[0];
        m_k_cache = out[1];
        m_v_cache = out[2];

        current_token = argmaxPenalized(logits->readMap<float>(), m_penalty_buf.data(),
                                        m_token_ids, REP_PENALTY);
        m_token_ids.push_back(current_token);
        gen_len++;
    }

    auto t_decode_1 = std::chrono::steady_clock::now();
    auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_1 - t_decode_0).count();
    auto decode_tokens = gen_len - 1;  // exclude first token (from prefill)
    double tok_per_s = decode_ms > 0 ? (decode_tokens * 1000.0 / decode_ms) : 0;
    double ms_per_tok = decode_tokens > 0 ? (decode_ms / (double)decode_tokens) : 0;
    LOGI("Perf [runDecoder]: prefill=%lldms (S=%d) | decode=%lldms, %d tokens (%.1f tok/s, %.1f ms/tok)",
         (long long)prefill_ms, m_prefill_token_count,
         (long long)decode_ms, decode_tokens, tok_per_s, ms_per_tok);
    m_decoder_ran = true;

#else
    LOGE("LLM_SUPPORT_AUDIO not defined at compile time — decoder is a no-op! "
         "Add LLM_SUPPORT_AUDIO to target_compile_definitions in CMakeLists.txt");
#endif
    return getResult();
}
