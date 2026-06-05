// Qwen3-ASR direct inference via MNN Module API with KV Cache
// Build: see CMakeLists.txt - add as target or compile manually
#include <llm/llm.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>

using timepoint = std::chrono::high_resolution_clock::time_point;
#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED_MS(start) std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - (start)).count()
#define ELAPSED_US(start) std::chrono::duration_cast<std::chrono::microseconds>(NOW() - (start)).count()

#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif

using namespace MNN::Express;

static const int HIDDEN = 1024;
static const int VOCAB  = 151936;
static const int EOS_TOKEN  = 151643;
static const int AUDIO_START = 151669;
static const int AUDIO_END   = 151670;
static const int AUDIO_PAD   = 151676;
static const int LAYERS = 28;
static const int KV_HEADS = 8;
static const int HEAD_DIM = 128;

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

// Load BF16 embedding table → VARP [vocab, hidden]
static VARP load_embed(const std::string& path) {
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

// Embedding lookup: token IDs → [1, S, HIDDEN]
static VARP embed_lookup(VARP tbl, const std::vector<int>& ids) {
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

// Greedy argmax
static int argmax(const float* logits) {
    int idx = 0;
    for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[idx]) idx = i;
    return idx;
}

// Create causal mask: [1, 1, S_new, S_total], where S_total = past_len + S_new
// For prefill (past_len=0): standard causal mask
// For decode (past_len>0, S_new=1): single query attending to all positions
static VARP create_causal_mask(int S_new, int past_len) {
    int S_total = past_len + S_new;
    auto mask = _Input({1, 1, S_new, S_total}, NCHW, halide_type_of<float>());
    float* mp = mask->writeMap<float>();
    for (int i = 0; i < S_new; i++) {
        for (int j = 0; j < S_total; j++) {
            // position (past_len + i) attends to all positions <= (past_len + i)
            mp[i * S_total + j] = (j <= past_len + i) ? 0.0f : -1e9f;
        }
    }
    return mask;
}

// Create empty K/V cache: [LAYERS, 1, KV_HEADS, 0, HEAD_DIM]
static std::vector<VARP> create_empty_cache() {
    auto k = _Input({LAYERS, 1, KV_HEADS, 0, HEAD_DIM}, NCHW, halide_type_of<float>());
    auto v = _Input({LAYERS, 1, KV_HEADS, 0, HEAD_DIM}, NCHW, halide_type_of<float>());
    return {k, v};
}

int main(int argc, char* argv[]) {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    std::string wav = "/tmp/test_audio.wav";
    if (argc > 1) dir = argv[1];
    if (argc > 2) wav = argv[2];

    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 8) num_threads = 8;
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), num_threads);
    ExecutorScope scope(executor);

    // ======= 1. Load audio encoder =======
    std::cout << "[1/6] Loading audio encoder..." << std::endl;
    auto audio_mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (nullptr == audio_mod) { std::cerr << "FAIL\n"; return 1; }

    // ======= 2. Load LLM decoder with KV Cache =======
    std::cout << "[2/6] Loading LLM decoder (KV Cache)..." << std::endl;
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(dir + "/llm_kv_8bit.mnn.weight");
    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    auto llm_mod = Module::load({}, {}, (dir + "/llm_kv_8bit.mnn").c_str(), rt, &mc);
    if (!llm_mod) { std::cerr << "FAIL\n"; return 1; }

    // ======= 3. Load embeddings =======
    std::cout << "[3/6] Loading embeddings..." << std::endl;
    auto embed_tbl = load_embed(dir + "/embeddings_bf16.bin");
    if (embed_tbl.get() == nullptr) { std::cerr << "FAIL\n"; return 1; }

    // ======= 4. Process audio =======
#ifdef LLM_SUPPORT_AUDIO
    std::cout << "[4/6] Processing audio: " << wav << std::endl;
    auto lr = MNN::AUDIO::load(wav, 16000);
    auto wf = lr.first;
    if (wf.get() == nullptr) { std::cerr << "FAIL\n"; return 1; }
    int nsamples = wf->getInfo()->size;

    auto feat = MNN::AUDIO::whisper_fbank(wf);
    if (feat.get() == nullptr || feat->getInfo() == nullptr) { std::cerr << "whisper_fbank FAIL\n"; return 1; }
    // Materialize fbank
    { auto info = feat->getInfo(); auto p = feat->readMap<float>();
      auto f = _Input(info->dim, NCHW, halide_type_of<float>());
      memcpy(f->writeMap<float>(), p, info->size * sizeof(float)); feat = f; }

    // Warmup encoder: 2 runs (first run penalized by shape inference + alloc)
    audio_mod->onForward({feat});
    audio_mod->onForward({feat});

    auto t0 = NOW();
    auto aout = audio_mod->onForward({feat});
    auto t_ae = ELAPSED_MS(t0);
    if (aout.empty()) { std::cerr << "audio encoder FAIL\n"; return 1; }

    // Permute to [T', 1, H]
    auto audio_emb = _Permute(aout[0], {1, 0, 2});
    int T = audio_emb->getInfo()->dim[0];
    std::cout << "  Audio frames: " << T << " ("
              << (nsamples / 16000.0) << "s audio, ~" << (nsamples / 160 / 8) << " expected)" << std::endl;

    // Build token sequence from the Qwen3-ASR chat template
    std::vector<int> prefix_tokens = {151644, 8948, 198, 151645, 198,  // system header
                                      151644, 872, 198};               // user header
    std::vector<int> suffix_tokens = {151670,                          // audio_end
                                      151645, 198,                     // <|im_end|>\n
                                      151644, 77091, 198};             // assistant header
    std::vector<int> tokens;
    tokens.insert(tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
    tokens.push_back(AUDIO_START);
    tokens.insert(tokens.end(), T, AUDIO_PAD);
    tokens.insert(tokens.end(), suffix_tokens.begin(), suffix_tokens.end());

    // Build merged embeddings: replace audio_pad → actual audio embeddings
    auto txt_emb = embed_lookup(embed_tbl, tokens);
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
    std::cout << "  Injected " << ai << "/" << T << " audio embeddings (S=" << S << ")" << std::endl;

    // Position IDs for prefill
    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
    auto pp = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pp[i] = i;

    // Causal mask for prefill (no cache yet)
    auto mask = create_causal_mask(S, 0);

    // Empty K/V cache
    auto cache = create_empty_cache();
    auto k_cache = cache[0];
    auto v_cache = cache[1];
    auto t_prep = ELAPSED_US(t0);

#else
    std::cerr << "LLM_SUPPORT_AUDIO not defined\n";
    return 1;
#endif

    // ======= 5. Prefill =======
    std::cout << "[5/6] Prefill with KV cache..." << std::endl;
    auto t1 = NOW();
    // Model inputs: [inputs_embeds, position_ids, attention_mask, k_cache, v_cache]
    auto out = llm_mod->onForward({merged, pos, mask, k_cache, v_cache});
    auto t_prefill = ELAPSED_MS(t1);
    if (out.size() < 3) { std::cerr << "Prefill FAIL (got " << out.size() << " outputs)\n"; return 1; }

    // Outputs: [logits, k_cache_out, v_cache_out]
    auto logits = out[0];
    k_cache = out[1];
    v_cache = out[2];

    int current_token = argmax(logits->readMap<float>() + (S - 1) * VOCAB);

    // Show top-5 logits at last position for debugging
    std::cout << "  [Prefill] top-5 tokens: ";
    const float* lp = logits->readMap<float>();
    std::vector<std::pair<float,int>> scores;
    for (int i = 0; i < VOCAB; i++)
        scores.push_back({lp[(S - 1) * VOCAB + i], i});
    std::sort(scores.rbegin(), scores.rend());
    for (int i = 0; i < 5; i++)
        std::cout << scores[i].second << "(" << scores[i].first << ") ";
    std::cout << "| EOS=" << lp[(S - 1) * VOCAB + 151645] << std::endl;

    // Dump merged embeddings and logits for accuracy comparison
    {
        std::ofstream f_emb("/tmp/dump_merged.bin", std::ios::binary);
        int dump_dims[3] = {1, S, HIDDEN};
        f_emb.write((char*)dump_dims, 12);
        f_emb.write((char*)merged->readMap<float>(), S * HIDDEN * sizeof(float));
        f_emb.close();

        std::ofstream f_log("/tmp/dump_mnn_logits.bin", std::ios::binary);
        int log_dims[3] = {1, 1, VOCAB};
        f_log.write((char*)log_dims, 12);
        f_log.write((char*)(lp + (S - 1) * VOCAB), VOCAB * sizeof(float));
        f_log.close();
        std::cout << "  [Dump] saved merged embeddings + logits for verification" << std::endl;
    }

    if (current_token == EOS_TOKEN || current_token == 151645) {
        std::cout << "  [EOS immediately - audio embeddings may not be effective]" << std::endl;
    }

    // ======= 6. Decode loop with KV Cache =======
    std::cout << "[6/6] Decoding with KV cache..." << std::endl;
    int gen_len = 1, max_new = 100;
    std::vector<int> decode_times_us;
    std::vector<int> token_ids;
    token_ids.push_back(current_token);

    while (gen_len < max_new && current_token != EOS_TOKEN && current_token != 151645) {
        auto td0 = NOW();

        // Embedding for current token
        auto tok_emb = embed_lookup(embed_tbl, {current_token});

        // Current cache length = total sequence length processed so far
        int cache_len = S;   // tokens from prefill
        // In subsequent steps, S = cache_len (since all tokens are now in cache)

        // Position ID for this single token
        auto pos_decode = _Input({1, 1}, NCHW, halide_type_of<int32_t>());
        auto ppd = pos_decode->writeMap<int32_t>();
        ppd[0] = cache_len;  // absolute position of this new token

        // Mask for decode: [1, 1, 1, cache_len+1], no masking (single query attends to all)
        auto mask_decode = create_causal_mask(1, cache_len);

        // Run decode step with cached K/V
        out = llm_mod->onForward({tok_emb, pos_decode, mask_decode, k_cache, v_cache});
        if (out.size() < 3) break;

        logits = out[0];
        k_cache = out[1];
        v_cache = out[2];

        current_token = argmax(logits->readMap<float>());  // [1, 1, V], only 1 position
        token_ids.push_back(current_token);
        gen_len++;
        S = cache_len + 1;  // update for next iteration

        decode_times_us.push_back(ELAPSED_US(td0));

        if (gen_len <= 30) {
            std::cout << current_token << " " << std::flush;
        }
    }
    auto t_total = ELAPSED_MS(t0);
    std::cout << std::endl;

    // ==== RTF Report ====
    double audio_dur = nsamples / 16000.0;
    double total_ms_d = t_total;
    double rtf = total_ms_d / (audio_dur * 1000);

    std::cout << "\n========== RTF REPORT ==========" << std::endl;
    std::cout << "Audio duration:    " << audio_dur << "s" << std::endl;
    std::cout << "Audio encoder:     " << t_ae << " ms" << std::endl;
    std::cout << "  + (prep embeds): " << (t_prep / 1000) << " ms" << std::endl;
    std::cout << "Decoder prefill:   " << t_prefill << " ms" << std::endl;
    std::cout << "  (S=" << (S - gen_len + 1) << " tokens)" << std::endl;
    std::cout << "Decode steps:      " << gen_len << " tokens" << std::endl;

    if (!decode_times_us.empty()) {
        long long sum = 0, min_v = decode_times_us[0], max_v = decode_times_us[0];
        for (auto v : decode_times_us) {
            sum += v;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        double avg_ms = (double)sum / decode_times_us.size() / 1000.0;
        std::cout << "  Avg decode step: " << avg_ms << " ms" << std::endl;
        std::cout << "  Min decode step: " << (min_v / 1000.0) << " ms" << std::endl;
        std::cout << "  Max decode step: " << (max_v / 1000.0) << " ms" << std::endl;
        std::cout << "  Decode throughput: " << (1000.0 / avg_ms) << " tok/s" << std::endl;
    }

    double decode_total = decode_times_us.empty() ? 0.0 :
        (double)decode_times_us.back() / 1000.0;
    std::cout << "Total inference:  " << total_ms_d / 1000.0 << "s" << std::endl;
    std::cout << "RTF:              " << rtf << std::endl;
    std::cout << "================================" << std::endl;

    std::cout << "\nToken sequence: ";
    for (auto t : token_ids) std::cout << t << " ";
    std::cout << std::endl;
    std::cout << "Generated " << gen_len << " tokens" << std::endl;
    if (current_token == EOS_TOKEN || current_token == 151645) std::cout << "EOS reached." << std::endl;

    std::cout << "\nDONE." << std::endl;
    return 0;
}
