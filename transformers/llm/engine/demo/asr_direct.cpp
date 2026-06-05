// Qwen3-ASR direct inference via MNN Module API (bypasses Llm engine)
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

int main(int argc, char* argv[]) {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    std::string wav = "/tmp/test_audio.wav";
    if (argc > 1) dir = argv[1];
    if (argc > 2) wav = argv[2];

    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    // ======= 1. Load audio encoder =======
    std::cout << "[1/5] Loading audio encoder..." << std::endl;
    auto audio_mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (nullptr == audio_mod) { std::cerr << "FAIL\n"; return 1; }

    // ======= 2. Load LLM decoder =======
    std::cout << "[2/5] Loading LLM decoder..." << std::endl;
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(dir + "/llm.mnn.weight");
    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    auto llm_mod = Module::load({}, {}, (dir + "/llm.mnn").c_str(), rt, &mc);
    if (!llm_mod) { std::cerr << "FAIL\n"; return 1; }

    // ======= 3. Load embeddings =======
    std::cout << "[3/5] Loading embeddings..." << std::endl;
    auto embed_tbl = load_embed(dir + "/embeddings_bf16.bin");
    if (embed_tbl.get() == nullptr) { std::cerr << "FAIL\n"; return 1; }

    // ======= 4. Process audio =======
#ifdef LLM_SUPPORT_AUDIO
    std::cout << "[4/5] Processing audio: " << wav << std::endl;
    auto lr = MNN::AUDIO::load(wav, 16000);
    auto wf = lr.first;
    if (wf.get() == nullptr) { std::cerr << "FAIL\n"; return 1; }
    int nsamples = wf->getInfo()->size;

    auto feat = MNN::AUDIO::whisper_fbank(wf);
    if (feat.get() == nullptr || feat->getInfo() == nullptr) { std::cerr << "whisper_fbank FAIL\n"; return 1; }
    // Materialize
    { auto info = feat->getInfo(); auto p = feat->readMap<float>();
      auto f = _Input(info->dim, NCHW, halide_type_of<float>());
      memcpy(f->writeMap<float>(), p, info->size * sizeof(float)); feat = f; }

    auto aout = audio_mod->onForward({feat});
    if (aout.empty()) { std::cerr << "audio encoder FAIL\n"; return 1; }

    // Permute to [T', 1, H]
    auto audio_emb = _Permute(aout[0], {1, 0, 2});
    int T = audio_emb->getInfo()->dim[0];
    std::cout << "  Audio frames: " << T << " ("
              << (nsamples / 16000.0) << "s audio, ~" << (nsamples / 160 / 8) << " expected)" << std::endl;

    // Build token sequence from the Qwen3-ASR chat template:
    //   <|im_start|>system\n<|im_end|>\n
    //   <|im_start|>user\n<|audio_start|><|audio_pad|>*T<|audio_end|><|im_end|>\n
    //   <|im_start|>assistant\n
    // The single <|audio_pad|> gets expanded to T audio frames during embedding injection
    std::vector<int> prefix_tokens = {151644, 8948, 198, 151645, 198,  // system header
                                      151644, 872, 198};               // user header
    // audio_start inserted here
    // audio_pad * T inserted here (replaced with actual embeddings)
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
    auto merged = _Input({1, (int)tokens.size(), HIDDEN}, NCHW, halide_type_of<float>());
    float* md = merged->writeMap<float>();
    const float* td = txt_emb->readMap<float>();
    const float* ad = audio_emb->readMap<float>();
    int ai = 0;
    for (int i = 0; i < (int)tokens.size(); i++) {
        if (tokens[i] == AUDIO_PAD && ai < T) {
            memcpy(md + i * HIDDEN, ad + ai * HIDDEN, HIDDEN * sizeof(float));
            ai++;
        } else {
            memcpy(md + i * HIDDEN, td + i * HIDDEN, HIDDEN * sizeof(float));
        }
    }
    std::cout << "  Injected " << ai << "/" << T << " audio embeddings" << std::endl;

    // Attention mask (causal)
    int S = (int)tokens.size();
    auto mask = _Input({1, 1, S, S}, NCHW, halide_type_of<float>());
    float* mp = mask->writeMap<float>();
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            mp[i * S + j] = (j <= i) ? 0.0f : -1e9f;

    // Position IDs
    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
    auto pp = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pp[i] = i;

#else
    std::cerr << "LLM_SUPPORT_AUDIO not defined\n";
    return 1;
#endif

    // ======= 5. Generate =======
    std::cout << "[5/5] Generating..." << std::endl << "  ";
    int gen_len = 0, max_new = 100;
    int current_token = -1;

    // Prefill: run full sequence
    auto out = llm_mod->onForward({merged, mask, pos});
    if (out.empty()) { std::cerr << "Prefill failed\n"; return 1; }
    current_token = argmax(out[0]->readMap<float>() + (S - 1) * VOCAB);

    // Show top-5 logits at last position for debugging
    std::cout << "\n  [Prefill] top-5 tokens: ";
    const float* lp = out[0]->readMap<float>();
    int last_pos = S - 1;
    std::vector<std::pair<float,int>> scores;
    for (int i = 0; i < VOCAB; i++)
        scores.push_back({lp[last_pos * VOCAB + i], i});
    std::sort(scores.rbegin(), scores.rend());
    for (int i = 0; i < 5; i++)
        std::cout << scores[i].second << "(" << scores[i].first << ") ";
    std::cout << "| EOS=" << lp[last_pos * VOCAB + 151645];

    if (current_token == EOS_TOKEN || current_token == 151645) {
        std::cout << "\n  [EOS immediately - audio embeddings may not be effective]" << std::endl;
    }

    gen_len++;
    while (gen_len < max_new && current_token != EOS_TOKEN && current_token != 151645) {
        // Decode: single token
        auto tok_emb = embed_lookup(embed_tbl, {current_token});
        int prev_len = S;
        S = prev_len + 1;
        auto new_merged = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
        float* nmd = new_merged->writeMap<float>();
        memcpy(nmd, merged->readMap<float>(), prev_len * HIDDEN * sizeof(float));
        memcpy(nmd + prev_len * HIDDEN, tok_emb->readMap<float>(), HIDDEN * sizeof(float));
        merged = new_merged;

        mask = _Input({1, 1, S, S}, NCHW, halide_type_of<float>());
        mp = mask->writeMap<float>();
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
                mp[i * S + j] = (j <= i) ? 0.0f : -1e9f;

        pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
        pp = pos->writeMap<int32_t>();
        for (int i = 0; i < S; i++) pp[i] = i;

        out = llm_mod->onForward({merged, mask, pos});
        if (out.empty()) break;
        current_token = argmax(out[0]->readMap<float>() + (S - 1) * VOCAB);
        gen_len++;

        if (gen_len <= 20)
            std::cout << current_token << " " << std::flush;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Generated " << gen_len << " tokens" << std::endl;
    if (current_token == EOS_TOKEN || current_token == 151645) std::cout << "EOS reached." << std::endl;

    std::cout << "\nDONE." << std::endl;
    return 0;
}
