#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
using namespace MNN::Express;

static const int VOCAB = 151936;
static const int HIDDEN = 1024;
static const int AUDIO_PAD = 151676;

void dump_top5(const float* logits, int n) {
    std::vector<std::pair<float,int>> scores;
    for (int i = 0; i < n; i++) scores.push_back({logits[i], i});
    std::sort(scores.rbegin(), scores.rend());
    for (int i = 0; i < 5; i++)
        std::cout << "    top-" << (i+1) << ": " << scores[i].second
                  << "(" << scores[i].first << ")" << std::endl;
}

int main() {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    // Load MNN decoder
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
    if (!llm_mod) { std::cerr << "FAIL load decoder\n"; return 1; }

    // Load HF AE output
    std::ifstream fae(dir + "/hf_ae_out.bin", std::ios::binary);
    fae.seekg(0, std::ios::end);
    size_t ae_size = fae.tellg() / sizeof(float);
    fae.seekg(0);
    std::vector<float> ae_data(ae_size);
    fae.read((char*)ae_data.data(), ae_size * sizeof(float));
    int T = (int)(ae_size / 1024);

    // Load embedding table
    std::ifstream femb(dir + "/embeddings_bf16.bin", std::ios::binary);
    std::vector<uint16_t> emb_buf((size_t)VOCAB * HIDDEN);
    femb.read((char*)emb_buf.data(), (size_t)VOCAB * HIDDEN * 2);
    std::vector<float> emb_tbl((size_t)VOCAB * HIDDEN);
    for (size_t i = 0; i < (size_t)VOCAB * HIDDEN; i++) {
        uint32_t bits = (uint32_t)emb_buf[i] << 16;
        memcpy(&emb_tbl[i], &bits, 4);
    }

    // Build token sequence (correct prompt format)
    std::vector<int> prompt = {151644, 8948, 198, 151645, 198, 151644, 872, 198};
    std::vector<int> suffix = {151670, 151645, 198, 151644, 77091, 198};
    std::vector<int> tokens;
    tokens.insert(tokens.end(), prompt.begin(), prompt.end());
    tokens.push_back(151669); // audio_start
    tokens.insert(tokens.end(), T, AUDIO_PAD);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    int S = (int)tokens.size();
    std::cout << "Seq len: " << S << " (T=" << T << ")" << std::endl;

    // Build merged embeddings
    auto merged = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* md = merged->writeMap<float>();
    int ai = 0;
    for (int i = 0; i < S; i++) {
        if (tokens[i] == AUDIO_PAD && ai < T) {
            memcpy(md + i * HIDDEN, ae_data.data() + ai * HIDDEN, HIDDEN * sizeof(float));
            ai++;
        } else {
            int tok_id = std::min(std::max(tokens[i], 0), VOCAB-1);
            memcpy(md + i * HIDDEN, emb_tbl.data() + tok_id * HIDDEN, HIDDEN * sizeof(float));
        }
    }

    // Mask
    auto mask = _Input({1, 1, S, S}, NCHW, halide_type_of<float>());
    float* mp = mask->writeMap<float>();
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            mp[i * S + j] = (j <= i) ? 0.0f : -1e9f;

    // Position IDs
    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());
    int* pp = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pp[i] = i;

    // Forward
    auto out = llm_mod->onForward({merged, mask, pos});
    if (out.empty()) { std::cerr << "FAIL forward\n"; return 1; }

    const float* lp = out[0]->readMap<float>();
    int last_pos = S - 1;
    std::cout << "\nMNN prefill result:" << std::endl;
    std::cout << "  First token: " << (std::max_element(lp + last_pos * VOCAB, lp + (last_pos+1) * VOCAB) - (lp + last_pos * VOCAB)) << std::endl;
    std::cout << "  Top-5 at last position:" << std::endl;
    dump_top5(lp + last_pos * VOCAB, VOCAB);
    std::cout << "  EOS score: " << lp[last_pos * VOCAB + 151645] << std::endl;

    std::cout << "\nDONE." << std::endl;
    return 0;
}
