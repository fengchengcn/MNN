// Minimal text-only test: does the LLM decoder produce reasonable text continuations?
#include <llm/llm.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <cstring>

using namespace MNN::Express;
static const int HIDDEN = 1024, VOCAB = 151936, MD = 3;

static float f2f_table[65536];
static bool f2f_init = false;
static float f2f(uint16_t v) {
    if (!f2f_init) {
        for (int i = 0; i < 65536; i++) {
            uint32_t s = (uint32_t)(i & 0x8000) << 16;
            uint32_t e = (i >> 10) & 0x1F, m = i & 0x03FF; uint32_t b;
            if (e == 0) { if (m == 0) b = s; else { uint32_t t=m; int sh=0; while(!(t&0x400)){t<<=1;sh++;} b=s|((113-sh)<<23)|((t&0x3FF)<<13); } }
            else if (e == 31) b = s | 0x7F800000 | (m<<13);
            else b = s | ((e+112)<<23) | (m<<13);
            memcpy(&f2f_table[i], &b, 4);
        }
        f2f_init = true;
    }
    return f2f_table[v];
}

int main(int, char**) {
    auto exec = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig{}, 8);
    ExecutorScope scope(exec);
    std::string dir = "/root/projects/MNN/mnn-models/Qwen3-ASR-MNN-FP16";

    // Load LLM decoder
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(MNN::ScheduleConfig{}));
    rt->setExternalFile(dir + "/llm.mnn.weight");
    Module::Config mc; mc.shapeMutable = true; mc.rearrange = true;
    auto mod = Module::load({"inputs_embeds","attention_mask","position_ids","logits_index"},
                            {"logits"}, (dir + "/llm.mnn").c_str(), rt, &mc);
    rt->setExternalFile("");
    if (!mod) { std::cerr << "FAIL\n"; return 1; }

    // Load embeddings
    std::ifstream f(dir + "/llm.mnn.weight", std::ios::binary);
    f.seekg(881328128);
    size_t n = VOCAB * HIDDEN;
    std::vector<uint16_t> buf(n);
    f.read((char*)buf.data(), n*2);
    auto tbl = _Input({VOCAB, HIDDEN}, NCHW, halide_type_of<float>());
    float* d = tbl->writeMap<float>();
    for (size_t i = 0; i < n; i++) d[i] = f2f_table[buf[i]];

    // Simple prompt: "The capital of France is" -> expect " Paris"
    std::vector<int> prompt = {151644, 8948, 198, 151645, 198, 151644, 872, 198, 1820, 842, 264, 14116, 374, 6527, 151645, 198, 151644, 77091, 198};
    int S = prompt.size();
    auto lookup = [&](const std::vector<int>& ids) {
        auto r = _Input({1, (int)ids.size(), HIDDEN}, NCHW, halide_type_of<float>());
        float* dst = r->writeMap<float>();
        const float* src = tbl->readMap<float>();
        for (int i = 0; i < ids.size(); i++) {
            int id = ids[i]; if (id < 0 || id >= VOCAB) id = 0;
            memcpy(dst + i*HIDDEN, src + id*HIDDEN, HIDDEN*4);
        }
        return r;
    };

    auto emb = lookup(prompt);
    auto pos = _Input({MD, S}, NCHW, halide_type_of<int32_t>());
    auto pp = pos->writeMap<int32_t>();
    for (int d = 0; d < MD; d++) for (int i = 0; i < S; i++) pp[d*S+i] = i;
    auto mask = _Input({1,1,S,S}, NCHW, halide_type_of<float>());
    auto mp = mask->writeMap<float>();
    for (int i = 0; i < S; i++) for (int j = 0; j < S; j++)
        mp[i*S+j] = (j <= i) ? 0.0f : -1e9f;
    auto li = _Input({1}, NCHW, halide_type_of<int32_t>());
    li->writeMap<int32_t>()[0] = S-1;

    std::cout << "Prefill...\n";
    auto out = mod->onForward({emb, mask, pos, li});

    // Top-10
    const float* lp = out[0]->readMap<float>();
    std::vector<std::pair<float,int>> scores;
    for (int i = 0; i < VOCAB; i++)
        scores.push_back({lp[(S-1)*VOCAB + i], i});
    std::sort(scores.rbegin(), scores.rend());
    std::cout << "Top-10 continuation tokens: ";
    for (int i = 0; i < 10; i++) std::cout << scores[i].second << "(" << scores[i].first << ") ";
    std::cout << "\n";

    // Decode a few tokens (new module instance each time avoids KV cache pollution)
    auto ids = prompt;
    for (int step = 0; step < 20; step++) {
        int S2 = ids.size();
        auto emb2 = lookup(ids);
        auto pos2 = _Input({MD, S2}, NCHW, halide_type_of<int32_t>());
        auto pp2 = pos2->writeMap<int32_t>();
        for (int d = 0; d < MD; d++) for (int i = 0; i < S2; i++) pp2[d*S2+i] = i;
        auto mask2 = _Input({1,1,S2,S2}, NCHW, halide_type_of<float>());
        auto mp2 = mask2->writeMap<float>();
        for (int i = 0; i < S2; i++) for (int j = 0; j < S2; j++)
            mp2[i*S2+j] = (j <= i) ? 0.0f : -1e9f;
        auto li2 = _Input({1}, NCHW, halide_type_of<int32_t>());
        li2->writeMap<int32_t>()[0] = S2-1;

        auto out2 = mod->onForward({emb2, mask2, pos2, li2});
        int next = 0;
        const float* logits = out2[0]->readMap<float>();
        for (int i = 1; i < VOCAB; i++) if (logits[(S2-1)*VOCAB + i] > logits[(S2-1)*VOCAB + next]) next = i;
        ids.push_back(next);
        std::cout << next << " " << std::flush;
    }
    std::cout << "\n";
    return 0;
}
