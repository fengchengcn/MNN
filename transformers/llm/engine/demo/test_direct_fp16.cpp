// Direct test for Qwen3-ASR FP16 model with FusedAttention
// Loads llm.mnn + llm.mnn.weight directly via Module::load with explicit input names
#include <llm/llm.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thread>

using namespace MNN::Express;

static const int HIDDEN = 1024;
static const int VOCAB = 151936;

int main(int argc, char* argv[]) {
    std::string dir = "/root/projects/MNN/mnn-models/Qwen3-ASR-MNN-FP16";
    if (argc > 1) dir = argv[1];

    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 8) num_threads = 8;
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), num_threads);
    ExecutorScope scope(executor);

    // ======= Load LLM decoder =======
    std::cout << "[1/2] Loading LLM decoder (" << dir << "/llm.mnn)..." << std::endl;
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

    std::vector<std::string> inputNames = {
        "inputs_embeds", "attention_mask", "position_ids", "logits_index"
    };
    std::vector<std::string> outputNames = {"logits"};

    auto llm_mod = Module::load(inputNames, outputNames, (dir + "/llm.mnn").c_str(), rt, &mc);
    rt->setExternalFile("");
    if (!llm_mod) {
        std::cerr << "FAIL: Module::load returned null" << std::endl;
        return 1;
    }
    std::cout << "  LLM decoder loaded OK" << std::endl;

    // ======= Create dummy inputs and run =======
    std::cout << "[2/2] Running forward pass..." << std::endl;

    int S = 4;
    int mrope_dims = 3;

    // inputs_embeds: [1, S, HIDDEN] (float)
    auto inputs_embeds = _Input({1, S, HIDDEN}, NCHW, halide_type_of<float>());
    float* ie = inputs_embeds->writeMap<float>();
    for (int i = 0; i < S * HIDDEN; i++) ie[i] = 0.0f;

    // attention_mask: [1, 1, S, S] (float, causal)
    auto attn_mask = _Input({1, 1, S, S}, NCHW, halide_type_of<float>());
    float* am = attn_mask->writeMap<float>();
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            am[i * S + j] = (j <= i) ? 0.0f : -1e9f;
        }
    }

    // position_ids: [3, S] (int32, mRoPE)
    auto pos_ids = _Input({mrope_dims, S}, NCHW, halide_type_of<int32_t>());
    int* pp = pos_ids->writeMap<int32_t>();
    for (int d = 0; d < mrope_dims; d++) {
        for (int i = 0; i < S; i++) {
            pp[d * S + i] = i;
        }
    }

    // logits_index: [1] (int32)
    auto logits_idx = _Input({1}, NCHW, halide_type_of<int32_t>());
    int* li = logits_idx->writeMap<int32_t>();
    li[0] = S - 1;  // get logits at last position

    // Run
    std::vector<VARP> fwd_inputs = {inputs_embeds, attn_mask, pos_ids, logits_idx};
    auto outputs = llm_mod->onForward(fwd_inputs);
    if (outputs.empty()) {
        std::cerr << "FAIL: onForward returned empty" << std::endl;
        return 1;
    }

    auto logits = outputs[0];
    auto info = logits->getInfo();
    if (info) {
        std::cout << "  Output shape: [";
        for (int d = 0; d < info->dim.size(); d++) {
            std::cout << info->dim[d] << (d < info->dim.size()-1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
        std::cout << "  Output type: " << info->type.code << std::endl;
    }

    const float* lp = logits->readMap<float>();
    if (lp) {
        int offset = (S - 1) * VOCAB;
        std::vector<std::pair<float,int>> scores;
        for (int i = 0; i < VOCAB; i++)
            scores.push_back({lp[offset + i], i});
        std::sort(scores.rbegin(), scores.rend());
        std::cout << "  Top-5 tokens: ";
        for (int i = 0; i < 5; i++)
            std::cout << scores[i].second << "(" << scores[i].first << ") ";
        std::cout << std::endl;
    } else {
        std::cerr << "FAIL: logits readMap returned null" << std::endl;
        return 1;
    }

    std::cout << "\nSUCCESS: Model loaded and ran correctly!" << std::endl;
    return 0;
}
