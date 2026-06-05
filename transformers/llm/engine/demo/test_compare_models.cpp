// Compare MNN decoder models (optimized vs noopt) and dump outputs for ONNX comparison
// Usage: ./test_compare_models <model_dir>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <llm/llm.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace MNN::Express;

static const int VOCAB = 151936;

static void dump_binary(const std::string& path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)data, n * sizeof(float));
    f.close();
    std::cout << "  Dumped " << (n * sizeof(float) / 1024 / 1024) << " MB to " << path << std::endl;
}

static float cosine_sim(const float* a, const float* b, size_t n) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
}

static float max_diff(const float* a, const float* b, size_t n) {
    float md = 0;
    for (size_t i = 0; i < n; i++) {
        float d = std::abs(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

struct ModelResult {
    float* logits;
    int seq_len;
    ModelResult(float* l, int s) : logits(l), seq_len(s) {}
    ModelResult() : logits(nullptr), seq_len(0) {}
};

static ModelResult run_model(const std::string& model_path, const std::string& weight_path,
                              const float* embeds, int B, int S, int D,
                              const float* mask, const int* pos) {
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(weight_path);
    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    auto mod = Module::load({}, {}, model_path.c_str(), rt, &mc);
    if (!mod) {
        std::cerr << "Failed to load: " << model_path << std::endl;
        return {nullptr, 0};
    }

    auto inp = _Input({B, S, D}, NCHW, halide_type_of<float>());
    ::memcpy(inp->writeMap<float>(), embeds, B * S * D * sizeof(float));

    auto msk = _Input({B, 1, S, S}, NCHW, halide_type_of<float>());
    ::memcpy(msk->writeMap<float>(), mask, B * 1 * S * S * sizeof(float));

    auto pst = _Input({B, S}, NCHW, halide_type_of<int32_t>());
    ::memcpy(pst->writeMap<int32_t>(), pos, B * S * sizeof(int32_t));

    auto out = mod->onForward({inp, msk, pst});
    if (out.empty()) {
        std::cerr << "Forward failed: " << model_path << std::endl;
        return {nullptr, 0};
    }

    auto info = out[0]->getInfo();
    auto ptr = out[0]->readMap<float>();
    if (!ptr || !info) {
        std::cerr << "No output data: " << model_path << std::endl;
        return {nullptr, 0};
    }

    size_t total = info->size;
    float* copy = new float[total];
    ::memcpy(copy, ptr, total * sizeof(float));
    return {copy, info->dim[1]};
}

int main(int argc, char* argv[]) {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    if (argc > 1) dir = argv[1];

    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    // Create synthetic input (simulates prefill with audio embeddings)
    int B = 1, S = 64, D = 1024;
    std::cout << "Creating synthetic input: B=" << B << " S=" << S << " D=" << D << std::endl;

    std::vector<float> embeds(B * S * D);
    std::vector<float> mask(B * 1 * S * S);
    std::vector<int> pos(B * S);
    for (int i = 0; i < B * S * D; i++) embeds[i] = (float)(i % 100) / 100.0f - 0.5f;
    for (int i = 0; i < B * S * S; i++) {
        int row = i / S, col = i % S;
        mask[i] = (col <= row) ? 0.0f : -1e9f;
    }
    for (int i = 0; i < B * S; i++) pos[i] = i;

    // Run OPTIMIZED model
    std::cout << "\n[1/3] Running optimized model (llm.mnn)..." << std::endl;
    auto opt_result = run_model(dir + "/llm.mnn", dir + "/llm.mnn.weight",
                                embeds.data(), B, S, D, mask.data(), pos.data());
    if (!opt_result.logits) return 1;

    // Run NOOPT model
    std::cout << "[2/3] Running no-optimization model (llm_noopt.mnn)..." << std::endl;
    auto noopt_result = run_model(dir + "/llm_noopt.mnn", dir + "/llm_noopt.mnn.weight",
                                  embeds.data(), B, S, D, mask.data(), pos.data());
    if (!noopt_result.logits) return 1;

    // Compare
    std::cout << "[3/3] Comparing..." << std::endl;

    // Compare full output (all positions)
    size_t total = (size_t)B * opt_result.seq_len * VOCAB;
    std::cout << "  opt seq_len=" << opt_result.seq_len
              << " noopt seq_len=" << noopt_result.seq_len << std::endl;

    size_t min_seq = std::min(opt_result.seq_len, noopt_result.seq_len);
    size_t compare_size = (size_t)B * min_seq * VOCAB;

    float cosim_all = cosine_sim(opt_result.logits, noopt_result.logits, compare_size);
    float maxdiff_all = max_diff(opt_result.logits, noopt_result.logits, compare_size);

    // Compare last position only (where argmax happens)
    int last_offset = (min_seq - 1) * VOCAB;
    float cosim_last = cosine_sim(opt_result.logits + last_offset,
                                  noopt_result.logits + last_offset, VOCAB);
    float maxdiff_last = max_diff(opt_result.logits + last_offset,
                                  noopt_result.logits + last_offset, VOCAB);

    // Compare first position only
    float cosim_first = cosine_sim(opt_result.logits, noopt_result.logits, VOCAB);
    float maxdiff_first = max_diff(opt_result.logits, noopt_result.logits, VOCAB);

    std::cout << "\n========== OPTIMIZED vs NOOPT COMPARISON ==========" << std::endl;
    std::cout << "  Full output:       cosim=" << cosim_all << "  maxdiff=" << maxdiff_all << std::endl;
    std::cout << "  First position:    cosim=" << cosim_first << "  maxdiff=" << maxdiff_first << std::endl;
    std::cout << "  Last position:     cosim=" << cosim_last << "  maxdiff=" << maxdiff_last << std::endl;

    // Show top-5 tokens from each at last position
    auto show_top5 = [](const float* logits, const char* label) {
        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < VOCAB; i++)
            scores.push_back({logits[i], i});
        std::sort(scores.rbegin(), scores.rend());
        std::cout << "  " << label << ": ";
        for (int i = 0; i < 5; i++)
            std::cout << scores[i].second << "(" << scores[i].first << ") ";
        std::cout << "| EOS=" << logits[151645];
        std::cout << std::endl;
    };

    std::cout << "\n  --- Last position top-5 ---" << std::endl;
    show_top5(opt_result.logits + last_offset, "opt  ");
    show_top5(noopt_result.logits + last_offset, "noopt");

    // Check if argmax matches
    auto argmax = [](const float* logits) {
        int idx = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[idx]) idx = i;
        return idx;
    };

    int opt_first = argmax(opt_result.logits + last_offset);
    int noopt_first = argmax(noopt_result.logits + last_offset);
    std::cout << "\n  First token: opt=" << opt_first << " noopt=" << noopt_first;
    if (opt_first == noopt_first) std::cout << "  ✅ MATCH";
    else std::cout << "  ❌ DIFFERENT";
    std::cout << std::endl;

    // Dump outputs for ONNX Runtime comparison
    std::cout << "\n  Dumping outputs for ONNX comparison..." << std::endl;
    dump_binary(dir + "/dump_opt_logits.bin", opt_result.logits, compare_size);
    dump_binary(dir + "/dump_noopt_logits.bin", noopt_result.logits, compare_size);
    dump_binary(dir + "/dump_input_embeds.bin", embeds.data(), B * S * D);
    dump_binary(dir + "/dump_mask.bin", mask.data(), B * 1 * S * S);
    std::ofstream fpos(dir + "/dump_positions.bin", std::ios::binary);
    fpos.write((const char*)pos.data(), B * S * sizeof(int));
    fpos.close();

    // Save metadata
    std::ofstream meta(dir + "/dump_meta.txt");
    meta << "B=" << B << " S=" << S << " D=" << D << " V=" << VOCAB << std::endl;
    meta << "opt_seq_len=" << opt_result.seq_len << std::endl;
    meta << "noopt_seq_len=" << noopt_result.seq_len << std::endl;
    meta << "opt_first_token=" << opt_first << std::endl;
    meta << "noopt_first_token=" << noopt_first << std::endl;
    meta.close();

    std::cout << "\n  Metadata saved to dump_meta.txt" << std::endl;

    // Cleanup
    delete[] opt_result.logits;
    delete[] noopt_result.logits;

    std::cout << "\nDONE." << std::endl;
    return 0;
}
