// Verify new single-model audio.mnn against old dual-model (conv_frontend + encoder)
// Generates same synthetic input, runs both paths, compares cosim
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace MNN::Express;

static float cosim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

static float max_diff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

int main(int argc, char** argv) {
    std::string model_dir = ".";
    if (argc > 1) model_dir = argv[1];

    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    // === Load models ===
    std::cout << "=== Loading models ===" << std::endl;

    // New single-model
    auto new_mod = Module::load({}, {}, (model_dir + "/audio.mnn").c_str());
    if (!new_mod) {
        std::cerr << "FAILED to load " << model_dir << "/audio.mnn" << std::endl;
        return 1;
    }
    auto new_inputs = new_mod->getInfo()->inputNames;
    std::cout << "  audio.mnn: " << new_inputs.size() << " input(s)";
    for (auto& n : new_inputs) std::cout << " [" << n << "]";
    std::cout << std::endl;

    // Old dual-model
    auto conv_mod = Module::load({}, {}, (model_dir + "/conv_frontend.mnn").c_str());
    if (!conv_mod) {
        std::cerr << "FAILED to load conv_frontend.mnn" << std::endl;
        return 1;
    }
    std::cout << "  conv_frontend.mnn: OK" << std::endl;

    auto enc_mod = Module::load({}, {}, (model_dir + "/encoder.mnn").c_str());
    if (!enc_mod) {
        std::cerr << "FAILED to load encoder.mnn" << std::endl;
        return 1;
    }
    std::cout << "  encoder.mnn: OK" << std::endl;

    // === Run inference with same synthetic input ===
    const int C = 128;
    const int test_lengths[] = {100, 200, 500, 1000};

    std::cout << "\n=== Cosim Comparison ===" << std::endl;
    std::cout << "  T_fbank | T_enc_new | T_enc_old | cosim   | max_diff" << std::endl;
    std::cout << "  --------|-----------|-----------|---------|----------" << std::endl;

    for (int T : test_lengths) {
        // Create input [1, C, T] with reproducible pattern
        auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
        auto* data = inp->writeMap<float>();
        for (int i = 0; i < C * T; i++) {
            data[i] = std::sin((float)i * 0.01f) * 0.5f;
        }

        // --- New path: audio.mnn (single model) ---
        auto new_out = new_mod->onForward({inp});
        if (new_out.empty() || new_out[0].get() == nullptr || new_out[0]->getInfo() == nullptr) {
            std::cerr << "  FAILED new path T=" << T << std::endl;
            continue;
        }
        auto new_info = new_out[0]->getInfo();
        auto new_ptr  = new_out[0]->readMap<float>();
        int new_T = new_info->dim[1];
        int new_D = new_info->dim[2];
        int new_N = new_info->size;

        // --- Old path: conv_frontend → Permute → encoder ---
        // conv_frontend expects [1, T, C] (time-major), so permute
        auto permuted = _Permute(inp, {0, 2, 1});  // [1, T, C]
        auto conv_out = conv_mod->forward(permuted);
        if (conv_out.get() == nullptr || conv_out->getInfo() == nullptr) {
            std::cerr << "  FAILED conv_frontend T=" << T << std::endl;
            continue;
        }
        int enc_T = conv_out->getInfo()->dim[1];
        int enc_D = conv_out->getInfo()->dim[2];

        // Create feature mask [1, enc_T] (all 1s)
        auto mask = _Input({1, enc_T}, NCHW, halide_type_of<float>());
        auto* mask_ptr = mask->writeMap<float>();
        std::fill(mask_ptr, mask_ptr + enc_T, 1.0f);

        auto enc_out = enc_mod->onForward({conv_out, mask});
        if (enc_out.empty() || enc_out[0].get() == nullptr || enc_out[0]->getInfo() == nullptr) {
            std::cerr << "  FAILED encoder T=" << T << std::endl;
            continue;
        }
        auto old_info = enc_out[0]->getInfo();
        auto old_ptr  = enc_out[0]->readMap<float>();
        int old_T = old_info->dim[1];
        int old_D = old_info->dim[2];
        int old_N = old_info->size;

        // Compare
        float c = cosim(new_ptr, old_ptr, std::min(new_N, old_N));
        float d = max_diff(new_ptr, old_ptr, std::min(new_N, old_N));

        printf("  %7d | %9d | %9d | %7.4f | %8.6f\n", T, new_T, old_T, c, d);
    }

    std::cout << "\n=== DONE ===" << std::endl;
    return 0;
}
