// Verify new audio.mnn vs old dual-model with constant input (eliminates time-alignment)
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
    for (int i = 0; i < n; i++) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static std::vector<float> run_new_model(VARP inp) {
    static auto mod = Module::load({}, {}, "/Users/bxy/Documents/MNN/mnn-models/Qwen3-ASR-0.6B-Omni-INT8/audio.mnn");
    auto out = mod->onForward({inp});
    auto info = out[0]->getInfo();
    auto ptr = out[0]->readMap<float>();
    return std::vector<float>(ptr, ptr + info->size);
}

static std::vector<float> run_old_model(VARP inp_mel_major) {
    static auto conv = Module::load({}, {}, "/Users/bxy/Documents/MNN/mnn-models/Qwen3-ASR-0.6B-Omni-INT8/conv_frontend.mnn");
    static auto enc  = Module::load({}, {}, "/Users/bxy/Documents/MNN/mnn-models/Qwen3-ASR-0.6B-Omni-INT8/encoder.mnn");

    // permute mel-major → time-major for conv_frontend
    auto permuted = _Permute(inp_mel_major, {0, 2, 1});
    auto conv_out = conv->forward(permuted);
    int enc_T = conv_out->getInfo()->dim[1];

    auto mask = _Input({1, enc_T}, NCHW, halide_type_of<float>());
    auto* mp = mask->writeMap<float>();
    std::fill(mp, mp + enc_T, 1.0f);

    auto enc_out = enc->onForward({conv_out, mask});
    auto info = enc_out[0]->getInfo();
    auto ptr = enc_out[0]->readMap<float>();
    return std::vector<float>(ptr, ptr + info->size);
}

int main() {
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    const int C = 128;
    const int test_Ts[] = {100, 200, 500};

    std::cout << "=== Test 1: Constant input (all 0.0) ===" << std::endl;
    {
        int T = 200;
        auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
        std::fill(inp->writeMap<float>(), inp->writeMap<float>() + C * T, 0.0f);

        auto new_out = run_new_model(inp);
        auto old_out = run_old_model(inp);

        int cmp_n = std::min((int)new_out.size(), (int)old_out.size());
        float c = cosim(new_out.data(), old_out.data(), cmp_n);
        float d = max_diff(new_out.data(), old_out.data(), cmp_n);

        std::cout << "  new size=" << new_out.size() << " old size=" << old_out.size() << std::endl;
        std::cout << "  cosim=" << c << " max_diff=" << d << std::endl;
        std::cout << "  new first 5: ";
        for (int i = 0; i < 5; i++) std::cout << new_out[i] << " ";
        std::cout << std::endl;
        std::cout << "  old first 5: ";
        for (int i = 0; i < 5; i++) std::cout << old_out[i] << " ";
        std::cout << std::endl;
    }

    std::cout << "\n=== Test 2: Constant input (all 0.5) ===" << std::endl;
    {
        int T = 200;
        auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
        std::fill(inp->writeMap<float>(), inp->writeMap<float>() + C * T, 0.5f);

        auto new_out = run_new_model(inp);
        auto old_out = run_old_model(inp);

        int cmp_n = std::min((int)new_out.size(), (int)old_out.size());
        float c = cosim(new_out.data(), old_out.data(), cmp_n);
        float d = max_diff(new_out.data(), old_out.data(), cmp_n);

        std::cout << "  cosim=" << c << " max_diff=" << d << std::endl;
        std::cout << "  new first 5: ";
        for (int i = 0; i < 5; i++) std::cout << new_out[i] << " ";
        std::cout << std::endl;
        std::cout << "  old first 5: ";
        for (int i = 0; i < 5; i++) std::cout << old_out[i] << " ";
        std::cout << std::endl;
    }

    std::cout << "\n=== Test 3: Sine input, per-T comparison ===" << std::endl;
    for (int T : test_Ts) {
        auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
        auto* d = inp->writeMap<float>();
        for (int i = 0; i < C * T; i++) d[i] = std::sin((float)i * 0.01f) * 0.5f;

        auto new_out = run_new_model(inp);
        auto old_out = run_old_model(inp);

        // old outputs [1, T_old, 1024] → flat: T_old * 1024
        // new outputs [1, T_new, 1024] → flat: T_new * 1024
        int new_T = (int)new_out.size() / 1024;
        int old_T = (int)old_out.size() / 1024;

        // Compare each frame (time-aligned by position ratio)
        float frame_cosim_sum = 0;
        int matched_frames = 0;
        for (int f = 0; f < std::min(new_T, old_T); f++) {
            int old_f = f; // Compare same-index frames
            float c = cosim(&new_out[f * 1024], &old_out[old_f * 1024], 1024);
            frame_cosim_sum += c;
            matched_frames++;
        }
        float avg_frame_cosim = frame_cosim_sum / matched_frames;

        // Also compare first frame only (time-aligned)
        float first_frame_cosim = cosim(new_out.data(), old_out.data(), 1024);

        printf("  T=%d: new_T=%d old_T=%d avg_frame_cosim=%.4f first_frame_cosim=%.4f\n",
               T, new_T, old_T, avg_frame_cosim, first_frame_cosim);
    }

    std::cout << "\nDONE" << std::endl;
    return 0;
}
