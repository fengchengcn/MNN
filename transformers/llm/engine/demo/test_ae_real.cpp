// Test audio encoder with REAL fbank features loaded from file
// Reads features dumped by test_ae_end_to_end.py, runs audio_encoder.mnn, compares with ONNX RT
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>

using namespace MNN::Express;

int main() {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";

    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);

    // Load audio encoder
    std::cout << "Loading audio encoder..." << std::endl;
    auto mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (!mod) {
        std::cerr << "FAILED to load audio_encoder.mnn" << std::endl;
        return 1;
    }
    std::cout << "  OK" << std::endl;

    // Load real fbank features
    std::cout << "\nLoading real fbank features..." << std::endl;
    std::ifstream fmeta(dir + "/dump_real_meta.txt");
    std::string line;
    int C = 128, T = 200;
    if (std::getline(fmeta, line)) {
        // Parse "C=128 T=501 samples=80000"
        sscanf(line.c_str(), "C=%d T=%d", &C, &T);
    }
    std::cout << "  Features: [1, " << C << ", " << T << "]" << std::endl;

    std::vector<float> feat(C * T);
    std::ifstream fin(dir + "/dump_real_features.bin", std::ios::binary);
    fin.read((char*)feat.data(), C * T * sizeof(float));
    fin.close();
    std::cout << "  Range: [" << feat[0] << ", " << feat[C*T-1] << "]" << std::endl;

    // Run audio encoder
    std::cout << "\nRunning MNN inference..." << std::endl;
    auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
    memcpy(inp->writeMap<float>(), feat.data(), C * T * sizeof(float));

    auto out = mod->onForward({inp});
    if (out.empty()) {
        std::cerr << "FAILED" << std::endl;
        return 1;
    }

    auto info = out[0]->getInfo();
    auto ptr = out[0]->readMap<float>();
    if (!ptr || !info) {
        std::cerr << "No output" << std::endl;
        return 1;
    }

    std::cout << "  Output shape: [";
    for (size_t i = 0; i < info->dim.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << info->dim[i];
    }
    std::cout << "]" << std::endl;

    // Find min/max
    float mn = ptr[0], mx = ptr[0];
    for (int i = 1; i < info->size; i++) {
        if (ptr[i] < mn) mn = ptr[i];
        if (ptr[i] > mx) mx = ptr[i];
    }
    std::cout << "  Output range: [" << mn << ", " << mx << "]" << std::endl;

    // Compare with ONNX Runtime output (loaded from file)
    std::cout << "\nComparing with ONNX Runtime output..." << std::endl;
    int onnx_size = info->size;
    std::vector<float> ort_out(onnx_size);
    std::ifstream fortd(dir + "/dump_real_ort_ae.bin", std::ios::binary);
    fortd.read((char*)ort_out.data(), onnx_size * sizeof(float));
    fortd.close();

    // Metrics
    double dot = 0, na = 0, nb = 0;
    float md = 0, sum_abs = 0;
    for (int i = 0; i < onnx_size; i++) {
        float d = ptr[i] - ort_out[i];
        dot += (double)ptr[i] * (double)ort_out[i];
        na += (double)ptr[i] * (double)ptr[i];
        nb += (double)ort_out[i] * (double)ort_out[i];
        if (std::abs(d) > md) md = std::abs(d);
        sum_abs += std::abs(d);
    }
    double cosim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-10);
    float mean_abs = sum_abs / onnx_size;

    std::cout << "  Cosine similarity: " << cosim << std::endl;
    std::cout << "  Max difference:    " << md << std::endl;
    std::cout << "  Mean abs diff:     " << mean_abs << std::endl;

    if (cosim > 0.9999) std::cout << "  => MATCH ✓" << std::endl;
    else if (cosim > 0.998) std::cout << "  => CLOSE" << std::endl;
    else std::cout << "  => SIGNIFICANT DIFFERENCE ✗" << std::endl;

    // Per-position comparison
    int seq_len = info->dim[1];
    int hidden = info->dim[2];
    std::cout << "\n  Per-position cosine similarity:" << std::endl;
    for (int pos = 0; pos < seq_len; pos++) {
        double pdot = 0, pna = 0, pnb = 0;
        float pmd = 0;
        for (int k = 0; k < hidden; k++) {
            int idx = pos * hidden + k;
            float a = ptr[idx], b = ort_out[idx];
            pdot += (double)a * b;
            pna += (double)a * a;
            pnb += (double)b * b;
            float d = std::abs(a - b);
            if (d > pmd) pmd = d;
        }
        double pcosim = pdot / (std::sqrt(pna) * std::sqrt(pnb) + 1e-10);
        if (pos < 5 || pos >= seq_len - 3 || pcosim < 0.998) {
            std::cout << "    pos " << pos << ": cosim=" << pcosim << " maxdiff=" << pmd << std::endl;
        }
    }

    std::cout << "\nDONE." << std::endl;
    return 0;
}
