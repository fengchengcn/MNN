// Test audio encoder: MNN audio_encoder.mnn vs ONNX Runtime
// Generate dummy input, run through MNN, dump output for Python comparison
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

    // Create dummy input: [1, 128, 200] (12.8s of audio features)
    int C = 128, T = 200;
    std::cout << "Creating dummy input: [1," << C << "," << T << "]" << std::endl;
    auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
    float* data = inp->writeMap<float>();
    for (int i = 0; i < C * T; i++) {
        data[i] = (float)(i % 100) / 100.0f - 0.5f;
    }

    // Run
    std::cout << "Running inference..." << std::endl;
    auto out = mod->onForward({inp});
    if (out.empty()) {
        std::cerr << "FAILED inference" << std::endl;
        return 1;
    }

    auto info = out[0]->getInfo();
    auto ptr = out[0]->readMap<float>();
    if (!ptr || !info) {
        std::cerr << "No output data" << std::endl;
        return 1;
    }

    // Print output shape
    std::cout << "  Output shape: [";
    for (size_t i = 0; i < info->dim.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << info->dim[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  Output range: [" << ptr[0] << ", " << ptr[info->size - 1] << "]" << std::endl;

    // Save input and output for Python comparison
    std::cout << "\nSaving dump files..." << std::endl;
    std::ofstream fout(dir + "/dump_ae_output.bin", std::ios::binary);
    fout.write((const char*)ptr, info->size * sizeof(float));
    fout.close();
    std::cout << "  Saved dump_ae_output.bin (" << (info->size * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;

    std::ofstream finp(dir + "/dump_ae_input.bin", std::ios::binary);
    for (int i = 0; i < C * T; i++) data[i] = (float)(i % 100) / 100.0f - 0.5f;
    finp.write((const char*)data, C * T * sizeof(float));
    finp.close();
    std::cout << "  Saved dump_ae_input.bin (" << (C * T * sizeof(float) / 1024) << " KB)" << std::endl;

    std::ofstream fmeta(dir + "/dump_ae_meta.txt");
    fmeta << "C=" << C << " T=" << T << std::endl;
    fmeta << "output_shape=";
    for (size_t i = 0; i < info->dim.size(); i++) {
        if (i > 0) fmeta << ",";
        fmeta << info->dim[i];
    }
    fmeta << std::endl;
    fmeta.close();
    std::cout << "  Saved dump_ae_meta.txt" << std::endl;

    std::cout << "\nDONE." << std::endl;
    return 0;
}
