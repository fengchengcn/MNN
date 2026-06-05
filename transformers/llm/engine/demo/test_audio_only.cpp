// Test audio encoder directly
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>

using namespace MNN::Express;

int main() {
    auto rt = Executor::RuntimeManager::createRuntimeManager({});
    
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    
    std::cout << "Loading audio encoder..." << std::endl;
    auto module = Module::load({}, {}, "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN/audio_encoder.mnn", rt, &config);
    if (!module) {
        std::cerr << "Failed to load audio encoder" << std::endl;
        return 1;
    }
    std::cout << "Audio encoder loaded OK" << std::endl;
    
    // Test forward
    auto input = _Input({1, 128, 200}, NCHW, halide_type_of<float>());
    auto ptr = input->writeMap<float>();
    for (int i = 0; i < 128 * 200; i++) ptr[i] = 0.0f;
    
    std::cout << "Running forward..." << std::endl;
    auto outputs = module->onForward({input});
    if (outputs.empty()) {
        std::cerr << "Forward failed" << std::endl;
        return 1;
    }
    auto info = outputs[0]->getInfo();
    std::cout << "Output shape: [";
    for (size_t i = 0; i < info->dim.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << info->dim[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "Audio encoder works!" << std::endl;
    return 0;
}
