// Step-by-step load test
#include <llm/llm.hpp>
#include <llmconfig.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>

using namespace MNN::Transformer;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " config.json" << std::endl; return 1; }

    std::string config_path = argv[1];
    auto config = std::make_shared<LlmConfig>(config_path);

    std::cout << "1. Config created" << std::endl;
    std::cout << "   llm_model: " << config->llm_model() << std::endl;
    std::cout << "   llm_weight: " << config->llm_weight() << std::endl;
    std::cout << "   tokenizer: " << config->tokenizer_file() << std::endl;
    std::cout << "   embedding: " << config->embedding_file() << std::endl;

    // Manually load the MNN module
    ScheduleConfig sched_config;
    BackendConfig cpuBackendConfig;
    sched_config.type = MNN_FORWARD_CPU;
    sched_config.numThread = 4;
    sched_config.backendConfig = &cpuBackendConfig;

    std::cout << "2. Creating runtime manager..." << std::endl;
    auto rtMgr = Executor::RuntimeManager::createRuntimeManager(sched_config);
    if (!rtMgr) { std::cerr << "Failed" << std::endl; return 1; }
    std::cout << "3. Runtime created" << std::endl;

    std::string model_path = config->llm_model();
    std::string weight_path = config->llm_weight();
    std::cout << "4. Loading model: " << model_path << std::endl;
    std::cout << "   weight: " << weight_path << std::endl;

    rtMgr->setExternalFile(weight_path);

    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;

    std::cout << "5. Module::load..." << std::endl;
    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};

    auto module = Module::load(inputNames, outputNames, model_path.c_str(), rtMgr, &module_config);
    if (!module) {
        std::cerr << "Module::load returned nullptr!" << std::endl;
        return 1;
    }
    std::cout << "6. Module loaded successfully!" << std::endl;

    // Try a forward pass
    std::cout << "7. Creating test input..." << std::endl;
    auto input = _Input({1, 4, 1024}, NCHV, halide_type_of<float>());
    auto mask = _Input({1, 1, 4, 4}, NCHW, halide_type_of<float>());
    auto pos = _Input({1, 4}, NCHW, halide_type_of<int>());

    auto ptr = input->writeMap<float>();
    for (int i = 0; i < 4 * 1024; i++) ptr[i] = 0.0f;
    auto mask_ptr = mask->writeMap<float>();
    for (int i = 0; i < 4 * 4; i++) mask_ptr[i] = (i % 5 == 0) ? 0.0f : -1e9f;
    auto pos_ptr = pos->writeMap<int>();
    for (int i = 0; i < 4; i++) pos_ptr[i] = i;

    auto logitsIndex = _Input({}, NCHW, halide_type_of<int>());
    *(logitsIndex->writeMap<int>()) = -1;

    std::cout << "8. Forward pass..." << std::endl;
    auto outputs = module->onForward({input, mask, pos, logitsIndex});
    if (outputs.empty()) {
        std::cerr << "Forward returned empty!" << std::endl;
        return 1;
    }
    std::cout << "9. Forward succeeded! Output shape: ";
    auto info = outputs[0]->getInfo();
    if (info) {
        std::cout << "[";
        for (size_t i = 0; i < info->dim.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << info->dim[i];
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "Test PASSED!" << std::endl;
    return 0;
}
