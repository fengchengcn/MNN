// Debug test: step-by-step load of Qwen3-ASR model
#include "llm/llm.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>

using namespace MNN::Transformer;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::cout << "Step 0: config = " << config_path << std::endl;

    // Create config manually and step through
    auto config = std::make_shared<LlmConfig>(config_path);
    std::cout << "Step 1: LlmConfig created" << std::endl;
    std::cout << "  is_audio = " << config->is_audio() << std::endl;
    std::cout << "  is_visual = " << config->is_visual() << std::endl;
    std::cout << "  audio_model = " << config->audio_model() << std::endl;
    std::cout << "  llm_model = " << config->llm_model() << std::endl;
    std::cout << "  llm_weight = " << config->llm_weight() << std::endl;
    std::cout << "  tokenizer_file = " << config->tokenizer_file() << std::endl;
    std::cout << "  embedding_file = " << config->embedding_file() << std::endl;

    // Create Omni directly
    auto llm = std::make_unique<Omni>(config);
    std::cout << "Step 2: Omni created" << std::endl;

    llm->set_config("{\"tmp_path\":\"tmp\"}");

    std::cout << "Step 3: About to load..." << std::endl;
    bool res = llm->load();
    if (!res) {
        std::cerr << "LLM load failed" << std::endl;
        return 1;
    }
    std::cout << "Step 4: Loaded successfully!" << std::endl;

    std::cout << "\nTest passed!" << std::endl;
    return 0;
}
