// Quick test: load Qwen3-ASR model and run a forward pass
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
    std::cout << "config path is " << config_path << std::endl;

    // Step 1: Create LLM (should return Omni since is_audio=true)
    auto config = std::make_shared<LlmConfig>(config_path);
    std::cout << "base_dir: " << config->base_dir_ << std::endl;
    std::cout << "llm_model: " << config->llm_model() << std::endl;
    std::cout << "llm_weight: " << config->llm_weight() << std::endl;
    std::cout << "audio_model: " << config->audio_model() << std::endl;
    std::cout << "tokenizer_file: " << config->tokenizer_file() << std::endl;
    std::cout << "has input_names: " << config->config_.contains("input_names") << std::endl;
    if (config->config_.contains("input_names")) {
        auto input_names = config->config_.value("input_names", std::vector<std::string>());
        std::cout << "input_names: [";
        for (const auto& n : input_names) std::cout << n << ", ";
        std::cout << "]" << std::endl;
    }
    std::cout << "has tie_embeddings: " << config->config_.contains("tie_embeddings") << std::endl;

    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    if (!llm) {
        std::cerr << "Failed to create LLM" << std::endl;
        return 1;
    }
    std::cout << "LLM created successfully" << std::endl;

    // Step 2: Load model
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    bool res = llm->load();
    if (!res) {
        std::cerr << "LLM load failed" << std::endl;
        return 1;
    }
    std::cout << "LLM loaded successfully" << std::endl;

    // Step 3: Try a simple text generation
    std::cout << "\nTest: generating with empty prompt" << std::endl;
    llm->response("");

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
