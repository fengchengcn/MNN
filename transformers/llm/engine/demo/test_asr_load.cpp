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
