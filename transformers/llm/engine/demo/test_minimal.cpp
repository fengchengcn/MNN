// Minimal test - just load and do a single forward
#include "llm/llm.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json [prompt]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::cout << "Creating LLM..." << std::endl;
    auto llm = std::unique_ptr<Llm>(Llm::createLLM(config_path));
    if (!llm) { std::cerr << "createLLM failed" << std::endl; return 1; }

    llm->set_config("{\"tmp_path\":\"tmp\"}");

    std::cout << "Loading..." << std::endl;
    if (!llm->load()) { std::cerr << "load failed" << std::endl; return 1; }

    std::cout << "Loaded! Generating response..." << std::endl;

    std::string prompt = (argc >= 3) ? argv[2] : "hello";
    std::cout << "Prompt: " << prompt << std::endl;

    llm->response(prompt, &std::cout);

    std::cout << "\nDone!" << std::endl;
    return 0;
}
