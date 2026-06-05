#include <llm/llm.hpp>
#include <iostream>
using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    auto llm = Llm::createLLM(argv[1]);
    std::cout << "createLLM OK" << std::endl;
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    std::cout << "set_config OK" << std::endl;
    std::cout << "Test PASSED" << std::endl;
    return 0;
}
