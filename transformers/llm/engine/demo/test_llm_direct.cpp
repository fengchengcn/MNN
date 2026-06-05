// Minimal test: load llm.mnn with auto-detected inputs
#include <llm/llm.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/Interpreter.hpp>
#include <iostream>

using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    std::string model_dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    std::string model_path = model_dir + "/llm.mnn";
    std::string weight_path = model_dir + "/llm.mnn.weight";

    // Create runtime
    MNN::ScheduleConfig sched;
    MNN::BackendConfig backend;
    sched.backendConfig = &backend;
    backend.precision = MNN::BackendConfig::Precision_Low;
    std::shared_ptr<Executor::RuntimeManager> rt(Executor::RuntimeManager::createRuntimeManager(sched));

    // Set external weight file
    rt->setExternalFile(weight_path);

    // Load module with auto-detected inputs
    Module::Config modConfig;
    modConfig.shapeMutable = true;
    modConfig.rearrange = true;

    std::cout << "Loading module..." << std::endl;
    auto module = Module::load({}, {}, model_path.c_str(), rt, &modConfig);
    if (!module) {
        std::cerr << "FAILED: Module::load returned null" << std::endl;
        return 1;
    }

    auto info = module->getInfo();
    std::cout << "Module loaded OK" << std::endl;
    std::cout << "Inputs: ";
    for (auto& n : info->inputNames) std::cout << n << " ";
    std::cout << std::endl;
    std::cout << "Outputs: ";
    for (auto& n : info->outputNames) std::cout << n << " ";
    std::cout << std::endl;

    // Basic forward test
    int S = 4, H = 1024;
    auto input = _Input({1, S, H}, NCHW, halide_type_of<float>());
    auto mask = _Input({1, 1, S, S}, NCHW, halide_type_of<float>());
    auto pos = _Input({1, S}, NCHW, halide_type_of<int32_t>());

    // Fill inputs with simple data
    auto inPtr = input->writeMap<float>();
    for (int i = 0; i < S * H; i++) inPtr[i] = 0.0f;

    auto mPtr = mask->writeMap<float>();
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            mPtr[i * S + j] = (j <= i) ? 0.0f : -1.0e9f;

    auto pPtr = pos->writeMap<int32_t>();
    for (int i = 0; i < S; i++) pPtr[i] = i;

    std::cout << "Running forward..." << std::endl;
    std::vector<VARP> ins = {input, mask, pos};
    auto outputs = module->onForward(ins);

    if (outputs.empty()) {
        std::cerr << "FAILED: Forward returned empty" << std::endl;
        return 1;
    }

    auto oinfo = outputs[0]->getInfo();
    std::cout << "Forward OK. Output shape: [";
    if (oinfo) {
        for (size_t i = 0; i < oinfo->dim.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << oinfo->dim[i];
        }
    }
    std::cout << "]" << std::endl;

    // Read output
    auto oPtr = outputs[0]->readMap<float>();
    if (oPtr) {
        float max_val = -1e9;
        int max_idx = 0;
        for (int i = 0; i < oinfo->dim[2]; i++) {
            if (oPtr[(S-1)*oinfo->dim[2] + i] > max_val) {
                max_val = oPtr[(S-1)*oinfo->dim[2] + i];
                max_idx = i;
            }
        }
        std::cout << "Top token at last pos: " << max_idx << " (score=" << max_val << ")" << std::endl;
    }

    std::cout << "SUCCESS!" << std::endl;
    return 0;
}
