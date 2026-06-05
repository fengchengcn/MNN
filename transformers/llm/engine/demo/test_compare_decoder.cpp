// Compare MNN decoder output with ONNX Runtime reference
#include <llm/llm.hpp>
#include <MNN/expr/Executor.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace MNN::Express;

// Simple npy reader (float32 only, row-major)
static bool read_npy(const std::string& path, std::vector<float>& data, std::vector<int>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Read header:  '\\x93NUMPY' magic + version + header length
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != (char)0x93 || magic[1] != 'N') return false;

    char ver[2];
    f.read(ver, 2);

    uint16_t header_len;
    f.read((char*)&header_len, 2);

    std::string header((size_t)header_len, '\0');
    f.read(&header[0], header_len);

    // Parse shape from header
    auto pos = header.find("shape");
    if (pos != std::string::npos) {
        pos = header.find("(", pos);
        auto end = header.find(")", pos);
        std::string shape_str = header.substr(pos+1, end-pos-1);
        // Parse comma-separated values
        pos = 0;
        while ((pos = shape_str.find_first_not_of(" \n", pos)) != std::string::npos) {
            if (!std::isdigit(shape_str[pos]) && shape_str[pos] != '-') break;
            char* endp;
            int dim = strtol(&shape_str[pos], &endp, 10);
            shape.push_back(dim);
            pos = endp - &shape_str[0] + 1;
        }
    }

    // Read data
    size_t total = 1;
    for (int d : shape) total *= d;
    data.resize(total);
    f.read((char*)data.data(), total * sizeof(float));
    f.close();
    return true;
}

int main() {
    // Load reference data
    std::vector<float> emb_data, mask_data, pos_data, ref_logits;
    std::vector<int> emb_shape, mask_shape, pos_shape, ref_shape;

    if (!read_npy("/tmp/mnn_input_embeds.npy", emb_data, emb_shape)) {
        std::cerr << "Failed to load embeds\n"; return 1;
    }
    if (!read_npy("/tmp/mnn_input_mask.npy", mask_data, mask_shape)) {
        std::cerr << "Failed to load mask\n"; return 1;
    }
    if (!read_npy("/tmp/mnn_input_pos.npy", pos_data, pos_shape)) {
        std::cerr << "Failed to load pos\n"; return 1;
    }
    if (!read_npy("/tmp/mnn_output_logits.npy", ref_logits, ref_shape)) {
        std::cerr << "Failed to load ref logits\n"; return 1;
    }

    std::cout << "Loaded:" << std::endl;
    std::cout << "  embeds: ["; for (auto d : emb_shape) std::cout << d << ","; std::cout << "]" << std::endl;
    std::cout << "  ref logits: ["; for (auto d : ref_shape) std::cout << d << ","; std::cout << "]" << std::endl;

    // Create MNN decoder module
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(dir + "/llm.mnn.weight");
    Module::Config mc;
    mc.shapeMutable = true;
    mc.rearrange = true;
    auto llm = Module::load({}, {}, (dir + "/llm.mnn").c_str(), rt, &mc);
    if (!llm) { std::cerr << "Module load FAILED\n"; return 1; }

    // Create MNN tensors from reference data
    int B = emb_shape[0], S = emb_shape[1], H = 1024;
    auto mnn_emb = _Input({B, S, H}, NCHW, halide_type_of<float>());
    memcpy(mnn_emb->writeMap<float>(), emb_data.data(), B * S * H * sizeof(float));

    int mask_B = mask_shape[0], mask_C = mask_shape[1], mask_S = mask_shape[2], mask_S2 = mask_shape[3];
    auto mnn_mask = _Input({mask_B, mask_C, mask_S, mask_S2}, NCHW, halide_type_of<float>());
    memcpy(mnn_mask->writeMap<float>(), mask_data.data(), mask_B * mask_C * mask_S * mask_S2 * sizeof(float));

    int pos_B = pos_shape[0], pos_S = pos_shape[1];
    auto mnn_pos = _Input({pos_B, pos_S}, NCHW, halide_type_of<int32_t>());
    memcpy(mnn_pos->writeMap<int32_t>(), pos_data.data(), pos_B * pos_S * sizeof(int32_t));

    // Run
    std::cout << "Running MNN decoder..." << std::endl;
    auto out = llm->onForward({mnn_emb, mnn_mask, mnn_pos});
    if (out.empty()) { std::cerr << "Forward FAILED\n"; return 1; }

    auto info = out[0]->getInfo();
    const float* mnn_logits = out[0]->readMap<float>();

    // Compare last token logits
    int V = info->dim[2];
    std::cout << "Output shape: [" << info->dim[0] << "," << info->dim[1] << "," << info->dim[2] << "]" << std::endl;

    // Find top-5 for last position
    int last_pos = info->dim[1] - 1;
    std::vector<std::pair<float, int>> mnn_top;
    for (int i = 0; i < V; i++) {
        mnn_top.push_back({mnn_logits[last_pos * V + i], i});
    }
    std::sort(mnn_top.rbegin(), mnn_top.rend());

    std::cout << "\nMNN decoder top-5:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  " << mnn_top[i].second << ": " << mnn_top[i].first << std::endl;
    }
    std::cout << "EOS(151645): " << mnn_logits[last_pos * V + 151645] << std::endl;

    // Compare with ONNX ref
    float mse = 0;
    for (int i = 0; i < V; i++) {
        float diff = mnn_logits[last_pos * V + i] - ref_logits[last_pos * V + i];
        mse += diff * diff;
    }
    mse /= V;
    std::cout << "\nMSE vs ONNX ref: " << mse << std::endl;

    // Cosine similarity
    float dot = 0, norm_m = 0, norm_r = 0;
    for (int i = 0; i < V; i++) {
        float m = mnn_logits[last_pos * V + i];
        float r = ref_logits[last_pos * V + i];
        dot += m * r;
        norm_m += m * m;
        norm_r += r * r;
    }
    float cos_sim = dot / (sqrt(norm_m) * sqrt(norm_r));
    std::cout << "Cosine similarity: " << cos_sim << std::endl;

    return 0;
}
