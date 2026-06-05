#include <llm/llm.hpp>
#include <MNN/expr/Executor.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif
using namespace MNN::Express;

int main() {
    std::string dir = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN";
    
    // Load MNN audio encoder
    auto ae_mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (!ae_mod) { std::cerr << "AE load fail\n"; return 1; }

    // Load audio
    auto lr = MNN::AUDIO::load("/tmp/test_short.wav", 16000);
    auto wf = lr.first;
    
    // Extract fbank
    auto feat = MNN::AUDIO::whisper_fbank(wf);
    { auto info = feat->getInfo(); auto p = feat->readMap<float>();
      auto f = _Input(info->dim, NCHW, halide_type_of<float>());
      memcpy(f->writeMap<float>(), p, info->size * sizeof(float)); feat = f; }
    
    auto info = feat->getInfo();
    std::cout << "MNN fbank shape: [" << info->dim[0] << "," << info->dim[1] << "," << info->dim[2] << "]" << std::endl;
    
    // Run audio encoder
    auto out = ae_mod->onForward({feat});
    if (out.empty()) return 1;
    auto oi = out[0]->getInfo();
    int T = oi->dim[1], H = oi->dim[2];
    std::cout << "MNN AE output: [" << oi->dim[0] << "," << T << "," << H << "]" << std::endl;
    
    // Print stats
    const float* d = out[0]->readMap<float>();
    float mean=0, std=0, max=-1e9, min=1e9;
    for (int i = 0; i < T * H; i++) { mean += d[i]; max = std::max(max, d[i]); min = std::min(min, d[i]); }
    mean /= (T * H);
    for (int i = 0; i < T * H; i++) std += (d[i]-mean)*(d[i]-mean);
    std = sqrt(std / (T * H));
    std::cout << "  mean=" << mean << " std=" << std << " max=" << max << " min=" << min << std::endl;
    std::cout << "  first 5: " << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << " " << d[4] << std::endl;
    
    // Also show fbank stats
    const float* fd = feat->readMap<float>();
    int fn = info->dim[0] * info->dim[1] * info->dim[2];
    float fm=0;
    for (int i=0;i<fn;i++) fm += fd[i];
    fm /= fn;
    std::cout << "Fbank mean=" << fm << std::endl;
    
    // Save AE output for ONNX comparison
    std::ofstream of("/tmp/mnn_ae_output.bin", std::ios::binary);
    of.write((char*)d, T * H * sizeof(float));
    of.close();
    std::cout << "Saved to /tmp/mnn_ae_output.bin" << std::endl;
    
    return 0;
}
