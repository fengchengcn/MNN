// Audio Encoder performance diagnostic
#include <llm/llm.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Interpreter.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>

#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif

using namespace MNN::Express;
#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED_MS(start) std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - (start)).count()

static void test_threads(const std::string& dir, int nt, const VARP& feat) {
    auto exec = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), nt);
    ExecutorScope scope(exec);
    auto mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (!mod) { std::cout << "  " << nt << " threads: FAIL\n"; return; }
    // Warmup: 2 runs
    mod->onForward({feat});
    mod->onForward({feat});
    // Timed
    auto t1 = NOW();
    mod->onForward({feat});
    std::cout << "  " << nt << " threads: " << ELAPSED_MS(t1) << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string dir = "/root/projects/MNN/mnn-models/Qwen3-ASR-0.6B-MNN";
    std::string wav = "/tmp/test_audio.wav";
    if (argc > 1) dir = argv[1];
    if (argc > 2) wav = argv[2];

    int ncpu = (int)std::thread::hardware_concurrency();
    if (ncpu <= 0) ncpu = 8;
    std::cout << "CPU cores: " << ncpu << std::endl;

#ifdef LLM_SUPPORT_AUDIO
    auto lr = MNN::AUDIO::load(wav, 16000);
    auto wf = lr.first;
    if (!wf.get()) { std::cerr << "FAIL load\n"; return 1; }
    auto feat = MNN::AUDIO::whisper_fbank_knf(wf);
    if (!feat.get() || !feat->getInfo()) { std::cerr << "FAIL fbank\n"; return 1; }
    { auto info = feat->getInfo(); auto p = feat->readMap<float>();
      auto f = _Input(info->dim, NCHW, halide_type_of<float>());
      memcpy(f->writeMap<float>(), p, info->size * sizeof(float)); feat = f; }
    auto fi = feat->getInfo();
    double dur = (wf->getInfo()->size) / 16000.0;
    std::cout << "Audio: " << dur << "s, FBank dims: " << fi->dim[0] << " " << fi->dim[1] << " " << fi->dim[2] << std::endl;

    // Thread scaling
    std::cout << "\n=== Thread scaling ===" << std::endl;
    for (int nt : {1, 2, 4, 8, 12, 16, 24, 32}) {
        if (nt > ncpu) continue;
        test_threads(dir, nt, feat);
    }

    // First-call vs warm
    int best = ncpu < 8 ? ncpu : 8;
    std::cout << "\n=== First-call penalty (" << best << " threads) ===" << std::endl;
    {
        auto exec = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), best);
        ExecutorScope scope(exec);
        auto mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
        for (int r = 0; r < 6; r++) {
            auto t1 = NOW();
            mod->onForward({feat});
            int dt = ELAPSED_MS(t1);
            if (r == 0) std::cout << "  First:  " << dt << " ms" << std::endl;
            else if (r == 1) std::cout << "  Second: " << dt << " ms" << std::endl;
            else if (r == 5) std::cout << "  Steady: " << dt << " ms" << std::endl;
        }
    }

    // Fixed input shape — second time should be fast
    std::cout << "\n=== Same shape, new module ===" << std::endl;
    {
        auto exec = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), best);
        ExecutorScope scope(exec);
        auto mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
        auto t1 = NOW();
        mod->onForward({feat});
        std::cout << "  Fresh load, first call: " << ELAPSED_MS(t1) << " ms" << std::endl;
    }

    std::cout << "\nDONE." << std::endl;
#else
    std::cerr << "LLM_SUPPORT_AUDIO not defined\n";
    return 1;
#endif
    return 0;
}
