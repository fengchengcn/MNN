// Self-contained Silero VAD + VoiceActivityDetector JNI implementation.
// Uses MNN Interpreter API (low-level) to load silero_vad.onnx directly.
// Compiled into libmnnllmapp.so alongside the rest of the app's native code.

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#define TAG "silero-vad-jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// =========================================================================
// Config structs — mirror Kotlin data classes in Vad.kt
// =========================================================================

struct SileroVadConfig {
    std::string model;
    float threshold = 0.5f;
    float minSilenceDuration = 0.25f;
    float minSpeechDuration = 0.25f;
    int32_t windowSize = 512;
    float maxSpeechDuration = 5.0f;
};

struct VadConfig {
    SileroVadConfig sileroVad;
    int32_t sampleRate = 16000;
    int32_t numThreads = 1;
    std::string provider = "cpu";
    bool debug = false;
};

// =========================================================================
// File reading utilities
// =========================================================================

static std::vector<char> readFileFromAsset(AAssetManager* mgr, const std::string& filename) {
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("Failed to open asset: %s", filename.c_str());
        return {};
    }
    size_t size = AAsset_getLength(asset);
    const void* buf = AAsset_getBuffer(asset);
    std::vector<char> result(reinterpret_cast<const char*>(buf),
                             reinterpret_cast<const char*>(buf) + size);
    AAsset_close(asset);
    LOGI("Read %zu bytes from asset: %s", size, filename.c_str());
    return result;
}

static std::vector<char> readFileFromDisk(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        LOGE("Failed to open file: %s", filename.c_str());
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) {
        LOGE("Failed to read file: %s", filename.c_str());
        return {};
    }
    LOGI("Read %zu bytes from file: %s", static_cast<size_t>(size), filename.c_str());
    return buffer;
}

// =========================================================================
// CircularBuffer — linear-index ring buffer for audio samples
// =========================================================================

class CircularBuffer {
public:
    explicit CircularBuffer(int32_t capacity) : buffer_(static_cast<size_t>(capacity)) {}

    void push(const float* p, int32_t n) {
        int32_t cap = static_cast<int32_t>(buffer_.size());
        int32_t sz = tail_ - head_;
        if (n + sz > cap) {
            int32_t newCap = std::max(cap * 2, n + sz);
            resize(newCap);
            cap = newCap;
        }
        int32_t start = tail_ % cap;
        tail_ += n;
        if (start + n <= cap) {
            std::copy(p, p + n, buffer_.begin() + start);
        } else {
            int32_t part1 = cap - start;
            std::copy(p, p + part1, buffer_.begin() + start);
            std::copy(p + part1, p + n, buffer_.begin());
        }
    }

    std::vector<float> get(int32_t startIdx, int32_t n) const {
        int32_t cap = static_cast<int32_t>(buffer_.size());
        int32_t start = startIdx % cap;
        std::vector<float> ans(static_cast<size_t>(n));
        if (start + n <= cap) {
            std::copy(buffer_.begin() + start, buffer_.begin() + start + n, ans.begin());
        } else {
            int32_t part1 = cap - start;
            std::copy(buffer_.begin() + start, buffer_.end(), ans.begin());
            std::copy(buffer_.begin(), buffer_.begin() + (n - part1), ans.begin() + part1);
        }
        return ans;
    }

    void pop(int32_t n) { head_ += n; }
    int32_t size() const { return tail_ - head_; }
    int32_t head() const { return head_; }
    int32_t tail() const { return tail_; }
    void reset() { head_ = 0; tail_ = 0; }

private:
    void resize(int32_t newCap) {
        int32_t oldCap = static_cast<int32_t>(buffer_.size());
        int32_t sz = size();
        if (sz == 0) {
            buffer_.resize(static_cast<size_t>(newCap));
            return;
        }
        std::vector<float> newBuf(static_cast<size_t>(newCap));
        int32_t oldStart = head_ % oldCap;
        if (oldStart + sz <= oldCap) {
            std::copy(buffer_.begin() + oldStart, buffer_.begin() + oldStart + sz,
                      newBuf.begin() + (head_ % newCap));
        } else {
            int32_t part1 = oldCap - oldStart;
            int32_t newStart = head_ % newCap;
            std::copy(buffer_.begin() + oldStart, buffer_.end(), newBuf.begin() + newStart);
            std::copy(buffer_.begin(), buffer_.begin() + (sz - part1),
                      newBuf.begin() + newStart + part1);
        }
        buffer_.swap(newBuf);
    }

    std::vector<float> buffer_;
    int32_t head_ = 0;
    int32_t tail_ = 0;
};

// =========================================================================
// SileroVadModel — loads silero_vad.onnx via MNN Interpreter, runs inference
// =========================================================================

class SileroVadModel {
public:
    SileroVadModel(AAssetManager* mgr, const VadConfig& config)
        : config_(config)
        , sampleRate_(config.sampleRate) {
        auto buf = readFileFromAsset(mgr, config.sileroVad.model);
        if (!buf.empty()) {
            init(buf.data(), buf.size());
        } else {
            LOGE("Failed to read model from asset");
            valid_ = false;
        }
    }

    SileroVadModel(const VadConfig& config)
        : config_(config)
        , sampleRate_(config.sampleRate) {
        auto buf = readFileFromDisk(config.sileroVad.model);
        if (!buf.empty()) {
            init(buf.data(), buf.size());
        } else {
            LOGE("Failed to read model from file");
            valid_ = false;
        }
    }

    bool valid() const { return valid_; }

    void reset() {
        if (!valid_) return;

        if (isV5_) {
            // V5: state shape {2, 1, 128}
            hState_.assign(2 * 1 * 128, 0.0f);
        } else {
            // V4: h and c, each {2, 1, 64}
            hState_.assign(2 * 1 * 64, 0.0f);
            cState_.assign(2 * 1 * 64, 0.0f);
        }
        triggered_ = false;
        currentSample_ = 0;
        tempStart_ = 0;
        tempEnd_ = 0;
    }

    bool isSpeech(const float* samples, int32_t n) {
        if (!valid_) return false;

        float prob = run(samples, n);
        float threshold = config_.sileroVad.threshold;

        currentSample_ += config_.sileroVad.windowSize;

        // Silero VAD state machine (from snakers4/silero-vad)
        if (prob > threshold && tempEnd_ != 0) {
            tempEnd_ = 0;
        }

        if (prob > threshold && tempStart_ == 0) {
            tempStart_ = currentSample_;
            return false;
        }

        if (prob > threshold && tempStart_ != 0 && !triggered_) {
            if (currentSample_ - tempStart_ < minSpeechSamples_) {
                return false;
            }
            triggered_ = true;
            return true;
        }

        if ((prob < threshold) && !triggered_) {
            tempStart_ = 0;
            tempEnd_ = 0;
            return false;
        }

        if ((prob > threshold - 0.15f) && triggered_) {
            return true;
        }

        if ((prob > threshold) && !triggered_) {
            triggered_ = true;
            return true;
        }

        if ((prob < threshold) && triggered_) {
            if (tempEnd_ == 0) {
                tempEnd_ = currentSample_;
            }
            if (currentSample_ - tempEnd_ < minSilenceSamples_) {
                return true;
            }
            tempStart_ = 0;
            tempEnd_ = 0;
            triggered_ = false;
            return false;
        }

        return false;
    }

    int32_t windowSize() const { return config_.sileroVad.windowSize + windowOverlap_; }
    int32_t windowShift() const { return config_.sileroVad.windowSize; }
    int32_t minSilenceDurationSamples() const { return minSilenceSamples_; }
    int32_t minSpeechDurationSamples() const { return minSpeechSamples_; }

    void setMinSilenceDuration(float s) {
        minSilenceSamples_ = static_cast<int32_t>(config_.sampleRate * s);
    }
    void setThreshold(float t) { config_.sileroVad.threshold = t; }

private:
    void init(void* modelData, size_t modelSize) {
        // Load MNN-format model via MNN Interpreter
        interpreter_ = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromBuffer(modelData, modelSize));
        if (!interpreter_) {
            LOGE("Interpreter::createFromBuffer returned nullptr");
            valid_ = false;
            return;
        }

        // Set up schedule config
        MNN::ScheduleConfig schedConfig;
        schedConfig.numThread = config_.numThreads;
        MNN::BackendConfig backendConfig;
        backendConfig.memory = MNN::BackendConfig::Memory_Low;
        schedConfig.backendConfig = &backendConfig;

        session_ = interpreter_->createSession(schedConfig);
        if (!session_) {
            LOGE("Failed to create MNN session");
            valid_ = false;
            return;
        }

        // Detect V4 vs V5. The MNN converter folds the constant 'sr' input,
        // so V4 becomes 3 inputs (x, h, c) and V5 becomes 2 inputs (x, state).
        auto allInputs = interpreter_->getSessionInputAll(session_);
        auto allOutputs = interpreter_->getSessionOutputAll(session_);

        LOGI("Model has %zu inputs, %zu outputs", allInputs.size(), allOutputs.size());

        // Get tensor by a set of possible names (handles both raw ONNX and MNN-converted)
        auto getInputByName = [&](std::initializer_list<const char*> names) -> MNN::Tensor* {
            for (auto name : names) {
                auto it = allInputs.find(name);
                if (it != allInputs.end()) return it->second;
            }
            return nullptr;
        };
        auto getOutputByName = [&](std::initializer_list<const char*> names) -> MNN::Tensor* {
            for (auto name : names) {
                auto it = allOutputs.find(name);
                if (it != allOutputs.end()) return it->second;
            }
            return nullptr;
        };

        // Log tensor names for debugging
        for (const auto& kv : allInputs) {
            LOGI("  Input: %s shape=%s", kv.first.c_str(),
                 tensorShapeStr(kv.second).c_str());
        }
        for (const auto& kv : allOutputs) {
            LOGI("  Output: %s shape=%s", kv.first.c_str(),
                 tensorShapeStr(kv.second).c_str());
        }

        if (allInputs.size() == 3 && allOutputs.size() == 3) {
            // V4 (MNN-converted): inputs = [x, h, c], outputs = [prob, new_h, new_c]
            isV5_ = false;

            inputTensor_ = getInputByName({"x", "input"});
            hTensor_     = getInputByName({"h"});
            cTensor_     = getInputByName({"c"});

            outputTensor_ = getOutputByName({"prob", "output"});
            hnTensor_     = getOutputByName({"new_h", "hn"});
            cnTensor_     = getOutputByName({"new_c", "cn"});

            if (!inputTensor_ || !hTensor_ || !cTensor_ ||
                !outputTensor_ || !hnTensor_ || !cnTensor_) {
                LOGE("Failed to get V4 I/O tensors");
                valid_ = false;
                return;
            }
            interpreter_->resizeTensor(inputTensor_, {1, windowSize()});
            interpreter_->resizeTensor(hTensor_, {2, 1, 64});
            interpreter_->resizeTensor(cTensor_, {2, 1, 64});
        } else if (allInputs.size() == 2 && allOutputs.size() == 2) {
            // V5 (MNN-converted): inputs = [x, state], outputs = [prob, new_state]
            isV5_ = true;
            windowOverlap_ = 64;

            inputTensor_   = getInputByName({"x", "input"});
            stateTensor_   = getInputByName({"state"});

            outputTensor_   = getOutputByName({"prob", "output"});
            stateOutTensor_ = getOutputByName({"new_state", "stateN"});

            if (!inputTensor_ || !stateTensor_ ||
                !outputTensor_ || !stateOutTensor_) {
                LOGE("Failed to get V5 I/O tensors");
                valid_ = false;
                return;
            }
            interpreter_->resizeTensor(inputTensor_, {1, windowSize()});
            interpreter_->resizeTensor(stateTensor_, {2, 1, 128});
        } else if (allInputs.size() == 4 && allOutputs.size() == 3) {
            // V4 (raw ONNX): inputs = [input, sr, h, c], outputs = [output, hn, cn]
            isV5_ = false;

            inputTensor_ = getInputByName({"input"});
            srTensor_    = getInputByName({"sr"});
            hTensor_     = getInputByName({"h"});
            cTensor_     = getInputByName({"c"});

            outputTensor_ = getOutputByName({"output"});
            hnTensor_     = getOutputByName({"hn"});
            cnTensor_     = getOutputByName({"cn"});

            if (!inputTensor_ || !srTensor_ || !hTensor_ || !cTensor_ ||
                !outputTensor_ || !hnTensor_ || !cnTensor_) {
                LOGE("Failed to get V4 (ONNX) I/O tensors");
                valid_ = false;
                return;
            }
            interpreter_->resizeTensor(inputTensor_, {1, windowSize()});
            interpreter_->resizeTensor(srTensor_, {1});
            interpreter_->resizeTensor(hTensor_, {2, 1, 64});
            interpreter_->resizeTensor(cTensor_, {2, 1, 64});
        } else if (allInputs.size() == 3 && allOutputs.size() == 2) {
            // V5 (raw ONNX): inputs = [input, state, sr], outputs = [output, stateN]
            isV5_ = true;
            windowOverlap_ = 64;

            inputTensor_   = getInputByName({"input"});
            stateTensor_   = getInputByName({"state"});
            srTensor_      = getInputByName({"sr"});

            outputTensor_   = getOutputByName({"output"});
            stateOutTensor_ = getOutputByName({"stateN"});

            if (!inputTensor_ || !stateTensor_ || !srTensor_ ||
                !outputTensor_ || !stateOutTensor_) {
                LOGE("Failed to get V5 (ONNX) I/O tensors");
                valid_ = false;
                return;
            }
            interpreter_->resizeTensor(inputTensor_, {1, windowSize()});
            interpreter_->resizeTensor(stateTensor_, {2, 1, 128});
            interpreter_->resizeTensor(srTensor_, {1});
        } else {
            LOGE("Unsupported Silero VAD model: %zu inputs, %zu outputs",
                 allInputs.size(), allOutputs.size());
            valid_ = false;
            return;
        }

        interpreter_->resizeSession(session_);

        minSilenceSamples_ =
            static_cast<int32_t>(config_.sampleRate * config_.sileroVad.minSilenceDuration);
        minSpeechSamples_ =
            static_cast<int32_t>(config_.sampleRate * config_.sileroVad.minSpeechDuration);

        valid_ = true;
        reset();
    }

    float run(const float* samples, int32_t n) {
        if (isV5_) {
            return runV5(samples, n);
        } else {
            return runV4(samples, n);
        }
    }

    float runV4(const float* samples, int32_t n) {
        // Fill audio input tensor
        interpreter_->resizeTensor(inputTensor_, {1, n});
        interpreter_->resizeSession(session_);

        auto inputHost = inputTensor_->host<float>();
        std::memcpy(inputHost, samples, n * sizeof(float));

        // Fill sr tensor if present (raw ONNX only; MNN converter folds it as constant)
        if (srTensor_) {
            auto srHost = srTensor_->host<int64_t>();
            srHost[0] = static_cast<int64_t>(sampleRate_);
        }

        // Fill h state
        auto hHost = hTensor_->host<float>();
        std::memcpy(hHost, hState_.data(), hState_.size() * sizeof(float));

        // Fill c state
        auto cHost = cTensor_->host<float>();
        std::memcpy(cHost, cState_.data(), cState_.size() * sizeof(float));

        // Run inference
        interpreter_->runSession(session_);

        // Read output probability
        float prob = outputTensor_->host<float>()[0];

        // Read new states
        auto hnHost = hnTensor_->host<float>();
        auto hnSize = hnTensor_->elementSize();
        hState_.resize(hnSize);
        std::memcpy(hState_.data(), hnHost, hnSize * sizeof(float));

        auto cnHost = cnTensor_->host<float>();
        auto cnSize = cnTensor_->elementSize();
        cState_.resize(cnSize);
        std::memcpy(cState_.data(), cnHost, cnSize * sizeof(float));

        return prob;
    }

    float runV5(const float* samples, int32_t n) {
        // Fill audio input tensor
        interpreter_->resizeTensor(inputTensor_, {1, n});
        interpreter_->resizeSession(session_);

        auto inputHost = inputTensor_->host<float>();
        std::memcpy(inputHost, samples, n * sizeof(float));

        // Fill sr tensor if present (raw ONNX only)
        if (srTensor_) {
            auto srHost = srTensor_->host<int64_t>();
            srHost[0] = static_cast<int64_t>(sampleRate_);
        }

        // Fill LSTM state
        auto stateHost = stateTensor_->host<float>();
        std::memcpy(stateHost, hState_.data(), hState_.size() * sizeof(float));

        // Run inference
        interpreter_->runSession(session_);

        // Read output probability
        float prob = outputTensor_->host<float>()[0];

        // Read new state
        auto stateOutHost = stateOutTensor_->host<float>();
        auto stateOutSize = stateOutTensor_->elementSize();
        hState_.resize(stateOutSize);
        std::memcpy(hState_.data(), stateOutHost, stateOutSize * sizeof(float));

        return prob;
    }

    static std::string tensorShapeStr(MNN::Tensor* t) {
        if (!t) return "null";
        auto dims = t->shape();
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }

    VadConfig config_;
    int32_t sampleRate_ = 16000;

    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;

    // I/O tensors (V4)
    MNN::Tensor* inputTensor_ = nullptr;
    MNN::Tensor* srTensor_ = nullptr;
    MNN::Tensor* hTensor_ = nullptr;
    MNN::Tensor* cTensor_ = nullptr;
    MNN::Tensor* outputTensor_ = nullptr;
    MNN::Tensor* hnTensor_ = nullptr;
    MNN::Tensor* cnTensor_ = nullptr;

    // I/O tensors (V5)
    MNN::Tensor* stateTensor_ = nullptr;
    MNN::Tensor* stateOutTensor_ = nullptr;

    // LSTM states stored as float vectors
    std::vector<float> hState_;
    std::vector<float> cState_;

    bool valid_ = false;
    bool isV5_ = false;
    int32_t windowOverlap_ = 0;

    // VAD state machine
    bool triggered_ = false;
    int32_t currentSample_ = 0;
    int32_t tempStart_ = 0;
    int32_t tempEnd_ = 0;

    int32_t minSilenceSamples_ = 0;
    int32_t minSpeechSamples_ = 0;
};

// =========================================================================
// VoiceActivityDetector — accumulates audio, extracts speech segments
// =========================================================================

class VoiceActivityDetector {
public:
    struct SpeechSegment {
        int32_t start;
        std::vector<float> samples;
    };

    VoiceActivityDetector(AAssetManager* mgr, const VadConfig& config)
        : model_(std::make_unique<SileroVadModel>(mgr, config))
        , config_(config)
        , buffer_(config.sampleRate * 60)
    {
        maxUtteranceLength_ =
            static_cast<int32_t>(config.sampleRate * config.sileroVad.maxSpeechDuration);
    }

    VoiceActivityDetector(const VadConfig& config)
        : model_(std::make_unique<SileroVadModel>(config))
        , config_(config)
        , buffer_(config.sampleRate * 60)
    {
        maxUtteranceLength_ =
            static_cast<int32_t>(config.sampleRate * config.sileroVad.maxSpeechDuration);
    }

    bool valid() const { return model_->valid(); }

    void acceptWaveform(const float* samples, int32_t n) {
        if (!model_->valid()) return;

        if (buffer_.size() > maxUtteranceLength_) {
            model_->setMinSilenceDuration(0.1f);
            model_->setThreshold(0.9f);
        } else {
            model_->setMinSilenceDuration(config_.sileroVad.minSilenceDuration);
            model_->setThreshold(config_.sileroVad.threshold);
        }

        int32_t windowSize = model_->windowSize();
        int32_t windowShift = model_->windowShift();

        last_.insert(last_.end(), samples, samples + n);

        if (static_cast<int32_t>(last_.size()) < windowSize) {
            return;
        }

        int32_t k = (static_cast<int32_t>(last_.size()) - windowSize) / windowShift + 1;
        const float* p = last_.data();
        bool isSpeech = false;

        for (int32_t i = 0; i < k; ++i, p += windowShift) {
            buffer_.push(p, windowShift);
            bool thisWindow = model_->isSpeech(p, windowSize);
            isSpeech = isSpeech || thisWindow;
        }

        last_ = std::vector<float>(p, last_.data() + last_.size());

        if (isSpeech) {
            if (start_ == -1) {
                start_ = std::max(buffer_.tail() - 2 * model_->windowSize()
                                  - model_->minSpeechDurationSamples(),
                                  buffer_.head());
            }
        } else {
            if (start_ != -1 && buffer_.size() > 0) {
                int32_t end = buffer_.tail() - model_->minSilenceDurationSamples();
                if (end > start_) {
                    auto s = buffer_.get(start_, end - start_);
                    SpeechSegment segment;
                    segment.start = start_;
                    segment.samples = std::move(s);
                    segments_.push(std::move(segment));
                    buffer_.pop(end - buffer_.head());
                }
            }

            if (start_ == -1) {
                int32_t end = buffer_.tail() - 2 * model_->windowSize()
                              - model_->minSpeechDurationSamples();
                int32_t nPop = std::max(0, end - buffer_.head());
                if (nPop > 0) {
                    buffer_.pop(nPop);
                }
            }

            start_ = -1;
        }
    }

    bool empty() const { return segments_.empty(); }
    void pop() { segments_.pop(); }
    void clear() { std::queue<SpeechSegment>().swap(segments_); }
    const SpeechSegment& front() const { return segments_.front(); }

    void reset() {
        std::queue<SpeechSegment>().swap(segments_);
        model_->reset();
        buffer_.reset();
        start_ = -1;
    }

    void flush() {
        if (start_ == -1 || buffer_.size() == 0) return;
        int32_t end = buffer_.tail();
        if (end <= start_) return;
        auto s = buffer_.get(start_, end - start_);
        SpeechSegment segment;
        segment.start = start_;
        segment.samples = std::move(s);
        segments_.push(std::move(segment));
        buffer_.pop(end - buffer_.head());
        start_ = -1;
    }

    bool isSpeechDetected() const { return start_ != -1; }

private:
    std::unique_ptr<SileroVadModel> model_;
    VadConfig config_;
    CircularBuffer buffer_;
    std::vector<float> last_;
    std::queue<SpeechSegment> segments_;
    int32_t start_ = -1;
    int32_t maxUtteranceLength_ = -1;
};

// =========================================================================
// JNI helper: parse VadConfig from Kotlin objects
// =========================================================================

static VadConfig parseVadConfig(JNIEnv* env, jobject configObj) {
    VadConfig cfg;

    jclass cls = env->GetObjectClass(configObj);

    jfieldID fid = env->GetFieldID(cls, "sileroVadModelConfig",
                                   "Lcom/k2fsa/sherpa/mnn/SileroVadModelConfig;");
    jobject sileroObj = env->GetObjectField(configObj, fid);
    if (sileroObj == nullptr) {
        LOGE("sileroVadModelConfig is null");
        env->DeleteLocalRef(cls);
        return cfg;
    }
    jclass sileroCls = env->GetObjectClass(sileroObj);

    fid = env->GetFieldID(sileroCls, "model", "Ljava/lang/String;");
    auto s = static_cast<jstring>(env->GetObjectField(sileroObj, fid));
    if (s) {
        auto p = env->GetStringUTFChars(s, nullptr);
        cfg.sileroVad.model = p;
        env->ReleaseStringUTFChars(s, p);
    }

    fid = env->GetFieldID(sileroCls, "threshold", "F");
    cfg.sileroVad.threshold = env->GetFloatField(sileroObj, fid);

    fid = env->GetFieldID(sileroCls, "minSilenceDuration", "F");
    cfg.sileroVad.minSilenceDuration = env->GetFloatField(sileroObj, fid);

    fid = env->GetFieldID(sileroCls, "minSpeechDuration", "F");
    cfg.sileroVad.minSpeechDuration = env->GetFloatField(sileroObj, fid);

    fid = env->GetFieldID(sileroCls, "windowSize", "I");
    cfg.sileroVad.windowSize = env->GetIntField(sileroObj, fid);

    fid = env->GetFieldID(sileroCls, "maxSpeechDuration", "F");
    cfg.sileroVad.maxSpeechDuration = env->GetFloatField(sileroObj, fid);

    env->DeleteLocalRef(sileroCls);
    env->DeleteLocalRef(sileroObj);

    fid = env->GetFieldID(cls, "sampleRate", "I");
    cfg.sampleRate = env->GetIntField(configObj, fid);

    fid = env->GetFieldID(cls, "numThreads", "I");
    cfg.numThreads = env->GetIntField(configObj, fid);

    fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
    s = static_cast<jstring>(env->GetObjectField(configObj, fid));
    if (s) {
        auto p = env->GetStringUTFChars(s, nullptr);
        cfg.provider = p;
        env->ReleaseStringUTFChars(s, p);
    }

    fid = env->GetFieldID(cls, "debug", "Z");
    cfg.debug = env->GetBooleanField(configObj, fid);

    env->DeleteLocalRef(cls);
    return cfg;
}

static jobject NewInteger(JNIEnv* env, int32_t value) {
    jclass cls = env->FindClass("java/lang/Integer");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(I)V");
    jobject obj = env->NewObject(cls, ctor, value);
    env->DeleteLocalRef(cls);
    return obj;
}

// =========================================================================
// JNI: native method implementations (same signatures as libsherpa-mnn-jni)
// =========================================================================

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_newFromAsset(
    JNIEnv* env, jobject /*thiz*/, jobject assetManager, jobject configObj) {
    auto* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("newFromAsset: Failed to get AAssetManager");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "Failed to get native AssetManager");
        return 0;
    }
    auto config = parseVadConfig(env, configObj);
    auto* vad = new VoiceActivityDetector(mgr, config);
    if (!vad->valid()) {
        LOGE("newFromAsset: VAD model failed to initialize");
        delete vad;
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "Failed to load Silero VAD model from assets");
        return 0;
    }
    LOGI("newFromAsset: VAD created successfully");
    return reinterpret_cast<jlong>(vad);
}

JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_newFromFile(
    JNIEnv* env, jobject /*thiz*/, jobject configObj) {
    auto config = parseVadConfig(env, configObj);
    if (config.sileroVad.model.empty()) {
        LOGE("newFromFile: model path is empty");
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "Model path is empty");
        return 0;
    }
    auto* vad = new VoiceActivityDetector(config);
    if (!vad->valid()) {
        LOGE("newFromFile: VAD model failed to initialize");
        delete vad;
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "Failed to load Silero VAD model from file");
        return 0;
    }
    return reinterpret_cast<jlong>(vad);
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_delete(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    delete reinterpret_cast<VoiceActivityDetector*>(ptr);
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_acceptWaveform(
    JNIEnv* env, jobject /*thiz*/, jlong ptr, jfloatArray samples) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (!vad) return;

    jfloat* p = env->GetFloatArrayElements(samples, nullptr);
    jsize n = env->GetArrayLength(samples);
    vad->acceptWaveform(p, n);
    env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_empty(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    return vad ? vad->empty() : JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_pop(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (vad) vad->pop();
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_clear(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (vad) vad->clear();
}

JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_front(
    JNIEnv* env, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (!vad || vad->empty()) {
        return nullptr;
    }

    const auto& front = vad->front();

    jfloatArray samplesArr = env->NewFloatArray(
        static_cast<jsize>(front.samples.size()));
    env->SetFloatArrayRegion(samplesArr, 0,
                             static_cast<jsize>(front.samples.size()),
                             front.samples.data());

    jobjectArray result = static_cast<jobjectArray>(
        env->NewObjectArray(2, env->FindClass("java/lang/Object"), nullptr));
    env->SetObjectArrayElement(result, 0, NewInteger(env, front.start));
    env->SetObjectArrayElement(result, 1, samplesArr);
    env->DeleteLocalRef(samplesArr);

    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_isSpeechDetected(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    return vad ? vad->isSpeechDetected() : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_reset(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (vad) vad->reset();
}

JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_mnn_Vad_flush(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* vad = reinterpret_cast<VoiceActivityDetector*>(ptr);
    if (vad) vad->flush();
}

}  // extern "C"
