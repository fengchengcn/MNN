// ASR Sampler Configuration — shared across all Qwen3-ASR inference paths
// Provides a MNN Sampler pre-configured for deterministic ASR (pure greedy).
//
// Usage:
//   // Option A: Read sampler params from config.json (recommended)
//   auto config = std::make_shared<LlmConfig>(config_json_path);
//   auto sampler = createAsrSampler(ctx, config);
//
//   // Option B: Use hardcoded defaults (fallback when no config.json)
//   auto sampler = createAsrSampler(ctx);
//
//   // Before each sample, update history:
//   ctx->history_tokens = token_ids;
//   int token = sampler->sample(logits);

#ifndef ASR_SAMPLER_HPP
#define ASR_SAMPLER_HPP

#include <memory>
#include <string>
#include "llmconfig.hpp"
#include "sampler.hpp"
#include "llm/llm.hpp"

namespace MNN {
namespace Transformer {

// ASR-optimized sampler defaults (Pure Greedy).
// Used as fallback when no config.json is provided.
// ASR is a deterministic task — one correct transcription per audio input.
// Pure greedy (argmax) for accuracy, matching the model's generation_config.json
// (do_sample=false, temperature≈0).
constexpr const char* ASR_SAMPLER_DEFAULTS_JSON = R"({
    "sampler_type": "greedy",
    "max_new_tokens": 256,
    "max_all_tokens": 2048
})";

// Create a Sampler pre-configured for ASR from config.json.
// When config is provided, sampler params are read from config.json (e.g.
// sampler_type, penalty, penalty_sampler, temperature, top_k, top_p, min_p,
// repetition_penalty). When nullptr, falls back to hardcoded defaults.
// Returns the sampler and (via out_ctx) the LlmContext that must be kept alive.
// Update out_ctx->history_tokens before each sample() call for penalty to work.
inline std::unique_ptr<Sampler> createAsrSampler(std::shared_ptr<LlmContext>& out_ctx,
                                                   std::shared_ptr<LlmConfig> config = nullptr) {
    out_ctx = std::make_shared<LlmContext>();
    if (!config) {
        config = std::make_shared<LlmConfig>();
        config->config_ = ujson::json::parse(ASR_SAMPLER_DEFAULTS_JSON);
    }
    return std::unique_ptr<Sampler>(Sampler::createSampler(out_ctx, config));
}

} // namespace Transformer
} // namespace MNN

#endif // ASR_SAMPLER_HPP
