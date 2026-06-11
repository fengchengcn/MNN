// ASR Sampler Configuration — shared across all Qwen3-ASR inference paths
// Provides a MNN Sampler pre-configured for deterministic ASR (penalty + greedy).
//
// Usage:
//   std::shared_ptr<MNN::Transformer::LlmContext> ctx;
//   auto sampler = MNN::Transformer::createAsrSampler(ctx);
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

// ASR-optimized sampler config (Penalty + Greedy).
// ASR is a deterministic task — one correct transcription per audio input.
// Greedy selection is preferred over temperature-based sampling for accuracy.
// Repetition penalty (1.15) prevents the decoder from looping on repeated tokens.
constexpr const char* ASR_SAMPLER_CONFIG_JSON = R"({
    "sampler_type": "penalty",
    "penalty": 1.15,
    "penalty_sampler": "greedy",
    "max_new_tokens": 256,
    "max_all_tokens": 2048
})";

// Create a Sampler pre-configured for ASR.
// Returns the sampler and (via out_ctx) the LlmContext that must be kept alive.
// Update out_ctx->history_tokens before each sample() call for penalty to work.
inline std::unique_ptr<Sampler> createAsrSampler(std::shared_ptr<LlmContext>& out_ctx) {
    out_ctx = std::make_shared<LlmContext>();
    auto config = std::make_shared<LlmConfig>();
    config->config_ = ujson::json::parse(ASR_SAMPLER_CONFIG_JSON);
    return std::unique_ptr<Sampler>(Sampler::createSampler(out_ctx, config));
}

} // namespace Transformer
} // namespace MNN

#endif // ASR_SAMPLER_HPP
