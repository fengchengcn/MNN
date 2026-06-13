// Audio Preprocessor — WebRTC-style AGC + HPF + Noise Gate
//
// Adaptive digital gain controller based on WebRTC AGC2:
//   webrtc.googlesource.com/src/+/HEAD/modules/audio_processing/agc2/
//
// Pipeline per audio chunk:
//   1. High-pass filter (DC removal + low-frequency noise)
//   2. Per-frame RMS computation
//   3. Noise floor estimation (minimum statistics)
//   4. Speech detection (RMS vs noise floor + margin)
//   5. Speech level estimation (smoothed, VAD-gated)
//   6. Target gain computation
//   7. Gain smoothing (attack/release time constants)
//   8. Apply gain + soft limiter

#include "audio_preprocessor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AudioPreprocessor::AudioPreprocessor(const Config& config)
    : cfg_(config)
    , noise_rms_(0.01f)
    , speech_rms_(0.01f)
    , current_gain_(1.0f)
    , hpf_x1_(0.0f), hpf_x2_(0.0f)
    , hpf_y1_(0.0f), hpf_y2_(0.0f)
    , hpf_b0_(0.0f), hpf_b1_(0.0f), hpf_b2_(0.0f)
    , hpf_a1_(0.0f), hpf_a2_(0.0f)
    , noise_gate_linear_(0.0f) {
    initHPF();
    noise_gate_linear_ = std::pow(10.0f, cfg_.noise_gate_db / 20.0f);
}

// ── High-Pass Filter ──
// 2nd-order Butterworth high-pass filter (biquad).
// Removes DC offset and low-frequency rumble (< 80 Hz).
// Coefficients computed using the bilinear transform.

void AudioPreprocessor::initHPF() {
    float fc = cfg_.hpf_cutoff_hz;
    float fs = static_cast<float>(cfg_.sample_rate);

    // Pre-warp cutoff frequency
    float omega = 2.0f * static_cast<float>(M_PI) * fc / fs;
    float sn = std::sin(omega);
    float cs = std::cos(omega);
    float alpha = sn / std::sqrt(2.0f);  // Q = 0.707 (Butterworth)

    // Biquad coefficients for 2nd-order HPF
    float b0 = (1.0f + cs) / 2.0f;
    float b1 = -(1.0f + cs);
    float b2 = (1.0f + cs) / 2.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cs;
    float a2 = 1.0f - alpha;

    // Normalize
    hpf_b0_ = b0 / a0;
    hpf_b1_ = b1 / a0;
    hpf_b2_ = b2 / a0;
    hpf_a1_ = a1 / a0;
    hpf_a2_ = a2 / a0;
}

float AudioPreprocessor::processHPF(float sample) {
    // Direct Form I biquad
    float y = hpf_b0_ * sample + hpf_b1_ * hpf_x1_ + hpf_b2_ * hpf_x2_
              - hpf_a1_ * hpf_y1_ - hpf_a2_ * hpf_y2_;

    hpf_x2_ = hpf_x1_;
    hpf_x1_ = sample;
    hpf_y2_ = hpf_y1_;
    hpf_y1_ = y;

    return y;
}

// ── Main Processing ──

void AudioPreprocessor::process(float* samples, int num_samples) {
    if (num_samples <= 0 || samples == nullptr) return;

    // ── Step 0: Compute chunk RMS (for noise gate and diagnostics) ──

    float sum_sq = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        sum_sq += samples[i] * samples[i];
    }
    float chunk_rms = std::sqrt(sum_sq / static_cast<float>(num_samples));

    // ── Step 1: Noise floor estimation (minimum statistics) ──
    //
    // WebRTC AGC2 tracks noise floor using a slow minimum tracker:
    //  - When frame RMS < noise_floor * 1.5: likely noise → fast update
    //  - Otherwise: slow upward creep to avoid locking onto silence
    //
    const float kNoiseFastUpdate = 0.95f;   // ~100ms window
    const float kNoiseSlowUpdate = 0.999f;  // ~5s window

    if (chunk_rms < noise_rms_ * 1.5f) {
        // Probable noise frame: update noise estimate quickly
        noise_rms_ = kNoiseFastUpdate * noise_rms_ + (1.0f - kNoiseFastUpdate) * chunk_rms;
    } else {
        // Probable speech or transient: slow upward adaptation only
        // Use the configured noise_smoothing for the slow rise
        noise_rms_ = cfg_.noise_smoothing * noise_rms_
                     + (1.0f - cfg_.noise_smoothing) * chunk_rms;
    }

    // ── Step 2: Noise gate ──
    //
    // If the entire chunk is below the noise gate threshold, pass through
    // with unity gain. This prevents amplifying pure silence/noise.
    if (chunk_rms < noise_gate_linear_) {
        // Still apply HPF (DC removal is always beneficial)
        for (int i = 0; i < num_samples; ++i) {
            samples[i] = processHPF(samples[i]);
        }
        // Keep gain at 1.0; don't update speech level
        return;
    }

    // ── Step 3: Speech detection ──
    //
    // A frame is "speech" if its RMS exceeds noise floor by the SNR margin.
    // This VAD-gated approach is the key WebRTC AGC2 innovation:
    // gain only adapts during actual speech, not during noise.
    bool is_speech = (chunk_rms > noise_rms_ * cfg_.snr_margin);

    // ── Step 4: Speech level estimation (VAD-gated) ──
    //
    // Fast-attack / slow-release envelope follower (WebRTC AGC2 design).
    //  - Upward (chunk > estimate): instant track — catches first phoneme.
    //  - Downward (chunk < estimate): slow smooth — avoids gain pumping.
    // Only updated during speech frames so background noise doesn't
    // pull the estimate down.
    if (is_speech) {
        if (chunk_rms > speech_rms_) {
            // Fast attack: catch up immediately to the speech level.
            // Prevents the first 500ms of speech from being over-gained
            // when the initial speech_rms_ seed (0.01) is far from truth.
            speech_rms_ = chunk_rms;
        } else {
            // Slow release: smooth downward to avoid pumping between words.
            speech_rms_ = cfg_.speech_smoothing * speech_rms_
                        + (1.0f - cfg_.speech_smoothing) * chunk_rms;
        }
    }
    // During non-speech, hold the last speech level estimate

    // ── Step 5: Compute target gain ──
    //
    // target = target_rms / effective_level
    // where effective_level = max(speech_rms, noise_rms * snr_margin)
    // The noise floor term ensures we don't apply extreme gain when
    // the speech estimate hasn't converged yet.
    float effective_level = std::max(speech_rms_, noise_rms_ * cfg_.snr_margin);
    float target_gain = cfg_.target_rms / (effective_level + 1e-8f);
    target_gain = std::max(cfg_.min_gain, std::min(cfg_.max_gain, target_gain));

    // ── Step 6: Gain smoothing (attack/release) ──
    //
    // WebRTC AGC2 uses asymmetric smoothing:
    //  - Attack (gain ↑): fast, to quickly boost quiet speech
    //  - Release (gain ↓): slow, to avoid gain pumping between words
    //
    // Smoothing coefficient: alpha = 1 - exp(-dt / tau)
    //  dt = num_samples / sample_rate
    //  tau = attack_seconds (if target > current) or release_seconds
    float dt = static_cast<float>(num_samples) / static_cast<float>(cfg_.sample_rate);
    float tau = (target_gain > current_gain_) ? cfg_.attack_seconds : cfg_.release_seconds;
    float alpha = 1.0f - std::exp(-dt / tau);

    // Clamp alpha to avoid instability with very small chunks
    alpha = std::max(0.0f, std::min(1.0f, alpha));

    float prev_gain = current_gain_;
    current_gain_ = alpha * target_gain + (1.0f - alpha) * prev_gain;

    // ── Step 7: Apply gain + HPF + soft limiter ──
    //
    // Process each sample:
    //   1. Apply HPF (DC removal)
    //   2. Apply smoothed gain
    //   3. Soft clip to prevent hard clipping on transients
    //
    // WebRTC uses per-sample gain interpolation to avoid zipper noise,
    // but for our chunk-based processing, using the smoothed gain is
    // sufficient since chunks are small (typically 10-30ms).

    float gain = current_gain_;
    float headroom = cfg_.headroom;

    for (int i = 0; i < num_samples; ++i) {
        // HPF
        float x = processHPF(samples[i]);

        // Apply gain
        x *= gain;

        // Soft limiter (WebRTC AGC2 uses a similar cubic soft-clipper)
        // tanh-based soft clip: smooth transition to ±headroom
        if (x > headroom) {
            x = headroom + (1.0f - headroom) * std::tanh((x - headroom) / (1.0f - headroom));
        } else if (x < -headroom) {
            x = -headroom - (1.0f - headroom) * std::tanh((-x - headroom) / (1.0f - headroom));
        }

        samples[i] = x;
    }
}

void AudioPreprocessor::reset() {
    noise_rms_ = 0.01f;
    speech_rms_ = 0.01f;
    current_gain_ = 1.0f;
    hpf_x1_ = hpf_x2_ = hpf_y1_ = hpf_y2_ = 0.0f;
}
