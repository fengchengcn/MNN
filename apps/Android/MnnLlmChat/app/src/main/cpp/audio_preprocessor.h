// Audio Preprocessor — WebRTC-style AGC + HPF + Noise Gate
//
// Implements an adaptive digital gain controller based on the WebRTC AGC2
// algorithm (see webrtc.googlesource.com/src/+/HEAD/modules/audio_processing/agc2/).
//
// Key design decisions from WebRTC AGC2:
//   1. VAD-gated gain: gain only adjusts during speech, not noise
//   2. Separate noise/speech level estimators with different time constants
//   3. Noise floor tracked via minimum statistics (slow rise, fast fall)
//   4. Attack (gain up) is fast (~20ms), release (gain down) is slow (~200ms)
//   5. Soft limiter prevents clipping on transients
//   6. DC-blocking high-pass filter removes low-frequency noise
//
// Target use: ASR audio preprocessing before VAD. Normalizes varying
// microphone distances so the VAD and ASR model see consistent levels.

#ifndef AUDIO_PREPROCESSOR_H_
#define AUDIO_PREPROCESSOR_H_

#include <cmath>
#include <algorithm>

class AudioPreprocessor {
public:
    struct Config {
        float target_rms;
        int sample_rate;
        float hpf_cutoff_hz;
        float noise_smoothing;
        float speech_smoothing;
        float snr_margin;
        float attack_seconds;
        float release_seconds;
        float max_gain;
        float min_gain;
        float headroom;
        float noise_gate_db;

        // Default config tuned for ASR with varying mic distance.
        Config()
            : target_rms(0.5f)        // -6 dBFS, matches model training level
            , sample_rate(16000)       // Standard ASR sample rate
            , hpf_cutoff_hz(80.0f)     // Blocks DC and low-frequency rumble
            , noise_smoothing(0.999f)  // Slow noise floor tracking (~5s window)
            , speech_smoothing(0.95f)  // Faster speech level tracking (~300ms)
            , snr_margin(2.5f)         // Speech must exceed noise * 2.5 to trigger AGC
            , attack_seconds(0.02f)    // Fast gain increase (20ms)
            , release_seconds(0.20f)   // Slow gain decrease (200ms)
            , max_gain(20.0f)          // Max amplification (26 dB)
            , min_gain(1.0f)           // Never attenuate below unity
            , headroom(0.9f)           // 90% = -0.9 dBFS headroom
            , noise_gate_db(-50.0f)    // Below -50 dBFS → pass through
        {}
    };

    explicit AudioPreprocessor(const Config& config = Config());
    ~AudioPreprocessor() = default;

    // Process audio samples in-place. Samples must be float in [-1, 1].
    // Call this for each audio chunk before feeding to VAD.
    void process(float* samples, int num_samples);

    // Reset all internal state (noise floor, speech level, gain, HPF).
    // Call at the start of each new recording session.
    void reset();

    // Read-only access to current state for diagnostics
    float current_gain() const { return current_gain_; }
    float noise_rms() const { return noise_rms_; }
    float speech_rms() const { return speech_rms_; }

private:
    Config cfg_;

    float noise_rms_;       // Estimated noise floor RMS (linear)
    float speech_rms_;      // Estimated speech RMS (linear)
    float current_gain_;    // Current smoothed gain

    // 2nd-order Butterworth high-pass filter state (Direct Form I biquad)
    float hpf_x1_, hpf_x2_;
    float hpf_y1_, hpf_y2_;
    float hpf_b0_, hpf_b1_, hpf_b2_;
    float hpf_a1_, hpf_a2_;

    float noise_gate_linear_;

    void initHPF();
    float processHPF(float sample);
};

#endif  // AUDIO_PREPROCESSOR_H_
