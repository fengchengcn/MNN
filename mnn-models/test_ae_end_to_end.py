#!/usr/bin/env python3
"""
End-to-end test: generate fbank features, compare MNN vs ONNX Runtime for audio encoder.
Uses torchaudio/librosa for fbank (matching MNN's whisper_fbank parameters).
"""
import os
import sys
import numpy as np
import onnxruntime as ort

# Try to find fbank implementation
try:
    import librosa
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False
    print("librosa not available")

try:
    import torch
    import torchaudio
    HAS_TORCHAUDIO = True
except:
    HAS_TORCHAUDIO = False
    print("torchaudio not available")

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb + 1e-8))

def whisper_fbank_librosa(audio, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    """
    Compute whisper-compatible mel spectrogram using librosa.
    Match MNN's whisper_fbank parameters.
    """
    # Compute STFT with Hann window
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                     win_length=n_fft, window='hann', center=True)
    # Power spectrogram
    S = np.abs(D) ** 2
    # Mel filterbank - whisper uses no normalization
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                    fmin=0.0, fmax=8000.0, norm=None, htk=True)
    mel = mel_basis @ S
    # Log (whisper uses log after mel, with cutoff)
    mel = np.log10(np.maximum(mel, 1e-10))
    # Transpose to [n_mels, time]
    return mel.astype(np.float32)

def whisper_fbank_torchaudio(audio, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    """Compute whisper-compatible mel using torchaudio."""
    audio_t = torch.from_numpy(audio).float()
    # Create MelSpectrogram with whisper-compatible params
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_fn=torch.hann_window,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
        norm=None,
        mel_scale='htk',
        power=2.0,
    )
    mel = mel_transform(audio_t)
    mel = mel.squeeze(0)  # [n_mels, time]
    mel = torch.log10(torch.clamp(mel, min=1e-10))
    return mel.numpy().astype(np.float32)

def main():
    print("=" * 60)
    print("End-to-End Audio Encoder Test (Real Audio)")
    print("=" * 60)

    # Load audio
    audio_path = "/tmp/test_audio_real.pcm"
    audio = np.fromfile(audio_path, dtype=np.float32)
    print(f"\n[0] Loaded audio: {len(audio)} samples @ 16kHz = {len(audio)/16000:.1f}s")

    # Compute fbank features
    print("\n[1] Computing fbank features...")
    if HAS_TORCHAUDIO:
        feat = whisper_fbank_torchaudio(audio)
        source = "torchaudio"
    elif HAS_LIBROSA:
        feat = whisper_fbank_librosa(audio)
        source = "librosa"
    else:
        print("ERROR: No audio library available (need librosa or torchaudio)")
        return 1

    print(f"  Using {source}")
    print(f"  Feature shape: [{feat.shape[0]}, {feat.shape[1]}]")
    print(f"  Range: [{feat.min():.4f}, {feat.max():.4f}]")

    # Save features for C++ test
    feat.tofile(os.path.join(MODEL_DIR, "dump_real_features.bin"))
    print(f"  Saved features to dump_real_features.bin")
    with open(os.path.join(MODEL_DIR, "dump_real_meta.txt"), 'w') as f:
        f.write(f"C={feat.shape[0]} T={feat.shape[1]} samples={len(audio)}\n")

    # Run through ONNX Runtime audio encoder
    print("\n[2] Running ONNX Runtime audio encoder...")
    onnx_path = os.path.join(MODEL_DIR, "onnx/audio_encoder.onnx")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {'input_features': feat.reshape(1, feat.shape[0], feat.shape[1]).astype(np.float32)}
    ort_out = sess.run(None, ort_inputs)
    ort_logits = ort_out[0]
    print(f"  Output shape: {ort_logits.shape}")
    print(f"  Output range: [{ort_logits.min():.4f}, {ort_logits.max():.4f}]")

    # Also save ONNX output for comparison later
    ort_logits.tofile(os.path.join(MODEL_DIR, "dump_real_ort_ae.bin"))
    print(f"  Saved ONNX AE output to dump_real_ort_ae.bin")

    # Check the first token prediction
    # If we use ONNX decoder with this output:
    print("\n[3] Quick sanity: check first few frames of audio embedding...")
    print(f"  Frame 0: min={ort_logits[0,0].min():.4f} max={ort_logits[0,0].max():.4f} mean={ort_logits[0,0].mean():.4f}")
    print(f"  Frame 10: min={ort_logits[0,10].min():.4f} max={ort_logits[0,10].max():.4f} mean={ort_logits[0,10].mean():.4f}")

    print("\n[4] Summary:")
    print(f"  Audio duration: {len(audio)/16000:.1f}s")
    print(f"  Fbank features: {feat.shape[1]} frames, {feat.shape[0]} mels")
    print(f"  Audio encoder output: {ort_logits.shape[1]} frames, {ort_logits.shape[2]} dims")
    print(f"  Compression factor: {feat.shape[1]} / {ort_logits.shape[1]} = {feat.shape[1]/ort_logits.shape[1]:.1f}x")

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
