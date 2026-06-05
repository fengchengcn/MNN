#!/usr/bin/env python3
"""
Compare ONNX Runtime vs MNN audio encoder outputs.
"""
import os
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb + 1e-8))

def main():
    print("=" * 60)
    print("Audio Encoder: MNN vs ONNX Runtime")
    print("=" * 60)

    # Load MNN dump
    print("\n[1] Loading MNN audio encoder output...")
    mnn_out = np.fromfile(os.path.join(MODEL_DIR, "dump_ae_output.bin"), dtype=np.float32)
    mnn_inp = np.fromfile(os.path.join(MODEL_DIR, "dump_ae_input.bin"), dtype=np.float32)

    # Read meta
    with open(os.path.join(MODEL_DIR, "dump_ae_meta.txt")) as f:
        meta = {}
        for line in f:
            for part in line.strip().split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k] = v

    C = int(meta["C"])
    T = int(meta["T"])
    out_shape = [int(x) for x in meta["output_shape"].split(",")]
    print(f"  Input: [1, {C}, {T}]")
    print(f"  Output shape: {out_shape}")
    mnn_out = mnn_out.reshape(out_shape)

    # Run ONNX Runtime
    print("\n[2] Running ONNX Runtime audio encoder...")
    onnx_path = os.path.join(MODEL_DIR, "onnx/audio_encoder.onnx")
    if not os.path.exists(onnx_path):
        print(f"  ERROR: {onnx_path} not found")
        return 1

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {'input_features': mnn_inp.reshape(1, C, T).astype(np.float32)}
    ort_out = sess.run(None, ort_inputs)
    ort_logits = ort_out[0]
    print(f"  Output: shape={ort_logits.shape}, range=[{ort_logits.min():.4f}, {ort_logits.max():.4f}]")

    # Compare
    print("\n[3] Comparing MNN vs ONNX Runtime:")
    cosim = cosine_sim(mnn_out, ort_logits)
    maxdiff = np.max(np.abs(mnn_out - ort_logits))
    meandiff = np.mean(np.abs(mnn_out - ort_logits))
    print(f"  Cosine similarity: {cosim:.6f}")
    print(f"  Max difference:    {maxdiff:.6f}")
    print(f"  Mean abs diff:     {meandiff:.6f}")
    if cosim > 0.9999:
        print(f"  => MATCH ✓")
    elif cosim > 0.99:
        print(f"  => CLOSE (within expected tolerance)")
    else:
        print(f"  => DIFFERENT ✗")

    # Per-position analysis
    print(f"\n[4] Per-position comparison (output seq):")
    for pos in range(out_shape[1]):
        m = mnn_out[0, pos]
        o = ort_logits[0, pos]
        c = cosine_sim(m, o)
        md = np.max(np.abs(m - o))
        marker = " <<<" if c < 0.999 else ""
        print(f"  pos {pos:2d}: cosim={c:.6f} maxdiff={md:.6f}{marker}")

    print(f"\n  MNN output range: [{mnn_out.min():.4f}, {mnn_out.max():.4f}]")
    print(f"  ONNX output range: [{ort_logits.min():.4f}, {ort_logits.max():.4f}]")

    # Compare statistics
    print(f"\n[5] Statistical comparison:")
    print(f"  MNN mean={mnn_out.mean():.6f} std={mnn_out.std():.6f}")
    print(f"  ONNX mean={ort_logits.mean():.6f} std={ort_logits.std():.6f}")
    print(f"  Mean diff: {np.mean(mnn_out - ort_logits):.6f}")

    print("\nDone!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
