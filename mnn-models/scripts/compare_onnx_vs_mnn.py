#!/usr/bin/env python3
"""
Compare ONNX Runtime vs MNN decoder outputs.
Loads dumped inputs from C++ test and runs ONNX Runtime with the same input.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"

def load_dump(path, dtype=np.float32):
    """Load binary dump file."""
    data = np.fromfile(path, dtype=dtype)
    print(f"  Loaded {path}: {data.shape}, range [{data.min():.4f}, {data.max():.4f}]")
    return data

def cosine_sim(a, b):
    """Compute cosine similarity."""
    dot = np.dot(a.flatten(), b.flatten())
    na = np.linalg.norm(a.flatten())
    nb = np.linalg.norm(b.flatten())
    return float(dot / (na * nb + 1e-8))

def main():
    print("=" * 60)
    print("MNN vs ONNX Runtime Decoder Comparison")
    print("=" * 60)

    # Read metadata - each line is "key=value" or space-separated "key=value" entries
    with open(os.path.join(MODEL_DIR, "dump_meta.txt")) as f:
        meta = {}
        for line in f:
            line = line.strip()
            if not line: continue
            # Handle both "k=v" per line and "k1=v1 k2=v2" on one line
            for part in line.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k.strip()] = v.strip()
    B = int(meta["B"])
    S = int(meta["S"])
    D = int(meta["D"])
    V = int(meta["V"])
    print(f"\nInput: B={B} S={S} D={D} V={V}")

    # Load MNN dumped outputs
    print("\n[1] Loading MNN dump files...")
    opt_logits = load_dump(os.path.join(MODEL_DIR, "dump_opt_logits.bin")).reshape(B, S, V)
    noopt_logits = load_dump(os.path.join(MODEL_DIR, "dump_noopt_logits.bin")).reshape(B, S, V)
    input_embeds = load_dump(os.path.join(MODEL_DIR, "dump_input_embeds.bin")).reshape(B, S, D)
    mask = load_dump(os.path.join(MODEL_DIR, "dump_mask.bin")).reshape(B, 1, S, S)
    pos = load_dump(os.path.join(MODEL_DIR, "dump_positions.bin"), dtype=np.int32).reshape(B, S)

    # Load ONNX Runtime model
    print("\n[2] Loading ONNX Runtime model...")
    onnx_path = os.path.join(MODEL_DIR, "onnx/llm.onnx")
    if not os.path.exists(onnx_path):
        print(f"  ERROR: {onnx_path} not found!")
        return 1

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Run ONNX Runtime
    print("\n[3] Running ONNX Runtime inference...")
    ort_inputs = {
        'inputs_embeds': input_embeds.astype(np.float32),
        'attention_mask': mask.astype(np.float32),
        'position_ids': pos.astype(np.int64),  # ONNX expects int64
    }
    ort_out = sess.run(None, ort_inputs)
    ort_logits = ort_out[0]
    print(f"  ONNX output: shape={ort_logits.shape}, range=[{ort_logits.min():.4f}, {ort_logits.max():.4f}]")

    # Compare MNN optimized vs NOOPT (sanity check)
    print("\n[4] MNN Optimized vs NOOPT (sanity check):")
    cosim_mnn = cosine_sim(opt_logits, noopt_logits)
    maxdiff_mnn = np.max(np.abs(opt_logits - noopt_logits))
    print(f"  Cosine similarity: {cosim_mnn:.6f}")
    print(f"  Max difference:    {maxdiff_mnn:.6f}")
    print(f"  => {'IDENTICAL ✓' if cosim_mnn > 0.9999 else 'DIFFERENT ✗'}")

    # Compare MNN vs ONNX Runtime
    print("\n[5] MNN (Optimized) vs ONNX Runtime:")
    cosim_full = cosine_sim(opt_logits, ort_logits)
    maxdiff_full = np.max(np.abs(opt_logits - ort_logits))
    print(f"  Full output:")
    print(f"    Cosine similarity: {cosim_full:.6f}")
    print(f"    Max difference:    {maxdiff_full:.6f}")

    # Per-position comparison
    print(f"\n  Per-position cosine similarity:")
    for pos_idx in range(S):
        cosim_pos = cosine_sim(opt_logits[0, pos_idx], ort_logits[0, pos_idx])
        if pos_idx < 5 or pos_idx >= S - 3 or cosim_pos < 0.99:
            marker = " <<< LOW" if cosim_pos < 0.99 else ""
            print(f"    pos {pos_idx:3d}: cosim={cosim_pos:.6f}{marker}")

    # Last position (where argmax happens)
    print(f"\n  Last position (pos {S-1}):")
    cosim_last = cosine_sim(opt_logits[0, S-1], ort_logits[0, S-1])
    maxdiff_last = np.max(np.abs(opt_logits[0, S-1] - ort_logits[0, S-1]))
    print(f"    Cosine similarity: {cosim_last:.6f}")
    print(f"    Max difference:    {maxdiff_last:.6f}")

    # Argmax comparison at last position
    opt_argmax = np.argmax(opt_logits[0, S-1])
    ort_argmax = np.argmax(ort_logits[0, S-1])
    print(f"    Argmax: MNN={opt_argmax}, ONNX={ort_argmax}")
    print(f"    EOS score: MNN={opt_logits[0, S-1, 151645]:.4f}, ONNX={ort_logits[0, S-1, 151645]:.4f}")
    if opt_argmax == ort_argmax:
        print(f"    => MATCH ✓")
    else:
        print(f"    => DIFFERENT ✗")

    # Top-5 tokens
    def top5(logits, label):
        idx = np.argsort(-logits)[:5]
        vals = logits[idx]
        print(f"    {label}: ", end="")
        for i, (v, j) in enumerate(zip(vals, idx)):
            print(f"{j}({v:.4f})", end=" ")
            if i < 4: print("", end="")
        print()

    top5(opt_logits[0, S-1], "MNN top-5")
    top5(ort_logits[0, S-1], "ONNX top-5")

    # Analyze where the difference comes from
    print(f"\n[6] Statistical analysis of differences:")
    diff = opt_logits - ort_logits
    print(f"    Mean diff:    {np.mean(diff):.6f}")
    print(f"    Std diff:     {np.std(diff):.6f}")
    print(f"    Mean abs diff:{np.mean(np.abs(diff)):.6f}")
    print(f"    Diff range:   [{diff.min():.6f}, {diff.max():.6f}]")

    # Check per-layer by creating sub-models?
    # For now, compare distributions
    print(f"\n[7] MNN logit stats per position:")
    for pos_idx in range(min(S, 10)):
        m = opt_logits[0, pos_idx]
        o = ort_logits[0, pos_idx]
        print(f"    pos {pos_idx:2d}: MNN mean={m.mean():.4f} std={m.std():.4f}  "
              f"ONNX mean={o.mean():.4f} std={o.std():.4f}  "
              f"cosim={cosine_sim(m, o):.6f}")

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
