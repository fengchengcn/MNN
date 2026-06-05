#!/usr/bin/env python3
"""
Compare full pipelines: ONNX AE→ONNX Dec vs MNN AE→ONNX Dec
This isolates the audio encoder contribution to the error.
"""
import os
import sys
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"
VOCAB = 151936
HIDDEN = 1024
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb + 1e-8))

def load_embedding_table():
    data = np.fromfile(os.path.join(MODEL_DIR, "embeddings_bf16.bin"), dtype=np.uint16)
    as_uint32 = data.astype(np.uint32) << 16
    return as_uint32.view(np.float32).reshape(VOCAB, HIDDEN)

def build_merged(tokens, audio_emb, embed_tbl):
    """Build merged embeddings with audio injection."""
    T = audio_emb.shape[1]
    txt_emb = embed_tbl[np.array(tokens)]  # [S, HIDDEN]
    merged = txt_emb.copy()
    ai = 0
    for i, tok in enumerate(tokens):
        if tok == AUDIO_PAD and ai < T:
            merged[i] = audio_emb[0, ai]
            ai += 1
    return merged  # [S, HIDDEN]

def main():
    print("=" * 60)
    print("Pipeline Comparison: Audio Encoder Error Isolation")
    print("=" * 60)

    # Load AE outputs
    print("\n[1] Loading audio encoder outputs...")
    onnx_ae = np.fromfile(os.path.join(MODEL_DIR, "dump_real_ort_ae.bin"), dtype=np.float32)
    onnx_ae = onnx_ae.reshape(1, -1, 1024)
    T = onnx_ae.shape[1]
    print(f"  ONNX AE: {onnx_ae.shape}  range=[{onnx_ae.min():.4f}, {onnx_ae.max():.4f}]")

    mnn_ae = np.fromfile(os.path.join(MODEL_DIR, "mnn_ae_output.bin"), dtype=np.float32)
    mnn_ae = mnn_ae.reshape(1, -1, 1024)
    min_T = min(onnx_ae.shape[1], mnn_ae.shape[1])
    onnx_ae = onnx_ae[:, :min_T]
    mnn_ae = mnn_ae[:, :min_T]
    T = min_T
    print(f"  MNN AE:  {mnn_ae.shape}  range=[{mnn_ae.min():.4f}, {mnn_ae.max():.4f}]")

    # Compare AE outputs
    ae_cosim = cosine_sim(onnx_ae, mnn_ae)
    ae_maxdiff = np.max(np.abs(onnx_ae - mnn_ae))
    print(f"\n  AE cosim={ae_cosim:.6f} maxdiff={ae_maxdiff:.6f}")

    # Build token sequence
    print(f"\n[2] Building decoder inputs ({T} audio frames)...")
    embed_tbl = load_embedding_table()
    tokens = [AUDIO_START] + [AUDIO_PAD] * T + [AUDIO_END]
    S = len(tokens)
    print(f"  Sequence length: {S}")

    merged_onnx = build_merged(tokens, onnx_ae, embed_tbl)
    merged_mnn = build_merged(tokens, mnn_ae, embed_tbl)

    # Causal mask
    mask = np.zeros((1, 1, S, S), dtype=np.float32)
    for i in range(S):
        mask[0, 0, i, i+1:] = -1e9
    pos_ids = np.arange(S, dtype=np.int64).reshape(1, -1)

    # Run ONNX decoder
    print(f"\n[3] Running ONNX decoder...")
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, "onnx/llm.onnx"),
        providers=['CPUExecutionProvider'])

    def run_dec(embeds, label):
        logits = sess.run(None, {
            'inputs_embeds': embeds.reshape(1, S, HIDDEN).astype(np.float32),
            'attention_mask': mask.astype(np.float32),
            'position_ids': pos_ids,
        })[0]
        last = logits[0, S-1]
        tok = np.argmax(last)
        eos = last[151645]
        top5_i = np.argsort(-last)[:5]
        top5_v = last[top5_i]
        print(f"  {label}:")
        print(f"    First token: {tok}  EOS score: {eos:.4f}")
        for i in range(5):
            print(f"    top-{i+1}: {top5_i[i]}({top5_v[i]:.4f})")
        return logits

    golden = run_dec(merged_onnx, "ONNX AE → ONNX Dec (GOLDEN)")
    mnn_ae_test = run_dec(merged_mnn, "MNN AE → ONNX Dec (AE error only)")

    # Compare
    print(f"\n[4] Is audio encoder the cause?")
    g_last = golden[0, S-1]
    m_last = mnn_ae_test[0, S-1]
    cosim = cosine_sim(g_last, m_last)
    maxdiff = np.max(np.abs(g_last - m_last))
    g_tok = np.argmax(g_last)
    m_tok = np.argmax(m_last)
    print(f"  Logits cosim: {cosim:.6f}")
    print(f"  Logits maxdiff: {maxdiff:.6f}")
    print(f"  Golden first token: {g_tok}")
    print(f"  MNN-AE first token: {m_tok}")
    if g_tok == m_tok:
        print(f"  => AUDIO ENCODER IS FINE ✓")
    else:
        print(f"  => AUDIO ENCODER CHANGES THE FIRST TOKEN ✗")
        print(f"  This means the 0.999-cosim AE error IS enough to change the prediction")

    # Check if token difference is a "near miss" (top-5)
    g_top5 = set(np.argsort(-g_last)[:5])
    if m_tok in g_top5:
        print(f"  (MNN-AE's token is in Golden's top-5 - close call)")
    else:
        print(f"  (MNN-AE's token NOT in Golden's top-5 - significant difference)")

    # Also compare the 28 individual decoder layer outputs
    # We can't easily get per-layer from ONNX, but we can compare
    # ALL S positions
    print(f"\n[5] Per-position comparison (all S={S} positions):")
    for pos in range(S):
        c = cosine_sim(golden[0, pos], mnn_ae_test[0, pos])
        if c < 0.999 or pos < 3 or pos >= S-2:
            marker = " <<<" if c < 0.999 else ""
            print(f"  pos {pos:3d}: cosim={c:.6f}{marker}")

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
