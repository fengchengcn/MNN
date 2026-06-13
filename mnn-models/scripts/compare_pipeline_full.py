#!/usr/bin/env python3
"""
Full pipeline comparison with correct text prompt.
Includes: text prompt + audio tokens → decoder → compare MNN vs ONNX
"""
import os
import sys
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"
VOCAB = 151936
HIDDEN = 1024
# Token IDs
AUDIO_START = 151669
AUDIO_END   = 151670
AUDIO_PAD   = 151676
IM_START    = 151644  # <|im_start|>
IM_END      = 151645  # <|im_end|>
# Prompt: "<|im_start|>user\nTranscribe:\n<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TOKENS = [151644, 872, 198, 3167, 3114, 510, 151645, 198, 151644, 77091, 198]

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb + 1e-8))

def load_embedding_table():
    data = np.fromfile(os.path.join(MODEL_DIR, "embeddings_bf16.bin"), dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32).reshape(VOCAB, HIDDEN)

def build_pipeline_inputs(audio_emb, embed_tbl):
    """Build full decoder inputs: prompt + audio embeddings + causal mask + pos_ids."""
    T = audio_emb.shape[1]
    # Full token sequence: prompt tokens + audio_start + audio_pad*T + audio_end
    tokens = list(PROMPT_TOKENS) + [AUDIO_START] + [AUDIO_PAD] * T + [AUDIO_END]
    S = len(tokens)

    # Text embeddings for prompt and special tokens
    txt_emb = embed_tbl[np.array(tokens)]  # [S, HIDDEN]

    # Inject audio embeddings at pad positions
    merged = txt_emb.copy()
    ai = 0
    prompt_len = len(PROMPT_TOKENS)
    for i in range(prompt_len, S):
        if tokens[i] == AUDIO_PAD and ai < T:
            merged[i] = audio_emb[0, ai]
            ai += 1

    # Causal mask
    mask = np.zeros((1, 1, S, S), dtype=np.float32)
    for i in range(S):
        mask[0, 0, i, i+1:] = -1e9

    # Position IDs
    pos_ids = np.arange(S, dtype=np.int64).reshape(1, -1)

    return merged, mask, pos_ids, tokens, S

def main():
    print("=" * 60)
    print("Full Pipeline Comparison (with text prompt)")
    print("=" * 60)

    # Load AE outputs
    print("\n[1] Loading audio encoder outputs...")
    onnx_ae = np.fromfile(os.path.join(MODEL_DIR, "dump_real_ort_ae.bin"), dtype=np.float32) \
                .reshape(1, -1, 1024)
    mnn_ae = np.fromfile(os.path.join(MODEL_DIR, "mnn_ae_output.bin"), dtype=np.float32) \
               .reshape(1, -1, 1024)
    min_T = min(onnx_ae.shape[1], mnn_ae.shape[1])
    onnx_ae, mnn_ae = onnx_ae[:, :min_T], mnn_ae[:, :min_T]
    T = min_T
    print(f"  ONNX AE: {onnx_ae.shape}  MNN AE: {mnn_ae.shape}")
    print(f"  AE cosim={cosine_sim(onnx_ae, mnn_ae):.6f}")

    # Build inputs
    print(f"\n[2] Building decoder inputs ({T} audio frames)...")
    embed_tbl = load_embedding_table()
    merged_onnx, mask, pos_ids, tokens, S = build_pipeline_inputs(onnx_ae, embed_tbl)
    merged_mnn, _, _, _, _ = build_pipeline_inputs(mnn_ae, embed_tbl)
    print(f"  Sequence: {S} tokens (prompt={len(PROMPT_TOKENS)}, audio={T}, special=2)")
    print(f"  Tokens: {tokens[:8]}...{tokens[-3:]}")

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
    mnn_test = run_dec(merged_mnn, "MNN AE → ONNX Dec (AE error only)")

    # Compare
    print(f"\n[4] === FINAL COMPARISON ===")
    g_last = golden[0, S-1]
    m_last = mnn_test[0, S-1]
    cosim = cosine_sim(g_last, m_last)
    maxdiff = np.max(np.abs(g_last - m_last))
    g_tok, m_tok = np.argmax(g_last), np.argmax(m_last)

    print(f"  Logits cosim: {cosim:.6f}")
    print(f"  Max diff: {maxdiff:.6f}")
    print(f"  Golden first token: {g_tok}")
    print(f"  MNN AE first token: {m_tok}")

    if g_tok == m_tok:
        print(f"  => ✅ FULL PIPELINE MATCHES!")
    else:
        print(f"  => ❌ DIFFERENT FIRST TOKEN")
        g_top5 = set(np.argsort(-g_last)[:5])
        if m_tok in g_top5:
            print(f"  (MNN's token in Golden's top-5)")
        else:
            print(f"  (MNN's token NOT in Golden's top-5)")

    # Per-position
    print(f"\n[5] Per-position:")
    for pos in range(S):
        c = cosine_sim(golden[0, pos], mnn_test[0, pos])
        if c < 0.999 or pos < 3 or pos >= S-2:
            marker = " <<<" if c < 0.999 else ""
            print(f"  pos {pos:3d}: cosim={c:.6f}{marker}")

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
