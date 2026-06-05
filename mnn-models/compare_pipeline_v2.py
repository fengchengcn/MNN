#!/usr/bin/env python3
"""
Compare MNN vs ONNX Runtime - full pipeline with CORRECT prompt format.
1. Compute fbank features from audio
2. Run through MNN audio encoder
3. Build merged embeddings with correct prompt format
4. Run through decoder (ONNX)
5. Compare with MNN full pipeline output
"""
import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"
VOCAB = 151936
HIDDEN = 1024

# Load tokenizer for decoding
tok = AutoTokenizer.from_pretrained(f'{MODEL_DIR}', trust_remote_code=True)

# Prompt tokens from chat template (before audio pad expansion):
# <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n
PROMPT_PREFIX = [151644, 8948, 198, 151645, 198, 151644, 872, 198]  # before audio_start
PROMPT_SUFFIX = [151670, 151645, 198, 151644, 77091, 198]  # after audio_end

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def load_embedding_table():
    data = np.fromfile(os.path.join(MODEL_DIR, "embeddings_bf16.bin"), dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32).reshape(VOCAB, HIDDEN)

def build_inputs(audio_emb, embed_tbl):
    """Build merged embeddings + mask + pos_ids."""
    T = audio_emb.shape[1]
    tokens = PROMPT_PREFIX + [151669] + [151676] * T + PROMPT_SUFFIX
    S = len(tokens)

    # Text embeddings
    txt_emb = embed_tbl[np.array(tokens)]
    merged = txt_emb.copy()

    # Inject audio embeddings at pad positions
    ai = 0
    for i, tok in enumerate(tokens):
        if tok == 151676 and ai < T:
            merged[i] = audio_emb[0, ai]
            ai += 1

    # Causal mask
    mask = np.zeros((1, 1, S, S), dtype=np.float32)
    for i in range(S):
        mask[0, 0, i, i+1:] = -1e9

    pos_ids = np.arange(S, dtype=np.int64).reshape(1, -1)
    return merged, mask, pos_ids, tokens, S

def run_mnn_ae(feat, C, T):
    """Run MNN audio encoder via C++ helper."""
    feat_path = os.path.join(MODEL_DIR, "v2_feat.bin")
    feat.tofile(feat_path)

    code = f'''
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
using namespace MNN::Express;
int main() {{
    std::string dir = "{MODEL_DIR}";
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);
    auto mod = Module::load({{}}, {{}}, (dir + "/audio_encoder.mnn").c_str());
    if (!mod) return 1;
    int C={C}, T={T};
    std::vector<float> feat(C*T);
    std::ifstream fin(dir + "/v2_feat.bin", std::ios::binary);
    fin.read((char*)feat.data(), C*T*sizeof(float));
    auto inp = _Input({{1, C, T}}, NCHW, halide_type_of<float>());
    memcpy(inp->writeMap<float>(), feat.data(), C*T*sizeof(float));
    auto out = mod->onForward({{inp}});
    auto ptr = out[0]->readMap<float>();
    auto sz = out[0]->getInfo()->size;
    std::ofstream fout(dir + "/v2_mnn_ae.bin", std::ios::binary);
    fout.write((const char*)ptr, sz*sizeof(float));
    fout.close();
    std::cout << "MNN_AE_DONE shape=";
    for (auto d : out[0]->getInfo()->dim) std::cout << d << ",";
    std::cout << std::endl;
    return 0;
}}'''
    src = os.path.join(MODEL_DIR, "v2_ae_helper.cpp")
    with open(src, 'w') as f:
        f.write(code)

    r = subprocess.run([
        "g++", "-std=c++11", "-O2", src,
        "-I/root/projects/MNN/include",
        "-I/root/projects/MNN/transformers/llm/engine/src",
        "/root/projects/MNN/build/libMNN.a",
        "-lpthread", "-lz", "-fopenmp",
        "-o", "/tmp/v2_ae_helper"
    ], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"Compile error: {r.stderr[:200]}")
        return None

    r = subprocess.run(["/tmp/v2_ae_helper"], capture_output=True, text=True, timeout=120)
    print(f"  MNN AE: {r.stdout.strip()}")
    if r.returncode != 0:
        print(f"  Run error: {r.stderr[:200]}")
        return None

    out = np.fromfile(os.path.join(MODEL_DIR, "v2_mnn_ae.bin"), dtype=np.float32)
    return out

def run_mnn_dec(embeds, mask, pos_ids, B, S, D):
    """Run MNN decoder via C++ helper."""
    embeds.tofile(os.path.join(MODEL_DIR, "v2_dec_in.bin"))
    mask.tofile(os.path.join(MODEL_DIR, "v2_dec_mask.bin"))
    pos_ids.astype(np.int32).tofile(os.path.join(MODEL_DIR, "v2_dec_pos.bin"))

    code = f'''
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
using namespace MNN::Express;
int main() {{
    std::string dir = "{MODEL_DIR}";
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);
    int B={B}, S={S}, D={D}, V={VOCAB};
    std::vector<float> emb(B*S*D), msk(B*1*S*S);
    std::vector<int> pos(B*S);
    std::ifstream(dir+"/v2_dec_in.bin", std::ios::binary).read((char*)emb.data(), B*S*D*sizeof(float));
    std::ifstream(dir+"/v2_dec_mask.bin", std::ios::binary).read((char*)msk.data(), B*1*S*S*sizeof(float));
    std::ifstream(dir+"/v2_dec_pos.bin", std::ios::binary).read((char*)pos.data(), B*S*sizeof(int));
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc; bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(dir + "/llm.mnn.weight");
    Module::Config mc; mc.shapeMutable = true; mc.rearrange = true;
    auto mod = Module::load({{}}, {{}}, (dir+"/llm.mnn").c_str(), rt, &mc);
    auto inp_v = _Input({{B,S,D}}, NCHW, halide_type_of<float>());
    memcpy(inp_v->writeMap<float>(), emb.data(), B*S*D*sizeof(float));
    auto msk_v = _Input({{B,1,S,S}}, NCHW, halide_type_of<float>());
    memcpy(msk_v->writeMap<float>(), msk.data(), B*1*S*S*sizeof(float));
    auto pos_v = _Input({{B,S}}, NCHW, halide_type_of<int32_t>());
    memcpy(pos_v->writeMap<int32_t>(), pos.data(), B*S*sizeof(int));
    auto out = mod->onForward({{inp_v, msk_v, pos_v}});
    auto ptr = out[0]->readMap<float>();
    auto sz = out[0]->getInfo()->size;
    std::ofstream(dir+"/v2_mnn_dec.bin", std::ios::binary).write((const char*)ptr, sz*sizeof(float));
    std::cout << "MNN_DEC_DONE size=" << sz << std::endl;
    return 0;
}}'''
    src = os.path.join(MODEL_DIR, "v2_dec_helper.cpp")
    with open(src, 'w') as f:
        f.write(code)

    r = subprocess.run([
        "g++", "-std=c++11", "-O2", src,
        "-I/root/projects/MNN/include",
        "-I/root/projects/MNN/transformers/llm/engine/src",
        "/root/projects/MNN/build/libMNN.a",
        "-lpthread", "-lz", "-fopenmp",
        "-o", "/tmp/v2_dec_helper"
    ], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"Dec compile error: {r.stderr[:200]}")
        return None

    r = subprocess.run(["/tmp/v2_dec_helper"], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"Dec run error: {r.stderr[:200]}")
        return None

    out = np.fromfile(os.path.join(MODEL_DIR, "v2_mnn_dec.bin"), dtype=np.float32)
    out = out.reshape(1, S, V)
    return out

def main():
    print("=" * 60)
    print("MNN vs ONNX - Full Pipeline v2 (correct prompt)")
    print("=" * 60)

    # Use downloaded speech sample
    audio_path = "/tmp/speech_sample.flac"
    if not os.path.exists(audio_path):
        audio_path = "/tmp/test_audio.wav"

    print(f"\n[1] Processing audio: {audio_path}")
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"  Audio: {len(audio)} samples @ {sr}Hz = {len(audio)/sr:.1f}s")

    # Compute fbank (same as MNN whisper_fbank params)
    D = librosa.stft(audio, n_fft=400, hop_length=160, win_length=400, window='hann', center=True)
    S_spec = np.abs(D) ** 2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=400, n_mels=128, fmin=0.0, fmax=8000.0, norm=None, htk=True)
    mel = mel_basis @ S_spec
    mel = np.log10(np.maximum(mel, 1e-10))
    feat = mel.astype(np.float32)
    C, T = feat.shape
    print(f"  Fbank: {T} frames")

    # Run ONNX AE
    print(f"\n[2] Audio encoders...")
    ae_onnx = ort.InferenceSession(
        os.path.join(MODEL_DIR, "onnx/audio_encoder.onnx"),
        providers=['CPUExecutionProvider'])
    onnx_ae = ae_onnx.run(None, {'input_features': feat.reshape(1, C, T).astype(np.float32)})[0]
    print(f"  ONNX AE: {onnx_ae.shape}")

    # Run MNN AE
    mnn_ae_flat = run_mnn_ae(feat, C, T)
    if mnn_ae_flat is None: return 1
    mnn_ae = mnn_ae_flat.reshape(1, -1, 1024)
    print(f"  MNN AE:  {mnn_ae.shape}")

    min_T = min(onnx_ae.shape[1], mnn_ae.shape[1])
    onnx_ae, mnn_ae = onnx_ae[:, :min_T], mnn_ae[:, :min_T]
    ae_cosim = cosine_sim(onnx_ae, mnn_ae)
    print(f"  AE cosim: {ae_cosim:.6f}")

    # Run ONNX Decoder with both AEs
    print(f"\n[3] Decoders...")
    embed_tbl = load_embedding_table()

    def run_onnx_dec(audio_emb, label):
        merged, mask, pos_ids, _, S = build_inputs(audio_emb, embed_tbl)
        print(f"  {label}: S={S}")
        sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "onnx/llm.onnx"),
            providers=['CPUExecutionProvider'])
        logits = sess.run(None, {
            'inputs_embeds': merged.reshape(1, S, HIDDEN).astype(np.float32),
            'attention_mask': mask.astype(np.float32),
            'position_ids': pos_ids,
        })[0]
        last = logits[0, S-1]
        first_token = int(np.argmax(last))
        tokens = [first_token]
        return logits, tokens, S

    onnx_logits, onnx_tokens, S = run_onnx_dec(onnx_ae, "ONNX AE → ONNX Dec")
    mnn_ae_logits, mnn_ae_tokens, _ = run_onnx_dec(mnn_ae, "MNN AE → ONNX Dec")

    # Run MNN Decoder with MNN AE
    merged_mnn, mask_mnn, pos_ids_mnn, _, _ = build_inputs(mnn_ae, embed_tbl)
    mnn_dec_logits = run_mnn_dec(merged_mnn, mask_mnn, pos_ids_mnn, 1, S, HIDDEN)
    if mnn_dec_logits is None:
        mnn_first = "FAILED"
    else:
        mnn_last = mnn_dec_logits[0, S-1]
        mnn_first = int(np.argmax(mnn_last))

    print(f"\n  MNN AE → MNN Dec: first token = {mnn_first}")

    # Compare
    g_last = onnx_logits[0, S-1]
    m_ae_last = mnn_ae_logits[0, S-1]
    golden_tok = int(np.argmax(g_last))
    ae_err_tok = int(np.argmax(m_ae_last))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  ONNX AE → ONNX Dec: first_token={golden_tok} ({tok.decode([golden_tok])})")
    print(f"  MNN AE → ONNX Dec:  first_token={ae_err_tok} ({tok.decode([ae_err_tok])})")

    cosim_ae = cosine_sim(g_last, m_ae_last)
    print(f"  AE-error logits cosim: {cosim_ae:.6f}")
    print(f"  AE-error logits maxdiff: {np.max(np.abs(g_last - m_ae_last)):.6f}")

    if isinstance(mnn_first, int):
        print(f"  MNN AE → MNN Dec:    first_token={mnn_first} ({tok.decode([mnn_first])})")
        cosim_full = cosine_sim(g_last, mnn_dec_logits[0, S-1])
        print(f"  Full pipeline logits cosim: {cosim_full:.6f}")

    if golden_tok == ae_err_tok == mnn_first:
        print(f"\n✅ ALL THREE PIPELINES MATCH!")
    elif golden_tok == ae_err_tok:
        print(f"\n⚠️ MNN decoder differs from ONNX decoder (but AE matches)")
    elif golden_tok == mnn_first:
        print(f"\n⚠️ MNN AE differs (but full MNN pipeline matches golden)")
    else:
        print(f"\n❌ Both AE and decoder contribute to error")

    # Cleanup
    for f in ["v2_feat.bin", "v2_mnn_ae.bin", "v2_mnn_dec.bin",
              "v2_ae_helper.cpp", "v2_dec_helper.cpp",
              "v2_dec_in.bin", "v2_dec_mask.bin", "v2_dec_pos.bin"]:
        p = os.path.join(MODEL_DIR, f)
        if os.path.exists(p): os.remove(p)

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
