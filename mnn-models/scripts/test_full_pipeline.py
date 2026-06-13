#!/usr/bin/env python3
"""
Full pipeline comparison: MNN vs ONNX Runtime
1. Compute fbank features
2. Run through BOTH audio encoders (MNN + ONNX)
3. Inject audio embeddings + text embeddings
4. Run through BOTH decoders (MNN + ONNX)
5. Compare final outputs
"""
import os
import sys
import numpy as np
import subprocess
import onnxruntime as ort

MODEL_DIR = "/root/projects/mnn-models/Qwen3-ASR-0.6B-MNN"
VOCAB = 151936
HIDDEN = 1024
EOS_TOKEN = 151643
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
    """Load embeddings_bf16.bin as float32."""
    data = np.fromfile(os.path.join(MODEL_DIR, "embeddings_bf16.bin"), dtype=np.uint16)
    # BF16 to FP32: shift left by 16
    as_uint32 = data.astype(np.uint32) << 16
    return as_uint32.view(np.float32).reshape(VOCAB, HIDDEN)

def embed_lookup(tbl, token_ids):
    """Lookup embeddings for token IDs."""
    token_ids = np.array(token_ids)
    token_ids = np.clip(token_ids, 0, VOCAB - 1)
    return tbl[token_ids]  # [S, HIDDEN]

def main():
    print("=" * 60)
    print("Full Pipeline: MNN vs ONNX Runtime")
    print("=" * 60)

    # Step 1: Load audio and compute fbank
    print("\n[1] Loading audio and computing fbank...")
    try:
        import librosa
        audio_path = "/tmp/test_audio_real.pcm"
        audio = np.fromfile(audio_path, dtype=np.float32)
        sr = 16000

        # Compute fbank matching MNN whisper_fbank
        D = librosa.stft(audio, n_fft=400, hop_length=160,
                         win_length=400, window='hann', center=True)
        S = np.abs(D) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=400, n_mels=128,
                                         fmin=0.0, fmax=8000.0, norm=None, htk=True)
        mel = mel_basis @ S
        mel = np.log10(np.maximum(mel, 1e-10))
        feat = mel.astype(np.float32)
        print(f"  Audio: {len(audio)} samples ({len(audio)/sr:.1f}s)")
        print(f"  Fbank: {feat.shape[1]} frames, {feat.shape[0]} mels")
        print(f"  Range: [{feat.min():.4f}, {feat.max():.4f}]")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    # Step 2: Run ONNX audio encoder
    print("\n[2] Running audio encoder...")
    # ONNX Runtime
    ae_onnx = ort.InferenceSession(
        os.path.join(MODEL_DIR, "onnx/audio_encoder.onnx"),
        providers=['CPUExecutionProvider'])
    onnx_ae_out = ae_onnx.run(None, {'input_features': feat.reshape(1, 128, -1).astype(np.float32)})[0]
    T = onnx_ae_out.shape[1]
    print(f"  ONNX AE output: {onnx_ae_out.shape}, range=[{onnx_ae_out.min():.4f}, {onnx_ae_out.max():.4f}]")

    # MNN audio encoder - run via C++ helper
    # Save features for MNN C++ program
    feat.tofile(os.path.join(MODEL_DIR, "tmp_pipeline_feat.bin"))
    with open(os.path.join(MODEL_DIR, "tmp_pipeline_meta.txt"), 'w') as f:
        f.write(f"C=128 T={feat.shape[1]} out_path=dump_pipeline_mnn_ae.bin\n")

    # Build and run MNN C++ program for AE
    print("  Running MNN audio encoder...")
    # Write a simple C++ program that reads features, runs AE, dumps output
    cpp_code = '''
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
int main() {
    std::string dir = "''' + MODEL_DIR + '''";
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);
    auto mod = Module::load({}, {}, (dir + "/audio_encoder.mnn").c_str());
    if (!mod) { std::cerr << "FAIL load\\n"; return 1; }
    std::ifstream fmeta(dir + "/tmp_pipeline_meta.txt");
    int C, T; fmeta >> C >> T;
    std::string dummy, out_path;
    fmeta >> dummy >> out_path;
    std::vector<float> feat(C * T);
    std::ifstream fin(dir + "/tmp_pipeline_feat.bin", std::ios::binary);
    fin.read((char*)feat.data(), C * T * sizeof(float));
    auto inp = _Input({1, C, T}, NCHW, halide_type_of<float>());
    memcpy(inp->writeMap<float>(), feat.data(), C * T * sizeof(float));
    auto out = mod->onForward({inp});
    auto ptr = out[0]->readMap<float>();
    auto sz = out[0]->getInfo()->size;
    std::ofstream fout(dir + "/" + out_path, std::ios::binary);
    fout.write((const char*)ptr, sz * sizeof(float));
    fout.close();
    std::cout << "DONE\\n";
    return 0;
}
'''
    # Compile and run
    src_path = "/root/projects/MNN/transformers/llm/engine/demo/tmp_pipeline_ae.cpp"
    with open(src_path, 'w') as f:
        f.write(cpp_code)

    result = subprocess.run(
        ["g++", "-std=c++11", "-O2", src_path,
         "-I/root/projects/MNN/include",
         "-I/root/projects/MNN/transformers/llm/engine/src",
         "-L/root/projects/MNN/build",
         "-lMNN", "-lMNNExpress", "-lllm",
         "-lpthread", "-lMNNCPU", "-lMNNAudio",
         "-o", "/tmp/run_pipeline_ae",
         "-Wl,-rpath,/root/projects/MNN/build"],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  Compile error: {result.stderr[:500]}")
        return 1

    result = subprocess.run(["/tmp/run_pipeline_ae"], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  Run error: {result.stderr[:500]}")
        return 1
    print(f"  MNN AE: {result.stdout.strip()}")

    mnn_ae_out = np.fromfile(os.path.join(MODEL_DIR, "dump_pipeline_mnn_ae.bin"), dtype=np.float32)
    mnn_ae_out = mnn_ae_out.reshape(1, -1, 1024)
    print(f"  MNN AE output: {mnn_ae_out.shape}, range=[{mnn_ae_out.min():.4f}, {mnn_ae_out.max():.4f}]")

    # Compare AE outputs
    print(f"\n[3] Audio encoder comparison:")
    ae_cosim = cosine_sim(onnx_ae_out, mnn_ae_out)
    ae_maxdiff = np.max(np.abs(onnx_ae_out - mnn_ae_out))
    print(f"  Cosine similarity: {ae_cosim:.6f}")
    print(f"  Max difference:    {ae_maxdiff:.6f}")

    # Step 3: Build decoder input
    print(f"\n[4] Building decoder input...")
    embed_tbl = load_embedding_table()
    min_T = min(onnx_ae_out.shape[1], mnn_ae_out.shape[1])
    T = min_T

    # Build token sequence
    tokens = [AUDIO_START] + [AUDIO_PAD] * T + [AUDIO_END]
    S = len(tokens)
    print(f"  Seq length: {S} ({T} audio frames)")

    # Text embedding
    txt_emb = embed_lookup(embed_tbl, tokens)  # [S, HIDDEN]

    # Build merged embeddings for both AE outputs
    def build_merged(audio_emb):
        # audio_emb: [1, T, HIDDEN]
        merged = txt_emb.copy()
        ai = 0
        for i, tok in enumerate(tokens):
            if tok == AUDIO_PAD and ai < T:
                merged[i] = audio_emb[0, ai]
                ai += 1
        return merged  # [S, HIDDEN]

    merged_onnx = build_merged(onnx_ae_out)
    merged_mnn = build_merged(mnn_ae_out)

    # Causal mask
    mask = np.zeros((1, 1, S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i+1, S):
            mask[0, 0, i, j] = -1e9

    # Position IDs
    pos_ids = np.arange(S, dtype=np.int64).reshape(1, -1)

    # Step 4: Run decoder
    print(f"\n[5] Running decoders...")
    # ONNX decoder with ONNX AE output
    dec_onnx = ort.InferenceSession(
        os.path.join(MODEL_DIR, "onnx/llm.onnx"),
        providers=['CPUExecutionProvider'])

    def run_onnx_decoder(embeds, label):
        logits = dec_onnx.run(None, {
            'inputs_embeds': embeds.reshape(1, S, HIDDEN).astype(np.float32),
            'attention_mask': mask.astype(np.float32),
            'position_ids': pos_ids,
        })[0]
        last = logits[0, S-1]
        first_token = np.argmax(last)
        eos_score = last[151645]
        top5_idx = np.argsort(-last)[:5]
        top5_vals = last[top5_idx]
        print(f"  {label}:")
        print(f"    First token: {first_token}")
        print(f"    EOS score:   {eos_score:.4f}")
        for i in range(5):
            print(f"    top-{i+1}: {top5_idx[i]}({top5_vals[i]:.4f})")
        return logits

    # ONNX AE → ONNX decoder
    onnx_logits = run_onnx_decoder(merged_onnx, "ONNX AE → ONNX Dec")

    # MNN AE → ONNX decoder
    print("")
    mnn_onnx_logits = run_onnx_decoder(merged_mnn, "MNN AE → ONNX Dec")

    # Step 5: Run MNN decoder
    # Use the existing test_compare_models approach
    print(f"\n[6] Running MNN decoder with MNN AE output...")

    # Save merged embeddings for MNN C++ Decoder
    merged_mnn.tofile(os.path.join(MODEL_DIR, "tmp_dec_input.bin"))
    mask.tofile(os.path.join(MODEL_DIR, "tmp_dec_mask.bin"))
    pos_ids.astype(np.int32).tofile(os.path.join(MODEL_DIR, "tmp_dec_pos.bin"))
    with open(os.path.join(MODEL_DIR, "tmp_dec_meta.txt"), 'w') as f:
        f.write(f"B=1 S={S} D={HIDDEN} V={VOCAB}\n")
        f.write(f"out_path=dump_pipeline_mnn_dec.bin\n")

    # Run MNN decoder via existing test_compare_models mechanism
    # Actually, let's reuse test_compare_models by creating input files in the right format
    # and running it with a new mode

    # Write a dedicated C++ program for this
    dec_cpp = '''
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
int main() {
    std::string dir = "''' + MODEL_DIR + '''";
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, MNN::BackendConfig(), 1);
    ExecutorScope scope(executor);
    std::ifstream fmeta(dir + "/tmp_dec_meta.txt");
    int B, S, D, V; fmeta >> B >> S >> D >> V;
    std::string dummy, out_path; fmeta >> dummy >> out_path;
    std::vector<float> emb(B*S*D), msk(B*1*S*S);
    int pmsk = B*S;
    std::vector<int> pos(pmsk);
    std::ifstream f1(dir + "/tmp_dec_input.bin", std::ios::binary);
    f1.read((char*)emb.data(), B*S*D*sizeof(float));
    std::ifstream f2(dir + "/tmp_dec_mask.bin", std::ios::binary);
    f2.read((char*)msk.data(), B*1*S*S*sizeof(float));
    std::ifstream f3(dir + "/tmp_dec_pos.bin", std::ios::binary);
    f3.read((char*)pos.data(), B*S*sizeof(int));
    MNN::ScheduleConfig sched;
    MNN::BackendConfig bc;
    bc.precision = MNN::BackendConfig::Precision_Normal;
    sched.backendConfig = &bc;
    auto rt = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sched));
    rt->setExternalFile(dir + "/llm.mnn.weight");
    Module::Config mc; mc.shapeMutable = true; mc.rearrange = true;
    auto mod = Module::load({}, {}, (dir + "/llm.mnn").c_str(), rt, &mc);
    auto inp = _Input({B, S, D}, NCHW, halide_type_of<float>());
    memcpy(inp->writeMap<float>(), emb.data(), B*S*D*sizeof(float));
    auto msk_v = _Input({B, 1, S, S}, NCHW, halide_type_of<float>());
    memcpy(msk_v->writeMap<float>(), msk.data(), B*1*S*S*sizeof(float));
    auto pos_v = _Input({B, S}, NCHW, halide_type_of<int32_t>());
    memcpy(pos_v->writeMap<int32_t>(), pos.data(), B*S*sizeof(int));
    auto out = mod->onForward({inp, msk_v, pos_v});
    auto ptr = out[0]->readMap<float>();
    auto sz = out[0]->getInfo()->size;
    std::ofstream fout(dir + "/" + out_path, std::ios::binary);
    fout.write((const char*)ptr, sz * sizeof(float));
    fout.close();
    std::cout << "DONE\\n";
    return 0;
}
'''
    dec_src_path = "/root/projects/MNN/transformers/llm/engine/demo/tmp_pipeline_dec.cpp"
    with open(dec_src_path, 'w') as f:
        f.write(dec_cpp)

    result = subprocess.run(
        ["g++", "-std=c++11", "-O2", dec_src_path,
         "-I/root/projects/MNN/include",
         "-I/root/projects/MNN/transformers/llm/engine/src",
         "-L/root/projects/MNN/build",
         "-lMNN", "-lMNNExpress", "-lllm",
         "-lpthread", "-lMNNCPU", "-lMNNAudio",
         "-o", "/tmp/run_pipeline_dec",
         "-Wl,-rpath,/root/projects/MNN/build"],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  Decoder compile error: {result.stderr[:500]}")
        return 1

    result = subprocess.run(["/tmp/run_pipeline_dec"], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Decoder run error: {result.stderr[:500]}")
        return 1

    mnn_dec_logits = np.fromfile(os.path.join(MODEL_DIR, "dump_pipeline_mnn_dec.bin"), dtype=np.float32)
    mnn_dec_logits = mnn_dec_logits.reshape(1, S, V) if mnn_dec_logits.size == S * V else \
                     mnn_dec_logits.reshape(1, -1, V)
    mnn_S = mnn_dec_logits.shape[1]
    mnn_last = mnn_dec_logits[0, mnn_S-1]
    mnn_token = np.argmax(mnn_last)
    mnn_eos = mnn_last[151645]
    top5_idx = np.argsort(-mnn_last)[:5]
    top5_vals = mnn_last[top5_idx]
    print(f"  MNN AE → MNN Dec:")
    print(f"    First token: {mnn_token}")
    print(f"    EOS score:   {mnn_eos:.4f}")
    for i in range(5):
        print(f"    top-{i+1}: {top5_idx[i]}({top5_vals[i]:.4f})")

    # Summary comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    print(f"\n  Golden (ONNX AE → ONNX Dec):")
    last_g = onnx_logits[0, S-1]
    print(f"    First token: {np.argmax(last_g)}, EOS: {last_g[151645]:.4f}")

    print(f"\n  MNN AE → ONNX Dec (isolates AE error):")
    last_mo = mnn_onnx_logits[0, S-1]
    print(f"    First token: {np.argmax(last_mo)}, EOS: {last_mo[151645]:.4f}")
    ae_err_cosim = cosine_sim(last_g, last_mo)
    print(f"    vs Golden cosim: {ae_err_cosim:.6f}")
    print(f"    vs Golden maxdiff: {np.max(np.abs(last_g - last_mo)):.4f}")

    print(f"\n  ONNX AE → MNN Dec (isolates decoder error, if we had it):")
    print(f"    (Decoder tested separately - cosim=1.0 with synthetic data)")

    print(f"\n  MNN AE → MNN Dec (full pipeline):")
    dec_err_cosim = cosine_sim(last_g, mnn_last)
    print(f"    First token: {mnn_token}, EOS: {mnn_eos:.4f}")
    print(f"    vs Golden cosim: {dec_err_cosim:.6f}")
    print(f"    vs Golden maxdiff: {np.max(np.abs(last_g - mnn_last)):.4f}")

    per_frame_cosim = cosine_sim(merged_onnx, merged_mnn)
    print(f"\n  Input embedding cosim: {per_frame_cosim:.6f}")

    # Final verdict
    print(f"\n{'='*60}")
    golden_token = np.argmax(last_g)
    mnn_ae_token = np.argmax(last_mo)
    pipeline_token = mnn_token
    if golden_token == pipeline_token:
        print(f"FULL PIPELINE MATCH: First token {golden_token} ✓")
    else:
        print(f"FULL PIPELINE MISMATCH: Golden={golden_token}, Pipeline={pipeline_token}")
        if golden_token != mnn_ae_token:
            print(f"  → Likely caused by AUDIO ENCODER error (MNN AE gives different results)")
        if golden_token != pipeline_token and golden_token == mnn_ae_token:
            print(f"  → Likely caused by DECODER error")
        elif golden_token != mnn_ae_token:
            print(f"  → Initial cause is audio encoder, potentially AMPLIFIED by decoder")
    print(f"{'='*60}")

    # Cleanup
    print(f"\n[Cleanup]")
    for f in ["tmp_pipeline_feat.bin", "tmp_pipeline_meta.txt", "tmp_pipeline_ae.cpp",
              "tmp_pipeline_dec.cpp", "tmp_dec_input.bin", "tmp_dec_mask.bin",
              "tmp_dec_pos.bin", "tmp_dec_meta.txt", "dump_pipeline_mnn_ae.bin",
              "dump_pipeline_mnn_dec.bin"]:
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path): os.remove(path)

    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
