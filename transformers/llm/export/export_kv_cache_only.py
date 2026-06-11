#!/usr/bin/env python3
"""
Minimal script: Export ONLY the KV-cache decoder for Qwen3-ASR.
Produces llm_kv.mnn (with embedded INT8 weights, no external weight file needed).
This is the model format that the Old Engine (Qwen3AsrEngine) expects.

Usage:
    python export_kv_cache_only.py --model_path ../Qwen3-ASR-0.6B --dst_path ./output_kv
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
from safetensors import safe_open

# Reuse the model definitions from export_qwen3_asr.py
sys.path.insert(0, os.path.dirname(__file__))
from export_qwen3_asr import Qwen3Decoder, Qwen3DecoderWithKVCache
from MNN.tools.mnnconvert import convert as mnn_convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to Qwen3-ASR-0.6B HF model')
    parser.add_argument('--dst_path', required=True, help='Output directory')
    parser.add_argument('--quant_bit', type=int, default=8)
    parser.add_argument('--quant_block', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.dst_path, exist_ok=True)

    # Load weights
    safetensors_path = os.path.join(args.model_path, 'model.safetensors')
    state_dict = {}
    with safe_open(safetensors_path, framework='pt') as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    print(f"Loaded {len(state_dict)} weight tensors")

    # Read decoder config
    config_path = os.path.join(args.model_path, 'config.json')
    with open(config_path) as f:
        full_config = json.load(f)
    decoder_config = full_config.get('thinker_config', {}).get('text_config', {})
    decoder_config['model_type'] = 'qwen3_asr'
    decoder_config.setdefault('num_hidden_layers', 28)
    decoder_config.setdefault('vocab_size', 151936)

    # Build decoder
    decoder = Qwen3Decoder(decoder_config)
    decoder.load_weights(state_dict)
    kv_decoder = Qwen3DecoderWithKVCache(decoder)
    kv_decoder.eval()
    kv_decoder.float()

    # ONNX export
    onnx_path = os.path.join(args.dst_path, 'llm_kv.onnx')
    B, S, D = 1, 16, 1024
    L, Hk, Hd = 28, 8, 128
    past_len = 1

    dummy_embeds = torch.randn(B, S, D)
    dummy_pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
    dummy_mask = torch.zeros(B, 1, S, past_len + S).float()
    dummy_k = torch.randn(L, B, Hk, past_len, Hd)
    dummy_v = torch.randn(L, B, Hk, past_len, Hd)

    print("Exporting ONNX...")
    try:
        dynamic_shapes = [
            {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
            {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
            {0: torch.export.Dim("batch"), 2: torch.export.Dim("seq_len"), 3: torch.export.Dim("total_len")},
            {3: torch.export.Dim("cache_len")},
            {3: torch.export.Dim("cache_len")},
        ]
        torch.onnx.export(kv_decoder,
            (dummy_embeds, dummy_pos, dummy_mask, dummy_k, dummy_v),
            onnx_path,
            input_names=['inputs_embeds', 'position_ids', 'attention_mask', 'k_cache', 'v_cache'],
            output_names=['logits', 'k_cache_out', 'v_cache_out'],
            dynamic_shapes=dynamic_shapes,
            opset_version=18, do_constant_folding=True)
    except Exception as e:
        print(f"  torch.export dynamic_shapes failed ({e}), falling back to dynamo=False...")
        torch.onnx.export(kv_decoder,
            (dummy_embeds, dummy_pos, dummy_mask, dummy_k, dummy_v),
            onnx_path,
            input_names=['inputs_embeds', 'position_ids', 'attention_mask', 'k_cache', 'v_cache'],
            output_names=['logits', 'k_cache_out', 'v_cache_out'],
            dynamic_axes={
                'inputs_embeds': {0: 'batch', 1: 'seq_len'},
                'position_ids': {0: 'batch', 1: 'seq_len'},
                'attention_mask': {0: 'batch', 2: 'seq_len', 3: 'total_len'},
                'k_cache': {3: 'cache_len'}, 'v_cache': {3: 'cache_len'},
                'logits': {0: 'batch', 1: 'seq_len'},
                'k_cache_out': {3: 'new_len'}, 'v_cache_out': {3: 'new_len'},
            },
            opset_version=18, do_constant_folding=True, dynamo=False)
    print(f"  ONNX saved: {onnx_path}")

    # Convert to MNN
    mnn_path = os.path.join(args.dst_path, 'llm_kv.mnn')
    quant = ''
    if args.quant_bit < 16:
        quant = f'--weightQuantBits {args.quant_bit} --weightQuantBlock {args.quant_block}'
    print(f"Converting to MNN (INT8, embedded weights)...")
    mnn_convert(f'-f ONNX --modelFile "{onnx_path}" --MNNModel "{mnn_path}" {quant}')
    if os.path.exists(mnn_path):
        size_mb = os.path.getsize(mnn_path) / (1024*1024)
        print(f"  MNN saved: {mnn_path} ({size_mb:.1f} MB)")
    else:
        print("  ERROR: MNN conversion failed")
        return 1

    print("\nDone! Copy llm_kv.mnn to the model directory on device.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
