#!/usr/bin/env python3
"""Tokenizer roundtrip verification: MNN tokenizer vs Qwen BPE reference."""
import sys, os, io

# Fix Windows GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TEST_TEXT = [
    ("common", "你好，今天天气怎么样？"),
    ("rare_char", "龘靐齉"),
    ("rare_word", "旮旯孑孓"),
    ("mixed", "这篇文章探讨了龘字的演变历史。"),
    ("tech", "傅里叶变换将时域信号转换到频域。"),
    ("short", "打开音乐。"),
]

def inspect_mnn_tokenizer(path, label):
    """Inspect MNN tokenizer.txt format."""
    print(f"\n{'='*60}")
    print(f"MNN tokenizer: {label}")
    print(f"  Path: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read(5000)  # first 5KB

    size = os.path.getsize(path)
    lines = content.split('\n')
    print(f"  Size: {size:,} bytes, sample {len(lines)} lines")

    # Show header lines
    for i in range(min(6, len(lines))):
        line = lines[i].strip()
        if len(line) > 120:
            line = line[:120] + "..."
        print(f"  Line {i}: {line}")

    # Check what's in the middle
    with open(path, 'rb') as f:
        f.seek(size // 2)
        mid = f.read(200).decode('utf-8', errors='replace')
    print(f"  Mid-content: {mid[:150]}...")

    # Check for rare character presence
    with open(path, 'r', encoding='utf-8') as f:
        full = f.read()
    for c in "龘靐旮旯":
        count = full.count(c)
        print(f"  Char '{c}' occurrences: {count}")

    return content


def compare_with_hf():
    """Load HF Qwen tokenizer and do roundtrip test."""
    print(f"\n{'='*60}")
    print("HuggingFace Qwen2 Tokenizer Roundtrip Test")
    print(f"{'='*60}")

    try:
        from transformers import AutoTokenizer
        # Qwen2 tokenizer — same architecture as Qwen3-ASR
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B", trust_remote_code=True,
            cache_dir=os.path.expanduser("~/.cache/huggingface")
        )
    except Exception as e:
        print(f"Cannot load HF model: {e}")
        print("Trying local tiktoken fallback...")
        try:
            import tiktoken
            tok = tiktoken.get_encoding("cl100k_base")
            print("Using cl100k_base as rough reference (NOT Qwen vocab)")
        except:
            print("No tokenizer available for comparison")
            return

    print(f"Tokenizer: {tok}")
    vs = getattr(tok, 'vocab_size', getattr(tok, 'n_vocab', '?'))
    print(f"Vocab size: {vs}")

    for name, text in TEST_TEXT:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        ok = "✓" if text == decoded else "✗ MISMATCH"
        print(f"  [{name}] '{text}' → {len(ids)} tokens → '{decoded}' {ok}")
        if text != decoded:
            print(f"         IDs: {ids}")


def main():
    omni = r"D:\mojing\MNN\mnn-models\Qwen3-ASR-MNN-FP16\tokenizer.txt"
    old = r"D:\mojing\MNN\mnn-models\Qwen3-ASR-sherpa-onnx-old-engine\tokenizer.txt"

    inspect_mnn_tokenizer(omni, "Omni FP16")
    inspect_mnn_tokenizer(old, "Old-Engine")

    compare_with_hf()

    print(f"\n{'='*60}")
    print("Conclusion: Compare rare character token counts above.")
    print("If MNN tokenizer has fewer rare char occurrences than expected,")
    print("BPE merge rules may be incomplete → wrong tokenization → poor ASR.")

if __name__ == "__main__":
    main()
