"""KV-cache size for Qwen2.5-7B (values from the model's config.json).
size = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_elem * tokens
Usage: python kv_cache.py [--key-8k BYTES] [--key-32k BYTES]
"""
import argparse

LAYERS = 28          # num_hidden_layers
HIDDEN = 3584        # hidden_size
HEADS = 28           # num_attention_heads
KV_HEADS = 4         # num_key_value_heads (GQA)
HEAD_DIM = HIDDEN // HEADS  # 128
BYTES = 2            # fp16 / bf16

per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES  # 57,344 B


def size(tokens):
    return per_token * tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-8k", type=float)
    ap.add_argument("--key-32k", type=float)
    a = ap.parse_args()
    print(f"per token = 2*{LAYERS}*{KV_HEADS}*{HEAD_DIM}*{BYTES} = {per_token} B ({per_token/1024:.0f} KiB)")
    for name, t, key in (("8k", 8192, a.key_8k), ("32k", 32768, a.key_32k)):
        s = size(t)
        print(f"{name} ({t} tokens): {s} B = {s/2**20:.0f} MiB = {s/2**30:.4f} GiB")
        if key:
            err = abs(s - key) / key * 100
            print(f"   vs key {key:.0f} B: {err:.2f}% -> {'PASS' if err <= 10 else 'FAIL'} (<=10%)")


if __name__ == "__main__":
    main()
