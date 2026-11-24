# Task 1: KV-cache size, Qwen2.5-7B (fp16/bf16 cache)

Config: 28 layers, 28 query heads, 4 KV heads (GQA), head_dim = 3584/28 = 128, 2 bytes/element.

Per token = 2 (K,V) x 28 x 4 x 128 x 2 B = 57,344 B = 56 KiB

| Context | Tokens | Bytes | Size |
|---|---|---|---|
| 8k  | 8,192  | 469,762,048   | 448 MiB (0.4375 GiB) |
| 32k | 32,768 | 1,879,048,192 | 1,792 MiB (1.75 GiB) |

If 8k = 8,000 and 32k = 32,000 tokens: 458.75 MB and 1,835 MB (within 2.4% of the above).

Comparison with the answer key: `python kv_cache.py --key-8k <bytes> --key-32k <bytes>` (pass if both within 10%).
