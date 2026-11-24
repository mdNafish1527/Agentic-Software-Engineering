# Agentic Software Engineering, Part 1, Class 2 (15 Nov)

Run (needs `ollama serve` and `ollama pull qwen2.5:7b`):

    ./run_all.sh

- Task 1: `task1_kv_cache/kv_cache.py`, working in `task1_kv_cache/answer.md`
  (8k = 469,762,048 B = 448 MiB; 32k = 1,879,048,192 B = 1.75 GiB)
- Task 2: `src/extract.py` (Method A prompt-only, Method B grammar-constrained, 3 runs each, 300 calls,
  all through `src/ledger.py`), `src/score.py` (invalid rate, field accuracy, cost, spread) -> `results/report.md`
- Schema v1: `schema/notice.schema.json` (version in `x-schema-version`)
- Corpus + hand-written answer key: `corpus/`
- Contract test: `tests/test_contract.py`
