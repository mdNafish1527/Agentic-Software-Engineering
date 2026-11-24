#!/bin/sh
# Requires: ollama serve; ollama pull qwen2.5:7b
set -e
python3 task1_kv_cache/kv_cache.py
python3 src/gen_corpus.py          # skip if using the class corpus
python3 src/extract.py --model qwen2.5:7b --runs 3
python3 src/score.py
python3 -m unittest discover -s tests -v
