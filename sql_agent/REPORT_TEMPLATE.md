# Assignment 2: English vs Bangla Token Cost

**Student:** [name]    **Model:** qwen2.5:3b via Ollama (temperature 0, seed 42, max output 256)    **Date:** [date]

## 1. Dataset
50 aligned pairs in six categories: question (10), instruction (10), explanation (10), arithmetic (8), summary (6), long 3-5 sentence prompts (6). The Bangla was written by the author [and checked by: name, if applicable]. Pair lengths range from 4 to 43 English words.

## 2. Method
- Token counts come from Ollama's `prompt_eval_count` and `eval_count`, read on every call.
- Fixed system prompt for all calls. Template overhead measured with `probe`: [overhead_tokens] tokens per call.
- Prompt cache: [cache_reduced_second_call from probe.json]. Each call starts with an unloaded model, so no cache is reused.
- Latency excludes model loading time.

## 3. Measures (fill from `summary.json` and `comparison.csv`)

| Measure | English | Bangla | Ratio (BN ÷ EN) |
|---|---|---|---|
| Total input tokens | [ ] | [ ] | [ ] |
| Total output tokens | [ ] | [ ] | [ ] |
| Total cost (USD) | [ ] | [ ] | [ ] |
| Mean latency per call (s) | [ ] | [ ] | [ ] |
| Median latency per call (s) | [ ] | [ ] | [ ] |

| Per-pair input ratio | Value |
|---|---|
| Mean (95% bootstrap interval of the mean) | [ ] ([ ], [ ]) |
| Median | [ ] |
| Minimum / maximum | [ ] / [ ] |

## 4. Chart
Insert `tokencost/results/chart.png`. Caption: ratio per category (left) and per pair (right); the dashed line marks 1.0 (equal cost).

## 5. Results by category
Name the category with the largest gap: [ ]. Name the one with the smallest gap: [ ].

## 6. Surprising prompt
[Pair id, the two prompts, the two token counts, the ratio, and your explanation.]

## 7. Quality check (10 pairs)
From `tokencost/data/quality_check.csv`: the Bangla answer was as correct as the English answer in [ ] of 10 pairs, and as useful in [ ] of 10. Explain the differences you found.

## 8. Conclusions
[Two or three sentences: how much more a Bangla prompt costs, whether the gap is consistent, and what this means for using Bangla with a small local model.]

## 9. Limitations
Single model, single seed, notional prices, a small sample of 50 pairs, and possible output-language effects.
