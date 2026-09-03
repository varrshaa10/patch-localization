# DL reranker experiments: honest results

This folder contains experiment code only. Neither reranker experiment is used in the production NCC pipeline. The verified production path remains the classical NCC matcher in the core code under `algorithm/core`, and `infer.py` is intentionally left untouched.

## 1) Classification-based reranker experiment

The first reranker used binary match/no-match labeling and a class-weighted BCE objective. On the held-out synthetic patch data, it achieved a valid ranking signal of about `0.916` on the patch-level validation/test split.

However, that signal did not transfer to the real end-to-end task. On the real pair-level evaluation against ground truth, the no-marker/ambiguous cases regressed badly:

- plain NCC no-marker mean error: ~`40px`
- reranked no-marker mean error: ~`195px`

For marker cases, the reranker was effectively neutral (`~8.09px` both plain and reranked), and on the full set the reranker was worse overall (`39.27px` vs `13.43px` mean error). This experiment was therefore not used in production and was considered a failed additive layer.

## 2) Regression reframe experiment

The second experiment reframed the task as distance-to-ground-truth regression using a continuous similarity target `exp(-distance / scale)`, with `scale = 20` as a starting point. The continuous target distribution was saved separately and showed a reasonable range:

- min = `0.000000`
- max = `1.000000`
- mean = `0.116332`

A 5-epoch proof run was executed to check for a learning signal before any larger training. The observed validation ranking stayed near chance level (~`0.50` after 2 epochs) and did not show a stable upward trend. The experiment was stopped early and not taken to the full 30-epoch regime.

## Final status

- The classical NCC pipeline is the verified production approach.
- The reranker experiments are retained only as negative evidence in this folder.
- Neither model is imported by the production pipeline, and `infer.py` continues to use the classical NCC path as shipped.
