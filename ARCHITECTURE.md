# Architecture

```
S1 / S2 / S3 records
      │  normalize (legal forms, abbreviations, accents, numbers, Indic transliteration)
      ▼
Multi-view TF-IDF blocking (name char 4-grams, address words, name+address; both directions)
      │  LightGBM pre-ranker: ≤12 candidates per S1 per source (~5.6 per S1)
      ▼
Stage 1  LightGBM on 76 pair features (similarity, numbers, token changes, frequencies)
Stage 2  + candidate competition and sibling context
Stage 3  context recomputed from stage-2 scores
      │  test-shift correction: token statistics recomputed with train statistics
      ▼
Holdout stacker (LightGBM on 20% held-out businesses)
   + raw-name descriptors (case, brackets, legal spelling)
   + copy-vs-decoy transformation likelihoods
   + char-level cross-encoder (2 models, averaged logits)
      ▼
Generator rules: category-swap, decoy-word, decoy-vocabulary rejection; France street-number gate
      ▼
Decision: one owner per target; per S1 the top-k that maximizes expected F0.5,
          with a learned prior on the number of true matches
      ▼
Same-address recovery of exact copies blocking missed  →  matching_results.tsv, candidate_pairs.tsv
```

## Why this shape

1. **Blocking sets the ceiling.** Recall after pruning is 98.2% of true pairs (perfect-matcher ceiling 0.9942),
   so everything later is about precision and choosing how many to accept.
2. **Context matters.** A pair's score depends on its rivals: stages 2–3 see how a target ranks among the S1's
   candidates and how strongly other S1s claim it.
3. **The generator is the signal.** The test set contains decoys (lookalike businesses at the same or neighbouring
   address) at rates train barely shows. Rules derived from data statistics, not hand lists, removed them and gave the
   biggest leaderboard jumps.
4. **Raw strings carry evidence normalization removes.** Exact casing, bracket and legal-form spelling, and a small
   transformer reading both raw records side by side, each added holdout gain.
5. **F0.5 is per business.** The decision layer optimizes expected F0.5 per S1 rather than a global threshold, and a
   cardinality model tells it how many candidates are likely true.

## Validation

- Folds by S1: A 40% / B 40% train the stage models; holdout H 20% trains the stacker (2-fold by S1 for estimates).
- Every change passed a gate on H: gain in both folds, US and India not worse, singletons not worse.
- France has no labels; France changes were tested only by leaderboard A/B uploads.

## What did not work (measured)

| Idea | Result |
|---|---|
| Entity profiles / graph / S2↔S3 agreement | no signal beyond the model (AUC 0.50) |
| Target-centric rescue retrieval | pool precision 0.05% |
| Sibling / anchor propagation for address-less records | 11–34% precise; ΔH ≤ 0 |
| Ranking stacker + isotonic calibration | −0.008 holdout |
| Bigger stackers, seed averaging | ≈ 0 |
| Full stacker for France | drops true copies |
