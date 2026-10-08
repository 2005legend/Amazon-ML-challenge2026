# Business Entity Resolution — Amazon ML Challenge 2026

Team 001: Sidaarth Krishnakanth and Khavin S (Sathyabama Institute of Science and Technology, Chennai).
**Public leaderboard: macro F0.5 0.986126 (top 100).**

The task: for each of 1.73M reference businesses (Source 1), find all of its noisy copies among ~10M records in
Sources 2 and 3. There are no shared IDs; names and addresses carry typos, abbreviations, reordering and Indic scripts.
Test adds France, a country that never appears in training. Scoring is macro F0.5 per business, so a wrong merge costs
about three times more than a missed one.

## How we got here

| Step | What changed | Holdout | Leaderboard |
|---|---|---:|---:|
| Baseline | TF-IDF blocking, LightGBM pair scorer, expected-F0.5 decision | 0.9840 | 0.97481 |
| Decoys found | Test contains planted lookalikes (category-word swaps at the same address); reject them | 0.9840 | 0.97832 |
| Test shift | Test is denser than train, so token statistics leaked; recompute them with train statistics, add decoy-word and decoy-vocabulary rules | 0.9858 | 0.98128 |
| France gate | France has no street-number typos, so a different number means a different business | 0.9858 | 0.98169 |
| Raw strings | Normalization was destroying evidence; add raw-name features to a holdout stacker, recover missed same-address copies | 0.9876 | 0.98452 |
| Copy vs decoy | Learn which name transformations the generator uses for true copies versus decoys | 0.9882 | 0.98511 |
| Neural + cardinality | Character-level transformer trained from scratch; learned prior on how many matches a business has | 0.9887 | 0.98579 |
| Final | Second neural model averaged in; France gate relaxed to legal-form conflicts only | **0.9889** | **0.98613** |

The largest jumps came from reverse-engineering how the data generator builds copies and decoys, not from bigger
models. Several ideas were measured and dropped: graph/entity profiles, target-side rescue retrieval, sibling and
anchor propagation, and ranking-based stackers. The remaining gap sits in address-less records whose names are shared
by several businesses, which the data cannot disambiguate. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

The competition data is not included; download it from the challenge portal.

This pipeline links every Source-1 business to its Source-2/Source-3 records. The steps are:

1. **Normalize** names and addresses. Indic-script names are transliterated with a token dictionary learned only from training pairs, with a rule-based fallback.
2. **Generate candidates** with sparse TF-IDF top-k retrieval in both directions: S1→S2/S3 and S2/S3→S1. It searches name character 4-grams, address tokens, and both combined.
3. **Prune candidates** with a small LightGBM pre-ranker. The pruned set is `candidate_pairs.tsv`.
4. **Score pairs** with LightGBM: stage 1 on 76 retrieval, similarity, frequency and token-change features; stage 2 adds 19 context, competition and sibling features; stage 3 recomputes that context from stage-2 scores. The base score is the mean of the A+B stage-2 and stage-3 scores.
5. **Correct the test-set density shift**: for test countries that exist in train, token-change features are recomputed with train statistics and stages 1–3 rescore the test pairs.
6. **Read the raw strings with a neural pair model (G9)**: a character-level transformer cross-encoder (0.87M parameters, trained from scratch on training folds A+B, GPU) reads the S1 and target name + address side by side and scores every holdout and test pair. Two such models are trained (A: 2 epochs, seed 7; B: 4 epochs, seed 11) and the G9 logit is the mean of their logits.
7. **Stack**: a LightGBM fitted on the held-out 20% of training entities rescores every pair, adding raw-name descriptors, copy-vs-decoy transformation likelihoods (learned on training folds A+B) and the neural logit with its rank and margin inside the S1 (the last two groups for training countries only).
8. **Apply generator rules**: remove planted decoys (category swaps, decoy words, decoy vocabulary) and restore same-address noise-suffix copies. For countries without street-number typos, a street-number gate rejects exact-name copies whose numbers share nothing; with `--gate open` (submitted) it only rejects the ones whose legal forms also conflict, because on the training holdout same-legal-form copies with renumbered streets are 82-98% true. Every threshold is computed per country from the data.
9. **Choose matches**: each S2/S3 record goes to at most one S1, and each S1's match set is the top-k that maximizes expected F0.5. For training countries the expected F0.5 uses a learned distribution of how many candidates are true (a multiclass LightGBM boosted from the Poisson-binomial of the scores, trained on the holdout) instead of assuming independent labels.
10. **Recover same-address copies** the search missed in crowded name groups (exact core name, same first number, identical address, a single S1 claimant); these pairs are added to both output files.

## Requirements

- Python 3.11, about 16 GB RAM, about 20 GB free disk for `work/` (a full run leaves ~12 GB), no network access.
- An NVIDIA GPU with at least 4 GB (CUDA 12) for the `neural` step only (developed on an RTX 3050 Laptop GPU). `torch` is installed from the PyTorch CUDA index listed in `requirements.txt`.
- Everything runs offline. No external data or services are used.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash;  PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Data location

By default the pipeline reads `../6ab10eb3b23ba_student_resource/student_resource/dataset/` (the folder that contains `train/` and `test/`). To use another location, set `BER_DATA_DIR`:

```bash
export BER_DATA_DIR=/path/to/dataset        # PowerShell: $env:BER_DATA_DIR="C:\path\to\dataset"
```

Intermediate files go to `work/` (`BER_WORK_DIR`) and outputs to `output/` (`BER_OUTPUT_DIR`).

## Reproduce end to end

One command runs every stage with the final settings:

```bash
python -m ber.cli all --final --card --neural --gate open
```

The same run, step by step (wall-clock times measured on a 16-thread, 16 GB Windows laptop with an RTX 3050 Laptop GPU; about 4 h in total):

```bash
python -m ber.cli prepare --split train     # normalize + learn transliteration maps
python -m ber.cli prepare --split test
python -m ber.cli block --split train       # raw candidates (TF-IDF top-k, both directions)
python -m ber.cli block --split test
python -m ber.cli prerank                   # learned pruning -> work/<split>/cands (= candidate_pairs.tsv)
python -m ber.cli features --split train    # pairwise similarity features
python -m ber.cli features --split test
python -m ber.cli stage1                    # LightGBM, cross-fitted on folds A/B
python -m ber.cli stage2                    # + context / competition / sibling features
python -m ber.cli stage3                    # stage 2 refit on folds A+B, stage 3; score holdout H and test
python -m ber.cli testshift                 # test token features with train statistics, rescore stages 1-3 (work/tc)
python -m ber.cli neural                    # char-level pair models A and B: prep, train on A+B (GPU), score holdout + test (work/nn)
python -m ber.cli final --card --neural --gate open   # stacker (+G9), generator rules, cardinality-prior decision, recovery; write + validate
```

| Step | Time |
|---|---|
| `prepare` train + test (24.0M records) | 5 min |
| `block` train / test | 55 min / 31 min |
| `prerank` (train + apply to both splits) | 15 min |
| `features` train + test (24.3M pairs) | 7 min |
| `stage1` | 28 min |
| `stage2` | 14 min |
| `stage3` | ≈ 1 h |
| `testshift` | ≈ 40 min |
| `neural` (RTX 3050 Laptop: model A 46 min + model B ≈ 2 h training; each scores holdout ≈ 6 min + test ≈ 30 min) | ≈ 4 h |
| `final --card --neural --gate open` | ≈ 25 min |

Macro F0.5 on the held-out 20% of training S1 entities (H): **0.9889**; public leaderboard **0.986126** (this version; the previous version with one neural model scored 0.985786).
The earlier `submit` command (stage-2 scores, expected-F0.5 rule, optional `--catswap`) still produces the first, simpler submission.

## Neural weights and reproducibility

GPU training uses atomic additions, so retraining gives a slightly different model. The weights behind the submitted
files are shipped in `weights/` (`model.pt` = model A, `model_b.pt` = model B, `vocab.npy`, 7 MB); to reproduce the submission exactly, copy them to
`work/nn/` and score instead of training:

```bash
mkdir -p work/nn && cp weights/model.pt weights/model_b.pt weights/vocab.npy work/nn/
python -m ber.cli neural --score-only       # prepares the pair tables if needed, scores holdout + test with the shipped weights
```

Everything else (LightGBM with fixed seeds and thread-independent histograms) reproduces bit for bit.

## Outputs

- `output/matching_results.tsv`: the final matches, one row per test S1 entity.
- `output/candidate_pairs.tsv`: the exact candidate set the matcher scored.

`submit` runs the official validator (`utils/validate_submission.py --check-ids`) and exits with its status.

## Module map (`src/ber/`)

| Module | Responsibility |
|---|---|
| `config.py` | paths, seed, worker count |
| `io.py` | TSV read/write (tab-separated, no quoting) |
| `normalize.py` | name/address normalization, legal forms, DBA split, state tables (US, India, France) |
| `translit.py` | Indic→Latin token/state maps learned from train pairs; anyascii fallback |
| `prepare.py` | parallel normalization of every source file to Parquet |
| `vectorize.py` | char 3/4-gram and hashed-word sparse vectors, TF-IDF |
| `topk.py` | numba parallel sparse top-k retrieval and pairwise dot products |
| `blocking.py` | candidate generation per country and source, recall report |
| `prerank.py` | cheap candidate features and the pruning model |
| `features.py` | pairwise similarity features (process pool) |
| `folds.py` | deterministic fold assignment (A 40% / B 40% / holdout H 20%) |
| `model.py` | LightGBM helpers, cross-fitted training and scoring |
| `context.py` | stage-2 context, competition and sibling features |
| `decide.py` | many-to-one assignment, threshold and expected-F0.5 selection |
| `evaluate.py` | macro F0.5 exactly as the challenge defines it |
| `diagnostics.py` | holdout breakdowns, orphan stress test, test country report |
| `stage3.py` | stage 2 refit on folds A+B, stage 3 context model, base score p23 |
| `testshift.py` | test token-change features with train statistics; rescoring of stages 1–3 |
| `rawform.py` | structural pair types, raw-name descriptors, copy-vs-decoy template likelihoods, S1 score shape |
| `neural.py` | character-level cross-encoder (G9): encoding, training models A and B on folds A+B, scoring, averaged logit with rank/margin context |
| `stacker.py` | holdout-trained LightGBM stacker, out-of-fold holdout scores |
| `cardinality.py` | cardinality prior: P(K true among candidates) model and the prior-weighted expected-F0.5 decision |
| `genrules.py` | generator-derived exclusion and boost rules and the rule chain |
| `recovery.py` | same-address recovery of missed exact copies |
| `final.py` | final scoring, decision and output writing |
| `package.py` | output files, validation, submission zip |
| `cli.py` | command-line entry point |

## Tests

```bash
python -m pytest
```
