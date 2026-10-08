# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** 001  
**Team Members:** Sidaarth Krishnakanth (Sathyabama Institute of Science and Technology, Chennai, Tamil Nadu); Khavin S (Sathyabama Institute of Science and Technology, Chennai, Tamil Nadu)  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We retrieve candidates with multi-view TF-IDF search, prune them with a learned pre-ranker, and score them with three LightGBM stages; later stages see each pair's competition and siblings. A stacker trained on a held-out 20% of training entities rescores every pair. Most gains came from reverse-engineering how the data generator makes copies and lookalikes (a test-density correction, decoy rules, raw-name evidence, copy-vs-decoy transformation likelihoods). Three late additions carry the final version:
- two small character-level transformers, trained from scratch, that read both raw records side by side (their logits are averaged);
- a learned prior on how many candidates of an S1 are true, used inside the expected-F0.5 decision;
- a corrected street-number rule for France: renumbered copies with the same legal form are kept, only legal-form clones are rejected.

Macro F0.5 on the held-out S1 entities is **0.9889**; public leaderboard **0.986126** for this final version. Fully offline, no external data, no pretrained weights.

## 2. Methodology
### 2.1 Problem Analysis
- **Star-shaped generator.** Every S2/S3 record is an independent noisy copy of one S1; siblings are never closer to each other than to their S1, so graph signals were empty (AUC 0.50 beyond the model).
- **Planted lookalikes.** The S1 name plus one word from a per-country list at a neighbouring number (US "holdings", France "participations"); one category word swapped at the same address (Club→Comité); independently written homonyms at neighbouring numbers.
- **One-way noise suffixes** ("& Fils", "Services", "Cie") are gained by copies but never lost. **France has no street-number digit typos** (digit-typo signature ratio 0.012 vs 0.30 US, 0.33 India). This does not rule out renumbering: on the training holdout, exact-name copies with the same legal form whose street numbers share nothing are 82% (US) / 91% (India) true, and their copies agree with each other on the new number (twin rate 26–28% vs 1–4% for clones); planted clones differ in legal form.
- **Test density shift.** Test has ~1.9× as many unmatched records per S1 (57.5% of targets owned vs 73.4% in train), so word statistics computed on test make decoy words look like filler.
- **Raw form carries identity.** A copy keeps the owner's raw legal form ("Private Limited" vs "Pvt Ltd"), casing and word order; normalisation erases this.
- **Where the remaining loss is.** Perfect ranking inside candidates would add only +0.0001; the loss is in *how many* candidates to accept and in no-address copies of names shared by several S1s (the copy's spelling singles out the owner in only 10% of these).

### 2.2 Solution Strategy
**Approach Type:** Hybrid (blocking → learned pruning → 3-stage gradient boosting + neural pair model → stacker → generator rules → expected-F0.5 set selection with a cardinality prior).  
**Core Innovation:** generator-derived evidence measured per country from the data (an unseen country such as France is handled without naming it), a from-scratch neural reader of raw strings, and a decision that models the joint number of true matches instead of assuming independent labels.

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** TF-IDF top-k per country and target source over name character 4-grams (top 15), address words (top 15) and name + address using each query's 10 rarest features (top 30); the combined view also in reverse (top 3); exact core-name groups of at most 5 S1s. A LightGBM pre-ranker keeps at most 12 candidates per S1 per source. An exact key (same core name and first street number, identical address, one S1 claimant; 96% precise) recovers same-address copies the search misses in crowded name groups.
- **Candidate pairs generated:** 153.2M raw test pairs, pruned to 11,846,824 (6.84 per S1), plus 6.6k recovered pairs = `candidate_pairs.tsv`.
- **Ensuring recall:** pair recall on train 98.19%; a perfect matcher on the pruned set would score 0.9942. Most misses are no-address copies of shared names and Devanagari copies with partial addresses in crowded name groups.

## 4. Matching Model
**Features used:**
- **Name:** fuzzy ratios, token sets, rarity-weighted overlap, initials/acronyms, legal-form agreement, token-change counts, filler scores; 14 raw descriptors (casefold/alnum equality, order, casing, brackets, legal words as written, edit scripts); 19 copy-vs-decoy log-likelihood ratios of the S1→target transformation (legal-form rewrite, added/dropped words, character substitutions, typed edits), learned on training folds A+B only.
- **Address:** token overlap, street-number equality/edit distance/log difference, number sets, state.
- **Neural (G9):** logit of a character-level cross-encoder over `[CLS] S1 name | S1 address | target name | target address` (raw case, positions restart per field so characters align across records, long addresses keep their head and last 16 characters), plus its rank and margin inside the S1. 4-layer transformer, d = 128, 0.87M parameters, trained from scratch on an RTX 3050 on 2.6M fold-A+B pairs (hard pairs by stage-2 score + 15% sample). Two models: A (2 epochs, seed 7, validation hard-pair AUC 0.906) and B (4 epochs, seed 11, 0.934); G9 = the mean of their logits.
- **Other:** retrieval scores/ranks, pre-ranker score, name frequency, 19 context features (rank and margin within the S1, competition for the target, similarity to the S1's other likely matches), structural pair type.

**Model type:** LightGBM stages 1–2 cross-fitted on folds A/B, stage 2 refitted on A+B, stage 3 on context recomputed from stage-2 scores; base score = mean of stage-2 A+B and stage-3. Test token features of training countries use train statistics (density correction). A LightGBM stacker trained on the holdout fold H (never seen by the base models or the neural model) rescores every pair; template and neural features are used for training countries only.

**Threshold selection method:** each target goes to at most one S1; per S1 the top-k set maximising expected F0.5 is chosen (exact Poisson-binomial dynamic program; k = 0 protects singletons). For training countries the expected F0.5 uses a learned P(K = k true among the candidates): a multiclass LightGBM boosted from the Poisson-binomial of the scores (score profile, candidate types, S1 traits), trained on holdout out-of-fold scores. Before the decision, generator rules zero out decoys (category swap, decoy word, decoy vocabulary, and for countries without digit typos the street-number gate, restricted to legal-form conflicts) and raise same-address noise-suffix copies to ≥ 0.95.

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** holdout H (441,402 S1s) **0.9889**.

| Step | Holdout | Leaderboard |
|---|---|---|
| Stages 1–2, expected-F0.5 decision | 0.9840 | 0.97481 |
| + category swap, stage 3, decoy rules, test-density correction | 0.9858 | 0.98128 |
| + France number gate, noise suffixes, dual-use words | 0.9858 | 0.98169 |
| + raw-name stacker, same-address recovery | 0.9876 | 0.98452 |
| + copy-vs-decoy templates | 0.9882 | 0.98511 |
| + neural pair model (G9) | 0.9885 | — |
| + cardinality prior | 0.9887 | — |
| + France gate restricted to legal-form conflicts | 0.9887 (no France labels) | 0.985786 |
| + second neural model, logits averaged (final) | **0.9889** | **0.986126** |

- **Robustness of the neural and cardinality steps:** +0.00048 to +0.00052 over three stacker seeds for model A with the prior, +0.00070 / +0.00076 for the A+B average (two seeds), positive in both holdout halves and both countries; the neural signal is stronger on targets it never saw in training (within-stratum AUC 0.69 vs 0.55), so it is not memorisation.
- **Common false positives:** independently written homonyms at a neighbouring street number (especially after a one-digit change that is also a plausible typo); random aliases at the S1's exact address.
- **Common false negatives:** targets never retrieved (47% of remaining holdout loss; half without an address); no-address copies of names shared by k S1s, where each S1 owns the copy with probability ≈ 1/k (27%).

## 6. Conclusion
Precision-first set selection over calibrated scores, generator-aware evidence and a from-scratch neural reader of the raw strings carried the score from 0.9748 to 0.985+ on the leaderboard. The largest late gains came from information that normalisation discards (raw spelling, transformation type, character-level alignment) and from modelling how many matches an S1 has. Every addition was checked on held-out labels before use; most of what remains is information-limited (shared names without addresses).

---

## Appendix
### A. Code Artefacts
- **Location:** `code/business_entity_resolution/`: Python 3.11 package `ber` under `src/`, `README.md`, pinned `requirements.txt`, unit tests, trained neural weights in `weights/`.
- **Reproduce both output files:** `python -m ber.cli all --final --card --neural --gate open` (prepare, block, prerank, features, stage1–3, testshift, neural, final). With the shipped weights (`neural --score-only`) the output files are reproduced exactly; GPU retraining gives a slightly different neural model.
- **Fair play:** no external data, APIs, geocoding or lookups; transliteration maps learned from training pairs; US/Indian state and French region names are static lists for address parsing (other countries get none). Models: LightGBM (MIT) and our own two 0.87M-parameter transformers trained from scratch with PyTorch (BSD-3), released under MIT; far below 8B parameters; no pretrained models or LLMs.

### B. Additional Results (holdout, tested and rejected)
- entity profiles, S2↔S3 twin corroboration (AUC 0.50; 28.5% precision when propagating matches);
- raw address spelling +0.00003; digit-level street-number signatures −0.00001; key-based rescue blocking ≤ +0.00002;
- 5-seed stacker averaging +0.00003; LambdaRank stacker with isotonic calibration −0.008 (loses cross-S1 calibration);
- target-centric rescue retrieval (each unclaimed record searches the S1 pool in reverse): the pool held 16% of the missed pairs at 0.05% precision; top-1 by similarity < 5% precise, so information-limited;
- a density-matched holdout was not needed: a label-free check (true copies of a structural class per S1 are density-invariant) showed US/India test scores are calibrated class by class;
- cardinality prior for France: it raised France's empty-prediction share further above the generator's singleton rate, so France keeps the Poisson-binomial decision.
