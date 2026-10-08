"""Cardinality prior for the expected-F0.5 decision.

The standard decision treats the candidates' labels as independent, so the number of true targets K among an S1's
candidates follows the Poisson-binomial of their scores. A multiclass LightGBM, boosted from that Poisson-binomial
(its log is the init score), learns P(K = k) per S1 from the sorted score profile, sums and counts, the candidate types
(no-address target, exact name) and a few S1 traits. The decision then picks the top-k maximising

    E[F0.5 | top-k] = sum_K P(K) sum_a P(a true in top-k | K) F(a, k, K),

with P(a | K) from the Poisson-binomials of the top-k and of the rest, conditioned on their sum being K. With
P(K) = the Poisson-binomial itself this is exactly decide.expected_f_select.

The model is trained on the holdout fold H from out-of-fold stacker scores, after the holdout's own exclusions and
noise boosts (noise-word statistics from the holdout's own pairs, as the test split's come from the test pairs), and
is applied to S1s of countries present in training; other countries keep the Poisson-binomial prior.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from numba import njit

from . import config
from .decide import _poisson_binomial, many_to_one
from .genrules import FC, boost, decoy_vocabulary, decoy_word_pairs, edits, noise_addition_pairs, vocab_pairs
from .model import load_table
from .rules import catswap_pairs, exclude_scores

MAXK, KC = 12, 13          # candidates considered per S1 (as in the decision); count classes 0..12
FE = ["n", "psum", "n05", "n09", "n02", "n_noaddr", "p_noaddr", "n_exact", "p_exact", "n_s2", "name_len", "name_tok", "is_us", "addr_empty"]
PARAMS = {"objective": "multiclass", "num_class": KC, "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": config.N_JOBS, "seed": 7}



@njit(cache=True)
def _choose(starts, p, prior, beta2):
    keep = np.zeros(len(p), np.bool_)
    for g in range(len(starts) - 1):
        s = starts[g]; e = min(starts[g + 1], s + MAXK); n = e - s
        q = p[s:e]; pk = prior[g].copy()
        for K in range(n + 1, KC):
            pk[K] = 0.0
        tot = pk[:n + 1].sum()
        if tot <= 0:
            continue
        pk = pk / tot
        best, bk = pk[0], 0
        for k in range(1, n + 1):
            top = _poisson_binomial(q[:k]); rest = _poisson_binomial(q[k:]); v = 0.0
            for K in range(1, min(n, KC - 1) + 1):
                if pk[K] <= 0:
                    continue
                z, num = 0.0, 0.0
                for a in range(max(0, K - (n - k)), min(k, K) + 1):
                    w = top[a] * rest[K - a]
                    z += w; num += w * (1.0 + beta2) * a / (beta2 * K + k)
                if z > 0:
                    v += pk[K] * num / z
            if v > best:
                best, bk = v, k
        for i in range(s, s + bk):
            keep[i] = True
    return keep


def choose(starts: np.ndarray, p: np.ndarray, prior: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Rows kept by the prior-weighted expected-F decision; p sorted descending inside each group [starts[g], starts[g+1])."""
    return _choose(starts, p, prior, beta * beta)


def pb_matrix(starts: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.zeros((len(starts) - 1, KC))
    for g in range(len(starts) - 1):
        d = _poisson_binomial(p[starts[g]:min(starts[g + 1], starts[g] + MAXK)])
        out[g, :len(d)] = d
    return out


def assigned(scores: pl.DataFrame, types: pl.DataFrame) -> pl.DataFrame:
    a = many_to_one(scores, "p2").join(types, on=["s1_id", "tgt_id"], how="left").sort(["s1_id", "p2"], descending=[False, True])
    return a.filter(pl.int_range(pl.len()).over("s1_id") < MAXK)


def design(a: pl.DataFrame, s1: pl.DataFrame):
    """a: assigned pairs sorted by S1 then score (s1_id, tgt_id, p2, ar, nr[, label]); s1: s1_id, country, name, addr_empty.
    Returns the S1 frame, the feature matrix, group starts, scores and the normalised Poisson-binomial prior."""
    top = a.group_by("s1_id", maintain_order=True).agg(pl.col("p2").alias("ps"))
    P = np.zeros((top.height, MAXK), np.float32)
    for i, v in enumerate(top["ps"].to_list()):
        P[i, :len(v)] = v
    extra = [pl.col("label").sum().alias("K")] if "label" in a.columns else []
    agg = a.group_by("s1_id").agg(
        pl.len().alias("n"), pl.col("p2").sum().alias("psum"), (pl.col("p2") > 0.5).sum().alias("n05"), (pl.col("p2") > 0.9).sum().alias("n09"),
        (pl.col("p2") > 0.2).sum().alias("n02"), (pl.col("ar") == "noaddr").sum().alias("n_noaddr"),
        pl.col("p2").filter(pl.col("ar") == "noaddr").sum().alias("p_noaddr"), (pl.col("nr") == "exact").sum().alias("n_exact"),
        pl.col("p2").filter(pl.col("nr") == "exact").sum().alias("p_exact"), pl.col("tgt_id").str.starts_with("S2").sum().alias("n_s2"), *extra)
    f = (top.select("s1_id").join(agg, on="s1_id", how="left").join(s1, on="s1_id", how="left")
         .with_columns(pl.col("name").str.len_chars().alias("name_len"), pl.col("name").str.split(" ").list.len().alias("name_tok"),
                       (pl.col("country") == "US").cast(pl.Int8).alias("is_us"), pl.col("addr_empty").cast(pl.Int8)))
    X = np.hstack([P, f.select(FE).fill_null(0).cast(pl.Float32).to_numpy()])
    gid = a.select(pl.col("s1_id").rle_id())["s1_id"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(gid)) + 1, [len(gid)]]).astype(np.int64)
    p = a["p2"].to_numpy().astype(np.float64)
    pb = pb_matrix(starts, p)
    return f, X, starts, p, pb / pb.sum(1, keepdims=True)


def card_probs(booster: lgb.Booster, X: np.ndarray, pb: np.ndarray) -> np.ndarray:
    z = booster.predict(X, raw_score=True) + np.log(np.clip(pb, 1e-6, 1))
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def holdout_flags(work_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Exclusions (category swap, decoy word, decoy vocabulary) and noise boosts for the holdout fold."""
    x = edits(work_dir, "train", load_table(work_dir, "train", "feats", FC))
    ex = pl.concat([catswap_pairs(work_dir, "train").select("s1_id", "tgt_id"), decoy_word_pairs(x), vocab_pairs(x, decoy_vocabulary(work_dir))]).unique()
    nb = noise_addition_pairs(edits(work_dir, "train", load_table(work_dir, "train", "feats", FC, pl.col("fold") == "H")))
    return ex, nb


def fit_cardinality(work_dir: Path, oof: pl.DataFrame, types: pl.DataFrame, s1: pl.DataFrame, model_dir: Path) -> lgb.Booster:
    """oof: holdout s1_id, tgt_id, label, p2 (out-of-fold stacker scores)."""
    ex, nb = holdout_flags(work_dir)
    a = assigned(boost(exclude_scores(oof.select("s1_id", "tgt_id", "p2"), ex, "p2"), nb), types)
    a = a.join(oof.select("s1_id", "tgt_id", "label"), on=["s1_id", "tgt_id"], how="left")
    f, X, _, _, pb = design(a, s1)
    y, init = f["K"].clip(0, KC - 1).to_numpy(), np.log(np.clip(pb, 1e-6, 1))
    b = lgb.train(PARAMS, lgb.Dataset(X, label=y, init_score=init), select_rounds(X, y, init, (f["s1_id"].hash(3) % 2).to_numpy()))
    b.save_model(str(Path(model_dir) / "cardinality.txt"))
    return b


def select_rounds(X: np.ndarray, y: np.ndarray, init: np.ndarray, fold: np.ndarray) -> int:
    """Mean early-stopping iteration over the two S1 folds (every 20th training row held out for stopping)."""
    its = []
    for k in (0, 1):
        tr = fold != k; va = tr & (np.arange(len(y)) % 20 == 0); fit = tr & ~va
        b = lgb.train(PARAMS, lgb.Dataset(X[fit], label=y[fit], init_score=init[fit]), 3000,
                      valid_sets=[lgb.Dataset(X[va], label=y[va], init_score=init[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        its.append(b.best_iteration)
    return int(round(np.mean(its)))


def prior_decision(booster: lgb.Booster, types: pl.DataFrame, s1: pl.DataFrame, seen: list[str]):
    """Returns the last-decision function for genrules.apply_rules: the cardinality prior for S1s of seen countries,
    the Poisson-binomial (= the standard decision) for the others."""
    def last(scores: pl.DataFrame) -> pl.DataFrame:
        a = assigned(scores, types)
        f, X, starts, p, pb = design(a, s1)
        use = f["country"].is_in(seen).to_numpy()
        prior = pb.copy()
        prior[use] = card_probs(booster, X[use], pb[use])
        return a.filter(pl.Series(choose(starts, p, prior))).select("s1_id", "tgt_id")
    return last
