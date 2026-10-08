"""Decision rules: many-to-one assignment, thresholding and expected-F0.5 set selection."""
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from .evaluate import macro_fbeta
from .folds import fold_expr
from .io import read_ground_truth
from .model import load_table
from .prepare import norm_path


def many_to_one(pairs: pl.DataFrame, score: str) -> pl.DataFrame:
    return (pairs.sort([score, "s1_id"], descending=[True, False])
                 .unique(subset=["tgt_id"], keep="first", maintain_order=True))


def apply_threshold(pairs: pl.DataFrame, score: str, tau: float) -> pl.DataFrame:
    return pairs.filter(pl.col(score) >= tau)


def tune_threshold(pairs, truth, s1_ids, score, grid=None) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        f = macro_fbeta(apply_threshold(pairs, score, float(t)), truth, s1_ids)
        if f > best[1]:
            best = (float(t), f)
    return best


@njit(cache=True)
def _poisson_binomial(p):
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for i in range(len(p)):
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1.0 - p[i]) + d[j - 1] * p[i]
        d[0] *= 1.0 - p[i]
    return d


@njit(cache=True)
def best_k(p, beta2):
    """Number of top candidates maximizing expected F-beta (p sorted descending, independent labels)."""
    K = len(p)
    best_val = 1.0
    for i in range(K):
        best_val *= 1.0 - p[i]
    best = 0
    for k in range(1, K + 1):
        tp = _poisson_binomial(p[:k])
        fn = _poisson_binomial(p[k:])
        e = 0.0
        for a in range(1, k + 1):
            for b in range(K - k + 1):
                e += tp[a] * fn[b] * (1.0 + beta2) * a / (beta2 * (a + b) + k)
        if e > best_val:
            best_val, best = e, k
    return best


@njit(cache=True)
def _select_groups(starts, p, beta2, max_k):
    keep = np.zeros(len(p), np.bool_)
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        k = best_k(p[s:min(e, s + max_k)], beta2)
        for i in range(s, s + k):
            keep[i] = True
    return keep


def expected_f_select(pairs: pl.DataFrame, score: str, beta: float = 0.5, max_k: int = 12) -> pl.DataFrame:
    df = pairs.sort(["s1_id", score], descending=[False, True])
    if df.height == 0:
        return df
    gid = df.select(pl.col("s1_id").rle_id())["s1_id"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(gid)) + 1, [len(gid)]]).astype(np.int64)
    keep = _select_groups(starts, df[score].to_numpy().astype(np.float64), beta * beta, max_k)
    return df.filter(pl.Series(keep))


def holdout_tune(work_dir: Path, data_dir: Path, score: str, rule: str, exclude: pl.DataFrame | None = None) -> dict:
    pairs = load_table(work_dir, "train", score, ["s1_id", "tgt_id", score, "fold"])
    if exclude is not None:
        from .rules import exclude_scores
        pairs = exclude_scores(pairs, exclude, score)
    s1 = (pl.read_parquet(norm_path(work_dir, "train", 1), columns=["entity_id"])
            .rename({"entity_id": "s1_id"}).with_columns(fold_expr().alias("fold")))
    h_ids = s1.filter(pl.col("fold") == "H")["s1_id"]
    gt = read_ground_truth(Path(data_dir) / "train" / "train_ground_truth.tsv")
    truth = gt.filter(pl.col("s1_id").is_in(h_ids.implode()))
    assigned = many_to_one(pairs, score).filter(pl.col("fold") == "H")
    raw = pairs.filter(pl.col("fold") == "H")
    if rule == "threshold":
        tau, f = tune_threshold(assigned, truth, h_ids, score)
        _, f_raw = tune_threshold(raw, truth, h_ids, score)
    else:
        tau = None
        f = macro_fbeta(expected_f_select(assigned, score), truth, h_ids)
        f_raw = macro_fbeta(expected_f_select(raw, score), truth, h_ids)
    return {"score": score, "rule": rule, "tau": tau, "f05_H": f, "f05_H_without_many_to_one": f_raw}


def decide(pairs: pl.DataFrame, score: str, decision: dict) -> pl.DataFrame:
    assigned = many_to_one(pairs, score)
    if decision["rule"] == "threshold":
        return apply_threshold(assigned, score, decision["tau"])
    return expected_f_select(assigned, score)
