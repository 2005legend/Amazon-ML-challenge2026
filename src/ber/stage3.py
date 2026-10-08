"""Stage 2 refit on folds A+B, and stage 3: context and sibling features recomputed from stage-2 scores.

The base score p23 of a pair is the mean of the A+B stage-2 score and the stage-3 score. Training rows are the
fold-H (holdout) pairs, which the A+B models never saw; test rows are scored the same way.
"""
from pathlib import Path

import numpy as np
import polars as pl

from .context import CTX_FEATS, GROUP_FEATS, SIB_FEATS, _tg_fields, group_features, sibling_features
from .features import MODEL_FEATS
from .model import KEYS, STAGE_PARAMS, load_table, neg_sample_expr, predict, save, table_files, train_lgb

K = ["s1_row", "tgt_src", "tgt_row"]
BASE = MODEL_FEATS + CTX_FEATS
Q = [f"q_{c}" for c in ["p2"] + GROUP_FEATS + SIB_FEATS]


def stage3_context(work_dir: Path, split: str, n_jobs: int) -> pl.DataFrame:
    """Group and sibling features recomputed from the stage-2 fold-model scores (columns prefixed q_)."""
    p = pl.concat([pl.read_parquet(f, columns=K + ["p2"]) for f in table_files(work_dir, split, "p2")])
    g = group_features(p, "p2").join(sibling_features(p, _tg_fields(work_dir, split), score="p2", n_jobs=n_jobs), on=K, how="left")
    return g.rename({c: f"q_{c}" for c in g.columns if c not in K})


def _fit(X: np.ndarray, y: np.ndarray, s1_ids: pl.Series, rounds: int = 4000):
    w = np.where(y == 1, 1.0, 2.0).astype(np.float32)
    val = (s1_ids.str.slice(3).cast(pl.Int64) % 20 == 0).to_numpy()
    return train_lgb(X[~val], y[~val], STAGE_PARAMS, rounds, valid=(X[val], y[val], w[val]), weight=w[~val])


def train_stage2_ab(work_dir: Path):
    df = load_table(work_dir, "train", "feats2", BASE + ["label", "s1_id"], pl.col("fold").is_in(["A", "B"]) & neg_sample_expr(0.5))
    b = _fit(df.select(BASE).cast(pl.Float32).to_numpy(), df["label"].to_numpy(), df["s1_id"])
    save(b, Path(work_dir) / "models" / "m2_AB.txt")
    return b


def train_stage3(work_dir: Path, ctx_train: pl.DataFrame):
    cols = list(dict.fromkeys(KEYS + BASE + ["label", "fold", "s1_id"]))
    parts = [pl.read_parquet(f, columns=cols).filter(pl.col("fold").is_in(["A", "B"]) & neg_sample_expr(0.5)).join(ctx_train, on=K, how="left")
             for f in table_files(work_dir, "train", "feats2")]
    df = pl.concat(parts)
    b = _fit(df.select(BASE + Q).cast(pl.Float32).to_numpy(), df["label"].to_numpy(), df["s1_id"])
    save(b, Path(work_dir) / "models" / "m3_AB.txt")
    return b


def score_split(work_dir: Path, split: str, out_name: str, booster, ctx: pl.DataFrame | None = None) -> None:
    """Score every feats2 table of `split` (train: holdout rows only) into <split>/<out_name>/ as column p2."""
    extra = ["label", "fold"] if split == "train" else []
    feats = BASE + (Q if ctx is not None else [])
    out = Path(work_dir) / split / out_name
    out.mkdir(parents=True, exist_ok=True)
    for f in table_files(work_dir, split, "feats2"):
        d = pl.read_parquet(f, columns=list(dict.fromkeys(KEYS + BASE + extra)))
        if split == "train":
            d = d.filter(pl.col("fold") == "H")
        if d.height == 0:
            continue
        if ctx is not None:
            d = d.join(ctx, on=K, how="left")
        p = predict(booster, d.select(feats).cast(pl.Float32).to_numpy())
        d.select(KEYS + extra).with_columns(pl.Series("p2", p, dtype=pl.Float32)).write_parquet(out / f.name)


def base_scores(work_dir: Path, split: str) -> pl.DataFrame:
    """p23 = mean of the A+B stage-2 and the stage-3 score (train: holdout pairs with labels)."""
    a = pl.scan_parquet(str(Path(work_dir) / split / "p2ab" / "*.parquet")).select("s1_id", "tgt_id", pl.col("p2").alias("a")).collect()
    extra = ["label"] if split == "train" else []
    b = pl.scan_parquet(str(Path(work_dir) / split / "p3" / "*.parquet")).select(["s1_id", "tgt_id", pl.col("p2").alias("b")] + extra).collect()
    return a.join(b, on=["s1_id", "tgt_id"]).select(["s1_id", "tgt_id", ((pl.col("a") + pl.col("b")) / 2).alias("p2")] + extra)


def run_stage3(work_dir: Path, n_jobs: int) -> None:
    m2 = train_stage2_ab(work_dir)
    ctx = {s: stage3_context(work_dir, s, n_jobs) for s in ("train", "test")}
    m3 = train_stage3(work_dir, ctx["train"])
    for split in ("train", "test"):
        score_split(work_dir, split, "p2ab", m2)
        score_split(work_dir, split, "p3", m3, ctx[split])
