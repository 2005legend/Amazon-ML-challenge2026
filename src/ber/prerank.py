"""Cheap candidate features and a small LightGBM that prunes raw candidates (supervised meta-blocking)."""
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .folds import fold_expr
from .blocking import gt_rows
from .model import predict, train_lgb
from .prepare import norm_path

VIEW_COLS = ("n", "a", "na")
PRE_FEATS = (["cos_n", "cos_a", "cos_na", "via"]
             + [f"{p}_{v}" for v in VIEW_COLS for p in ("rk1", "gap1", "rkt", "gapt")]
             + ["n_s1", "n_tgt"])
FLOOR, CAP = 0.002, 12
PRE_PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "seed": config.SEED, "num_threads": config.N_JOBS}


def prerank_features(c: pl.DataFrame) -> pl.DataFrame:
    """Window features inside one (country, target-source) candidate file."""
    exprs = []
    for v in VIEW_COLS:
        col = pl.col(f"cos_{v}")
        exprs += [col.rank("ordinal", descending=True).over("s1_row").cast(pl.Int32).alias(f"rk1_{v}"),
                  (col.max().over("s1_row") - col).alias(f"gap1_{v}"),
                  col.rank("ordinal", descending=True).over("tgt_row").cast(pl.Int32).alias(f"rkt_{v}"),
                  (col.max().over("tgt_row") - col).alias(f"gapt_{v}")]
    exprs += [pl.len().over("s1_row").cast(pl.Int32).alias("n_s1"),
              pl.len().over("tgt_row").cast(pl.Int32).alias("n_tgt")]
    return c.with_columns(exprs)


def prune(c: pl.DataFrame, floor: float, cap: int) -> pl.DataFrame:
    rank = pl.col("p_pre").rank("ordinal", descending=True).over("s1_row")
    return c.filter((pl.col("p_pre") >= floor) & (rank <= cap))


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(PRE_FEATS).cast(pl.Float32).to_numpy()


def train_prerank(work_dir: Path, data_dir: Path, sample_frac: float = 0.25, folds=("A",), rounds: int = 300):
    """Fit on a sample of fold-A S1 rows; window features use whole files, labels join on int keys."""
    keys = ["s1_row", "tgt_src", "tgt_row"]
    s1 = pl.read_parquet(norm_path(work_dir, "train", 1), columns=["row", "entity_id"]).rename({"entity_id": "s1_id"})
    sampled = s1.filter(fold_expr().is_in(list(folds)) & (
        (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 100) < int(sample_frac * 100)))["row"]
    labels = gt_rows(work_dir, data_dir).select(keys).with_columns(pl.lit(1, pl.Int8).alias("label"))
    frames = []
    for f in sorted((Path(work_dir) / "train" / "cands_raw").glob("*.parquet")):
        c = pl.read_parquet(f)
        if c.height == 0:
            continue
        c = prerank_features(c).filter(pl.col("s1_row").is_in(sampled.implode()))
        frames.append(c.join(labels, on=keys, how="left").select(PRE_FEATS + ["label"]))
    c = pl.concat(frames).with_columns(pl.col("label").fill_null(0))
    return train_lgb(_matrix(c), c["label"].to_numpy(), PRE_PARAMS, rounds=rounds)


def apply_prerank(split: str, work_dir: Path, booster, floor: float = FLOOR, cap: int = CAP) -> None:
    out_dir = Path(work_dir) / split / "cands"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted((Path(work_dir) / split / "cands_raw").glob("*.parquet")):
        c = prerank_features(pl.read_parquet(f))
        p = predict(booster, _matrix(c)) if (booster is not None and c.height) else np.ones(c.height, np.float32)
        c = c.with_columns(pl.Series("p_pre", p, dtype=pl.Float32))
        (prune(c, floor, cap) if booster is not None else c).write_parquet(out_dir / f.name)
