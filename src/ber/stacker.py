"""Residual stacker: a LightGBM fitted on the holdout pairs (fold H), which the base models never saw, so its inputs
look like test inputs. It rescores every pair from logit(p23), the 76 pair features, the 19 context features, the
structural type, the raw-name descriptors and the copy-vs-decoy template features (optionally the S1 score shape).

The template tables are learned from training countries only, so for S1s of a country absent from training the
stacker WITHOUT template features is used (same principle as the test-set density correction).
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .context import CTX_FEATS
from .features import MODEL_FEATS
from .rawform import AR, G2, G5, G8, NR, TR

PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": config.N_JOBS, "seed": 7}
ROUNDS = 300
BASE = list(dict.fromkeys(["lg"] + MODEL_FEATS + CTX_FEATS + ["ar_c", "nr_c", "tr_c"]))
FULL = BASE + G2


def encode(x: pl.DataFrame) -> pl.DataFrame:
    m = lambda c, v: pl.col(c).replace_strict(v, list(range(len(v))), default=-1).cast(pl.Float32).alias(c + "_c")
    p = pl.col("p2").clip(1e-6, 1 - 1e-6)
    return x.with_columns([m("ar", AR), m("nr", NR), m("tr", TR)] + [pl.col(c).cast(pl.Float32) for c in G2]
                          + [(p / (1 - p)).log().cast(pl.Float32).alias("lg")])


def stack_rows(scores: pl.DataFrame, feats2: pl.LazyFrame, types: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    f2 = feats2.select(["s1_id", "tgt_id", "country"] + list(dict.fromkeys(MODEL_FEATS + CTX_FEATS))).collect()
    x = scores.join(f2, on=["s1_id", "tgt_id"]).join(types, on=["s1_id", "tgt_id"]).join(raw.select(["s1_id", "tgt_id"] + G2), on=["s1_id", "tgt_id"], how="left")
    return encode(x)


def add_groups(x: pl.DataFrame, shape: pl.DataFrame | None, templates: pl.DataFrame) -> pl.DataFrame:
    if shape is not None:
        x = x.join(shape, on=["s1_id", "tgt_id"], how="left").with_columns([pl.col(c).cast(pl.Float32) for c in G5])
    return x.join(templates, on=["s1_id", "tgt_id"], how="left")


def fit(x: pl.DataFrame, cols: list[str]) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(x.select(cols).cast(pl.Float32).to_numpy(), label=x["label"].to_numpy()), num_boost_round=ROUNDS)


def oof_scores(x: pl.DataFrame, cols: list[str], rounds: int = ROUNDS) -> pl.DataFrame:
    """2-fold out-of-fold stacker scores on the holdout, folds by S1 (hash seed 3), for models fitted downstream."""
    X, y = x.select(cols).cast(pl.Float32).to_numpy(), x["label"].to_numpy()
    fold = (x["s1_id"].hash(3) % 2).to_numpy()
    o = np.zeros(len(y), np.float32)
    for k in (0, 1):
        o[fold == k] = lgb.train(PARAMS, lgb.Dataset(X[fold != k], label=y[fold != k]), num_boost_round=rounds).predict(X[fold == k])
    return x.select("s1_id", "tgt_id").with_columns(pl.Series("p2", o))


def stacked_scores(train: pl.DataFrame, test: pl.DataFrame, base_test: pl.DataFrame, use_shape: bool, seen: list[str],
                   model_dir: Path, extra: list[str] = ()) -> pl.DataFrame:
    """Scores for every base test pair: template stacker (+ `extra` columns, e.g. the neural G9) for training
    countries, plain stacker otherwise, the base p23 for the few pairs without stacker inputs."""
    ref_cols = FULL + (G5 if use_shape else [])
    out = base_test.select("s1_id", "tgt_id", pl.col("p2").alias("p_base"))
    for name, cols in (("ref", ref_cols), ("tmpl", ref_cols + G8 + list(extra))):
        b = fit(train, cols)
        b.save_model(str(Path(model_dir) / f"stack_{name}.txt"))
        p = test.select("s1_id", "tgt_id").with_columns(pl.Series(name, b.predict(test.select(cols).cast(pl.Float32).to_numpy()).astype(np.float32)))
        out = out.join(p, on=["s1_id", "tgt_id"], how="left")
    c = test.select("s1_id", "country").unique("s1_id")
    out = out.join(c, on="s1_id", how="left")
    p = pl.when(pl.col("country").is_in(seen)).then(pl.col("tmpl")).otherwise(pl.col("ref"))
    return out.with_columns(p.fill_null(pl.col("p_base")).alias("p2")).select("s1_id", "tgt_id", "p2")
