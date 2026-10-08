"""Thin LightGBM wrappers shared by the pre-ranker and both matcher stages."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config


def train_lgb(X, y, params: dict, rounds: int, valid=None, weight=None) -> lgb.Booster:
    dtrain = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True)
    sets, callbacks = [dtrain], []
    if valid is not None:
        vw = valid[2] if len(valid) > 2 else None
        sets.append(lgb.Dataset(valid[0], label=valid[1], weight=vw, reference=dtrain))
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=sets, callbacks=callbacks)


def predict(booster: lgb.Booster, X) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration or None).astype(np.float32)


def save(booster: lgb.Booster, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


# ------------------------------------------------------------- stage helpers
STAGE_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                "verbose": -1, "seed": config.SEED, "num_threads": config.N_JOBS}
KEYS = ["s1_id", "tgt_id", "s1_row", "tgt_src", "tgt_row", "country"]


def table_files(work_dir: Path, split: str, table: str) -> list[Path]:
    return sorted((Path(work_dir) / split / table).glob("*.parquet"))


def load_table(work_dir: Path, split: str, table: str, columns: list[str], where=None) -> pl.DataFrame:
    lf = pl.scan_parquet(table_files(work_dir, split, table))
    if where is not None:
        lf = lf.filter(where)
    return lf.select(columns).collect()


def neg_sample_expr(rate: float) -> pl.Expr:
    h = pl.concat_str("s1_id", "tgt_id").hash(seed=config.SEED) % 1000
    return (pl.col("label") == 1) | (h < int(rate * 1000))


def fit_fold_models(work_dir: Path, table: str, feats: list[str], name: str,
                    neg_rate: float = 0.5, rounds: int = 3000) -> dict:
    models = {}
    for fold in ("A", "B"):
        df = load_table(work_dir, "train", table, feats + ["label", "s1_id"],
                        (pl.col("fold") == fold) & neg_sample_expr(neg_rate))
        X = df.select(feats).cast(pl.Float32).to_numpy()
        y = df["label"].to_numpy()
        w = np.where(y == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
        val = (df["s1_id"].str.slice(3).cast(pl.Int64) % 20 == 0).to_numpy()
        b = train_lgb(X[~val], y[~val], STAGE_PARAMS, rounds, valid=(X[val], y[val], w[val]), weight=w[~val])
        save(b, Path(work_dir) / "models" / f"{name}_{fold}.txt")
        models[fold] = b
    return models


def predict_stage(work_dir: Path, split: str, table: str, feats: list[str], models: dict, out_name: str) -> None:
    out_dir = Path(work_dir) / split / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["label", "fold"] if split == "train" else []
    for f in table_files(work_dir, split, table):
        df = pl.read_parquet(f)
        X = df.select(feats).cast(pl.Float32).to_numpy()
        pa, pb = predict(models["A"], X), predict(models["B"], X)
        if split == "train":
            fold = df["fold"].to_numpy()
            p = np.where(fold == "A", pb, np.where(fold == "B", pa, (pa + pb) / 2))
        else:
            p = (pa + pb) / 2
        df.select(KEYS + extra).with_columns(pl.Series(out_name, p, dtype=pl.Float32)).write_parquet(out_dir / f.name)
