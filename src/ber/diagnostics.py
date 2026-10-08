"""Holdout breakdowns, orphan stress test, test-set country sanity and error samples."""
from pathlib import Path

import polars as pl

from .context import CTX_FEATS, SIB_FEATS, group_features
from .decide import decide, many_to_one, tune_threshold
from .evaluate import macro_fbeta, per_s1_scores
from .features import MODEL_FEATS
from .folds import fold_expr
from .io import read_ground_truth
from .model import load, load_table, predict
from .prepare import norm_path


def drop_ids(ids: pl.Series, frac: float) -> pl.Series:
    h = (ids.str.slice(3).cast(pl.Int64) % 1_000_003) * 40503 % 65536 % 1000
    return ids.filter(h < int(frac * 1000))


def country_report(pairs: pl.DataFrame, s1: pl.DataFrame, decision: dict) -> pl.DataFrame:
    score = decision["score"]
    pred = decide(pairs, score, decision)
    n = pred.group_by("s1_id").len("n_matches")
    best = pairs.group_by("s1_id").agg(pl.col(score).max().alias("best"))
    per = (s1.join(n, on="s1_id", how="left").join(best, on="s1_id", how="left")
             .with_columns(pl.col("n_matches").fill_null(0)))
    return per.group_by("country").agg(
        pl.len().alias("s1"), (pl.col("n_matches") > 0).mean().alias("share_matched"),
        pl.col("n_matches").mean().alias("mean_matches"), pl.col("best").median().alias("median_best_score"),
        pl.col("best").is_null().mean().alias("share_no_candidates"))


def _holdout(work_dir: Path, data_dir: Path):
    s1 = (pl.read_parquet(norm_path(work_dir, "train", 1), columns=["entity_id", "country"])
            .rename({"entity_id": "s1_id"}).with_columns(fold_expr().alias("fold")))
    h = s1.filter(pl.col("fold") == "H")
    gt = read_ground_truth(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return h, gt.filter(pl.col("s1_id").is_in(h["s1_id"].implode()))


def _stressed_pairs(work_dir: Path, score: str, dropped: pl.Series) -> pl.DataFrame:
    not_dropped = ~pl.col("s1_id").is_in(dropped.implode())
    if score == "p1":
        return load_table(work_dir, "train", "p1", ["s1_id", "tgt_id", "p1", "fold"]).filter(not_dropped)
    keys = ["s1_row", "tgt_src", "tgt_row"]
    p = load_table(work_dir, "train", "p1", keys + ["s1_id", "p1"]).filter(not_dropped)
    g = group_features(p.select(keys + ["p1"]))
    cols = ["s1_id", "tgt_id", "fold"] + keys + [c for c in MODEL_FEATS + SIB_FEATS if c not in keys]
    h = load_table(work_dir, "train", "feats2", cols, pl.col("fold") == "H").filter(not_dropped).join(g, on=keys)
    X = h.select(MODEL_FEATS + CTX_FEATS).cast(pl.Float32).to_numpy()
    ma, mb = (load(Path(work_dir) / "models" / f"m2_{k}.txt") for k in ("A", "B"))
    h = h.with_columns(pl.Series("p2", (predict(ma, X) + predict(mb, X)) / 2))
    others = load_table(work_dir, "train", "p2", ["s1_id", "tgt_id", "p2", "fold"], pl.col("fold") != "H")
    return pl.concat([others, h.select("s1_id", "tgt_id", "p2", "fold")], how="vertical_relaxed")


def run_diagnostics(work_dir: Path, data_dir: Path, decision: dict) -> dict:
    score = decision["score"]
    h, truth = _holdout(work_dir, data_dir)
    pairs = load_table(work_dir, "train", score, ["s1_id", "tgt_id", score, "fold"])
    pred = decide(pairs, score, decision).filter(pl.col("fold") == "H")
    per = (per_s1_scores(pred, truth, h["s1_id"]).join(h.select("s1_id", "country"), on="s1_id")
             .with_columns(pl.col("n_true").clip(0, 6).alias("n_true_bucket")))
    out = {"by_country": per.group_by("country").agg(pl.len(), pl.col("f").mean()).sort("country"),
           "by_n_true": per.group_by("n_true_bucket").agg(pl.len(), pl.col("f").mean()).sort("n_true_bucket")}
    dropped = drop_ids(h["s1_id"], 0.2)
    keep_ids = h.filter(~pl.col("s1_id").is_in(dropped.implode()))["s1_id"]
    truth_keep = truth.filter(pl.col("s1_id").is_in(keep_ids.implode()))
    stressed = _stressed_pairs(work_dir, score, dropped)
    in_keep = pl.col("s1_id").is_in(keep_ids.implode())
    stress = {"f05_fixed_rule": macro_fbeta(decide(stressed, score, decision).filter(in_keep), truth_keep, keep_ids)}
    if decision["rule"] == "threshold":
        stress["tau_retuned"], stress["f05_retuned"] = tune_threshold(
            many_to_one(stressed, score).filter(in_keep), truth_keep, keep_ids, score)
    out["orphan_stress"] = stress
    test_pairs = load_table(work_dir, "test", score, ["s1_id", "tgt_id", score])
    s1_test = pl.read_parquet(norm_path(work_dir, "test", 1), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    out["test_countries"] = country_report(test_pairs, s1_test, decision).sort("country")
    out["h_countries"] = country_report(pairs.filter(pl.col("fold") == "H"), h.select("s1_id", "country"),
                                        decision).sort("country")
    raw = {s: pl.read_parquet(norm_path(work_dir, "train", s), columns=["entity_id", "name", "address"])
           for s in (1, 2, 3)}
    s1raw = raw[1].rename({"entity_id": "s1_id", "name": "s1_name", "address": "s1_address"})
    tgraw = pl.concat([raw[2], raw[3]]).rename({"entity_id": "tgt_id", "name": "tgt_name", "address": "tgt_address"})
    fp = pred.join(truth, on=["s1_id", "tgt_id"], how="anti").head(30)
    fn = truth.join(pred, on=["s1_id", "tgt_id"], how="anti").head(30)
    for name, df in (("errors_fp.tsv", fp), ("errors_fn.tsv", fn)):
        (df.select("s1_id", "tgt_id").join(s1raw, on="s1_id").join(tgraw, on="tgt_id")
           .write_csv(Path(work_dir) / name, separator="\t"))
    for k, v in out.items():
        print(f"== {k}\n{v}")
    return out
