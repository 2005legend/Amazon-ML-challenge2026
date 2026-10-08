import numpy as np
import polars as pl
from ber.decide import best_k, expected_f_select, many_to_one, tune_threshold


def test_many_to_one_keeps_argmax():
    p = pl.DataFrame({"s1_id": ["A", "B", "B"], "tgt_id": ["S2-1", "S2-1", "S2-2"], "p": [0.7, 0.9, 0.4]})
    assert sorted(many_to_one(p, "p").select("s1_id", "tgt_id").rows()) == [("B", "S2-1"), ("B", "S2-2")]


def test_tune_threshold_excludes_low_false_positive():
    pairs = pl.DataFrame({"s1_id": ["A", "A", "B"], "tgt_id": ["S2-1", "S2-2", "S2-3"], "p": [0.9, 0.3, 0.2]})
    truth = pl.DataFrame({"s1_id": ["A"], "tgt_id": ["S2-1"]})
    tau, f = tune_threshold(pairs, truth, pl.Series(["A", "B"]), "p")
    assert 0.3 < tau <= 0.9 and f == 1.0


def test_best_k_rules():
    assert best_k(np.array([0.95, 0.9, 0.05]), 0.25) == 2
    assert best_k(np.array([0.1, 0.05]), 0.25) == 0
    assert best_k(np.array([0.6]), 0.25) == 1


def test_expected_f_select_per_group():
    pairs = pl.DataFrame({"s1_id": ["A", "A", "B"], "tgt_id": ["S2-1", "S2-2", "S2-3"], "p": [0.95, 0.02, 0.1]})
    assert expected_f_select(pairs, "p").select("s1_id", "tgt_id").rows() == [("A", "S2-1")]


def test_holdout_tune_honours_excluded_pairs(tmp_path):
    from ber.decide import holdout_tune
    from ber.folds import fold_expr
    ids = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(200)]}).with_columns(fold_expr().alias("fold"))
    a, b = ids.filter(pl.col("fold") == "H")["s1_id"].head(2).to_list()
    work, data = tmp_path / "work", tmp_path / "data"
    (work / "train" / "p2").mkdir(parents=True)
    (data / "train").mkdir(parents=True)
    pl.DataFrame({"entity_id": [a, b]}).write_parquet(work / "train" / "norm_s1.parquet")
    (data / "train" / "train_ground_truth.tsv").write_text(
        f"source1_entity_id\tmatched_entity_ids\n{a}\tS2-1\n{b}\tS2-3\n", encoding="utf-8")
    pl.DataFrame({"s1_id": [a, a, b], "tgt_id": ["S2-1", "S2-2", "S2-3"], "p2": [0.99, 0.97, 0.99],
                  "fold": ["H", "H", "H"]}).write_parquet(work / "train" / "p2" / "x.parquet")
    base = holdout_tune(work, data, "p2", "expected_f")
    excl = holdout_tune(work, data, "p2", "expected_f", exclude=pl.DataFrame({"s1_id": [a], "tgt_id": ["S2-2"]}))
    assert base["f05_H"] < 1.0 and excl["f05_H"] == 1.0
