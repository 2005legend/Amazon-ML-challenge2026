import numpy as np
import polars as pl
from ber.model import fit_fold_models, load_table, predict_stage


def _fake_table(root, split):
    rng = np.random.default_rng(0)
    n = 6000
    s1 = [f"S1-{i // 3}" for i in range(n)]
    x = rng.random(n).astype(np.float32)
    df = pl.DataFrame({"s1_id": s1, "tgt_id": [f"S2-{i}" for i in range(n)], "s1_row": np.arange(n) // 3,
                       "tgt_src": np.full(n, 2, np.int8), "tgt_row": np.arange(n), "country": ["US"] * n,
                       "x": x, "noise": rng.random(n).astype(np.float32)})
    if split == "train":
        from ber.folds import fold_expr
        df = df.with_columns(pl.Series("label", (x > 0.6).astype(np.int8)), fold_expr().alias("fold"))
    path = root / split / "feats"
    path.mkdir(parents=True)
    df.write_parquet(path / "US_s2_000.parquet")


def test_fit_and_predict_out_of_fold(tmp_path):
    for split in ("train", "test"):
        _fake_table(tmp_path, split)
    models = fit_fold_models(tmp_path, "feats", ["x", "noise"], name="m1", neg_rate=0.5, rounds=100)
    assert set(models) == {"A", "B"} and (tmp_path / "models" / "m1_A.txt").exists()
    predict_stage(tmp_path, "train", "feats", ["x", "noise"], models, "p1")
    predict_stage(tmp_path, "test", "feats", ["x", "noise"], models, "p1")
    tr = load_table(tmp_path, "train", "p1", ["label", "p1", "fold"])
    auc_proxy = tr.filter(pl.col("label") == 1)["p1"].mean() - tr.filter(pl.col("label") == 0)["p1"].mean()
    assert auc_proxy > 0.8
    te = load_table(tmp_path, "test", "p1", ["s1_id", "tgt_id", "p1"])
    assert te.height == 6000 and te["p1"].is_between(0, 1).all()
