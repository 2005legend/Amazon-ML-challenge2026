import polars as pl
from ber.diagnostics import country_report, drop_ids


def test_drop_ids_fraction_is_close():
    ids = pl.Series([f"S1-{i}" for i in range(20000)])
    d = drop_ids(ids, 0.2)
    assert abs(len(d) / 20000 - 0.2) < 0.01
    assert drop_ids(ids, 0.2).to_list() == d.to_list()


def test_country_report_counts_matches_and_missing_candidates():
    pairs = pl.DataFrame({"s1_id": ["A", "A", "B"], "tgt_id": ["S2-1", "S3-1", "S2-2"],
                          "p1": [0.9, 0.8, 0.2]})
    s1 = pl.DataFrame({"s1_id": ["A", "B", "C"], "country": ["US", "France", "France"]})
    rep = country_report(pairs, s1, {"score": "p1", "rule": "threshold", "tau": 0.5})
    us = rep.filter(pl.col("country") == "US").row(0, named=True)
    fr = rep.filter(pl.col("country") == "France").row(0, named=True)
    assert us["share_matched"] == 1.0 and us["mean_matches"] == 2.0
    assert fr["s1"] == 2 and fr["share_matched"] == 0.0 and fr["share_no_candidates"] == 0.5


def test_stressed_pairs_rescores_holdout_with_stage2_models(tmp_path):
    import numpy as np
    from ber.context import CTX_FEATS, SIB_FEATS
    from ber.diagnostics import _stressed_pairs
    from ber.features import MODEL_FEATS
    from ber.model import save, train_lgb

    rng = np.random.default_rng(0)
    n = 40
    base = pl.DataFrame({"s1_id": [f"S1-{i // 2}" for i in range(n)], "tgt_id": [f"S2-{i}" for i in range(n)],
                         "s1_row": [i // 2 for i in range(n)], "tgt_src": [2] * n, "tgt_row": list(range(n)),
                         "fold": ["H" if i < 20 else "A" for i in range(n)]}).with_columns(pl.col("tgt_src").cast(pl.Int8))
    feats = [c for c in MODEL_FEATS + SIB_FEATS if c not in base.columns]
    f2 = base.hstack([pl.Series(c, rng.random(n), dtype=pl.Float32) for c in feats])
    p1 = base.with_columns(pl.Series("p1", rng.random(n), dtype=pl.Float32))
    p2 = base.with_columns(pl.Series("p2", rng.random(n), dtype=pl.Float32))
    for t, df in (("feats2", f2), ("p1", p1), ("p2", p2)):
        (tmp_path / "train" / t).mkdir(parents=True)
        df.write_parquet(tmp_path / "train" / t / "US_s2.parquet")
    X = rng.random((200, len(MODEL_FEATS + CTX_FEATS))).astype(np.float32)
    m = train_lgb(X, (X[:, 0] > 0.5).astype(int), {"objective": "binary", "verbose": -1}, 5)
    for k in ("A", "B"):
        save(m, tmp_path / "models" / f"m2_{k}.txt")
    out = _stressed_pairs(tmp_path, "p2", pl.Series(["S1-0"]))
    assert out.height == n - 2 and "S1-0" not in out["s1_id"].to_list()
    assert out.filter(pl.col("fold") == "H").height == 18 and out["p2"].is_between(0, 1).all()
