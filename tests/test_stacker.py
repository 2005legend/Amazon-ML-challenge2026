import numpy as np
import polars as pl

from ber.stacker import oof_scores


def test_oof_scores_come_from_the_model_not_trained_on_the_s1():
    rng = np.random.default_rng(0)
    n = 4000
    s1 = [f"s{i // 4}" for i in range(n)]
    x = pl.DataFrame({"s1_id": s1, "tgt_id": [f"t{i}" for i in range(n)], "f": rng.normal(size=n)})
    x = x.with_columns((pl.col("f") + 0.3 * pl.Series(rng.normal(size=n)) > 0).cast(pl.Int8).alias("label"))
    fold = (x["s1_id"].hash(3) % 2).to_numpy()
    o = oof_scores(x, ["f"], rounds=30)
    assert o.columns == ["s1_id", "tgt_id", "p2"] and o.height == n
    assert np.corrcoef(o["p2"].to_numpy(), x["f"].to_numpy())[0, 1] > 0.8
    # every S1's pairs sit in one fold, and both folds are scored
    assert x.with_columns(pl.Series("k", fold)).group_by("s1_id").agg(pl.col("k").n_unique())["k"].max() == 1
    assert set(fold.tolist()) == {0, 1}
