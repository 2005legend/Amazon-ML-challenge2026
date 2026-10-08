import polars as pl
from ber.blocking import BlockConfig, build_candidates
from ber.prepare import prepare_split
from ber.prerank import PRE_FEATS, apply_prerank, prerank_features, prune


def test_prerank_features_rank_within_groups():
    c = pl.DataFrame({"s1_row": [0, 0, 1], "tgt_row": [5, 6, 5], "cos_n": [0.2, 0.9, 0.5],
                      "cos_a": [0.1, 0.1, 0.1], "cos_na": [0.3, 0.8, 0.4], "via": [1, 1, 1]})
    f = prerank_features(c)
    assert f["rk1_n"].to_list() == [2, 1, 1]
    assert f["rkt_n"].to_list() == [2, 1, 1]
    assert [round(x, 6) for x in f["gap1_na"].to_list()] == [0.5, 0.0, 0.0]
    assert set(PRE_FEATS) <= set(f.columns)


def test_prune_floor_and_cap():
    c = pl.DataFrame({"s1_row": [0, 0, 0, 1], "tgt_row": [1, 2, 3, 1], "p_pre": [0.9, 0.5, 0.001, 0.3]})
    kept = prune(c, floor=0.002, cap=1)
    assert sorted(zip(kept["s1_row"].to_list(), kept["tgt_row"].to_list())) == [(0, 1), (1, 1)]


def test_apply_without_model_keeps_everything(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    build_candidates("train", work, BlockConfig({"n": 2, "a": 2, "na": 2}, {"n": 1, "a": 1, "na": 1}, 0.01, 5))
    apply_prerank("train", work, None, floor=0.002, cap=12)
    raw = sum(pl.read_parquet(f).height for f in (work / "train" / "cands_raw").glob("*.parquet"))
    kept = sum(pl.read_parquet(f).height for f in (work / "train" / "cands").glob("*.parquet"))
    assert raw == kept > 0


def test_train_prerank_runs_on_int_keys(tiny_data, tmp_path):
    import numpy as np
    from ber.prerank import train_prerank
    from ber.model import predict
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    build_candidates("train", work, BlockConfig({"n": 2, "a": 2, "na": 2}, {"n": 1, "a": 1, "na": 1}, 0.01, 5))
    b = train_prerank(work, tiny_data, sample_frac=1.0, folds=("A", "B", "H"), rounds=5)
    p = predict(b, np.zeros((2, len(PRE_FEATS)), np.float32))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
