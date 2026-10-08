import polars as pl
from ber.blocking import BlockConfig, build_candidates, load_candidates
from ber.prepare import prepare_split

SMALL = BlockConfig(k_fwd={"n": 2, "a": 2, "na": 2}, k_rev={"n": 1, "a": 1, "na": 1},
                    max_df_frac=0.01, exact_max_group=5)


def test_blocking_finds_true_pairs(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    build_candidates("train", work, SMALL)
    c = load_candidates(work, "train")
    got = set(zip(c["s1_id"].to_list(), c["tgt_id"].to_list()))
    assert {("S1-1", "S2-1"), ("S1-2", "S2-2"), ("S1-2", "S3-2"), ("S1-3", "S2-3"), ("S1-3", "S3-1")} <= got
    # blocking never crosses countries: US S1-3 is never paired with Indian S2-1/S2-2
    cross = c.filter((pl.col("s1_id") == "S1-3") & pl.col("tgt_id").is_in(["S2-1", "S2-2"]))
    assert cross.height == 0
    assert c["cos_na"].min() >= 0 and c["via"].min() > 0


def test_countries_come_from_data(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    prepare_split("test", tiny_data, work, n_jobs=1)
    build_candidates("test", work, SMALL)
    c = load_candidates(work, "test")
    assert ("S1-5", "S2-5") in set(zip(c["s1_id"].to_list(), c["tgt_id"].to_list()))
    assert "France" in c["country"].unique().to_list()


def test_reverse_views_and_per_view_limits_follow_config(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    cfg = BlockConfig(k_fwd={"n": 2, "a": 2, "na": 2}, k_rev={"na": 1}, max_df_frac=0.01, exact_max_group=5,
                      df_frac={"n": 0.003, "a": 0.003, "na": 0.01}, max_feats={"na": 10})
    build_candidates("train", work, cfg)
    c = load_candidates(work, "train")
    assert c.filter((pl.col("via") & (8 | 16)) != 0).height == 0
    assert ("S1-3", "S2-3") in set(zip(c["s1_id"].to_list(), c["tgt_id"].to_list()))


def test_recall_report_full_and_subset(tiny_data, tmp_path):
    from ber.blocking import recall_report
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    build_candidates("train", work, SMALL)
    rep = recall_report(work, tiny_data)
    assert rep["pair_recall"] == 1.0 and rep["ceiling_f05"] == 1.0 and rep["recall_US"] == 1.0
    sub = recall_report(work, tiny_data, s1_subset=pl.Series(["S1-3", "S1-4"]))
    assert sub["pair_recall"] == 1.0 and sub["n_s1"] == 2
