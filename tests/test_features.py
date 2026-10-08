import math

import polars as pl
from ber.blocking import BlockConfig, build_candidates
from ber.features import MODEL_FEATS, PAIR_FEATS, build_features, pair_features
from ber.prepare import prepare_split
from ber.prerank import apply_prerank

REC = ("acme widgets", "acmewidgets", "", "inc", "aw", False,
       "12 main st springfield il", "main springfield", "12", "12", "il", False)


def feats(a, b):
    out = pair_features(a, b)
    assert len(out) == len(PAIR_FEATS)
    return dict(zip(PAIR_FEATS, out))


def test_identical_records():
    f = feats(REC, REC)
    assert f["n_ratio"] == 1 and f["name_exact"] == 1 and f["nf_eq"] == 1
    assert f["a_tset"] == 1 and f["state_eq"] == 1 and f["legal_same"] == 1


def test_empty_address_gives_missing_not_zero():
    b = ("acme widgets", "acmewidgets", "", "", "aw", False, "", "", "", "", "", True)
    f = feats(REC, b)
    assert math.isnan(f["a_tset"]) and math.isnan(f["nf_eq"]) and math.isnan(f["state_eq"])
    assert f["addr_empty_b"] == 1


def test_acronym_and_domain_containment():
    a = ("ann marie lemus advanced textiles", "annmarielemusadvancedtextiles", "", "", "amlat", False,
         "", "", "", "", "", True)
    b = ("amlat", "amlat", "", "", "a", False, "", "", "", "", "", True)
    c = ("smithmedicalcenter", "smithmedicalcenter", "", "", "s", False, "", "", "", "", "", True)
    d = ("smith medical center", "smithmedicalcenter", "", "corp", "smc", False, "", "", "", "", "", True)
    assert feats(a, b)["acro"] == 1
    assert feats(d, c)["contain"] == 1


def test_empty_names_do_not_crash():
    e = ("", "", "", "", "", False, "", "", "", "", "", True)
    assert math.isnan(feats(e, e)["n_ratio"])


def test_build_features_tiny(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    build_candidates("train", work, BlockConfig({"n": 2, "a": 2, "na": 2}, {"n": 1, "a": 1, "na": 1}, 0.01, 5))
    apply_prerank("train", work, None)
    build_features("train", work, tiny_data, n_jobs=1)
    df = pl.concat([pl.read_parquet(f) for f in (work / "train" / "feats").glob("*.parquet")])
    assert set(MODEL_FEATS) <= set(df.columns) and {"label", "fold", "s1_id", "tgt_id"} <= set(df.columns)
    from ber.tokens import TC_FEATS
    assert set(TC_FEATS) <= set(MODEL_FEATS)
    ident = df.filter((pl.col("s1_id") == "S1-3") & (pl.col("tgt_id") == "S2-3")).row(0, named=True)
    assert ident["tc_n_ua"] == 0 and ident["tc_n_ub"] == 0
    pos = df.filter(pl.col("label") == 1)
    assert {("S1-3", "S2-3"), ("S1-1", "S2-1")} <= set(zip(pos["s1_id"].to_list(), pos["tgt_id"].to_list()))
