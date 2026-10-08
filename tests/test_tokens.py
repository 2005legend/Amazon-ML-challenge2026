import math

import polars as pl
from ber.tokens import TC_FEATS, token_change_features, token_stats, unmatched


def test_unmatched_tokens_tolerate_typos():
    assert unmatched("laboratoires club", "laboratoires groupe") == (["club"], ["groupe"])
    assert unmatched("bradley hall", "bradley heall") == ([], [])
    assert unmatched("", "x") == ([], ["x"])


def _frames():
    s1 = pl.DataFrame({"country": ["C"] * 4 + ["D"] * 2,
                       "name_core": ["alpha club", "beta club", "gamma union", "delta union", "club x", "y"]})
    tg = pl.DataFrame({"country": ["C"] * 6 + ["D"] * 2,
                       "name_core": ["alpha club", "beta club", "gamma union", "alpha holding", "beta holding",
                                     "delta holding", "club", "y"]})
    return s1, tg


def test_token_stats_are_per_country_and_mark_categories():
    st = token_stats(*_frames(), min_share=0.4)
    c = {r["t"]: r for r in st.filter(pl.col("country") == "C").iter_rows(named=True)}
    assert math.isclose(c["club"]["share1"], math.log(2 / 4))
    # holding: absent from S1, 3 of 6 targets -> strongly filler-like, never a category word
    assert c["holding"]["f"] > 0.9 and not c["holding"]["cat"]
    assert c["club"]["cat"] and c["union"]["cat"] and not c["alpha"]["cat"]
    d = {r["t"]: r for r in st.filter(pl.col("country") == "D").iter_rows(named=True)}
    assert math.isclose(d["club"]["share1"], math.log(1 / 2))


def test_token_change_features_flag_category_swaps():
    st = token_stats(*_frames(), min_share=0.4)
    pairs = pl.DataFrame({"country": ["C", "C", "C", "C"],
                          "core_a": ["alpha club", "alpha club", "alpha club", "gamma union"],
                          "core_b": ["alpha union", "alpha club", "alpha holding", "gamma union holding"]})
    out = token_change_features(pairs, st, n_jobs=1)
    assert out.columns == TC_FEATS and out.height == 4
    swap, same, filler, add = out.rows(named=True)
    assert (swap["tc_n_ua"], swap["tc_n_ub"], swap["tc_c_ua"], swap["tc_c_ub"]) == (1, 1, 1, 1)
    assert (same["tc_n_ua"], same["tc_n_ub"], same["tc_c_ua"]) == (0, 0, 0) and math.isnan(same["tc_fmax_ub"])
    assert filler["tc_c_ub"] == 0 and filler["tc_fmax_ub"] > 0.9
    assert (add["tc_n_ua"], add["tc_n_ub"], add["tc_c_ub"]) == (0, 1, 0)
