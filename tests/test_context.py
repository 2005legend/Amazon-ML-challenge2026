import polars as pl
from ber.context import CTX_FEATS, group_features, sibling_features


def _pairs():
    return pl.DataFrame({"s1_row": [0, 0, 0, 1], "tgt_src": [2, 2, 3, 2], "tgt_row": [10, 11, 20, 10],
                         "p1": [0.9, 0.2, 0.8, 0.6]})


def test_group_features_ranks_gaps_and_margins():
    g = group_features(_pairs())
    r = {(row["s1_row"], row["tgt_src"], row["tgt_row"]): row for row in g.iter_rows(named=True)}
    assert r[(0, 2, 10)]["c_rank_s1"] == 1 and r[(0, 3, 20)]["c_rank_s1"] == 2
    assert abs(r[(0, 2, 11)]["c_gap_s1"] - 0.7) < 1e-6
    # target (2, 10) is claimed by S1 0 (0.9) and S1 1 (0.6)
    assert abs(r[(0, 2, 10)]["c_margin_tgt"] - 0.3) < 1e-6
    assert abs(r[(1, 2, 10)]["c_margin_tgt"] + 0.3) < 1e-6
    assert r[(0, 2, 11)]["c_n_tgt"] == 1 and abs(r[(0, 2, 11)]["c_margin_tgt"] - 0.2) < 1e-6


def test_sibling_features_reward_address_agreement():
    tg = pl.DataFrame({"tgt_src": [2, 2, 3, 2], "tgt_row": [10, 11, 20, 12],
                       "name_core": ["acme widgets", "zyx", "acme widgets", "other co"],
                       "name_compact": ["acmewidgets", "zyx", "acmewidgets", "otherco"],
                       "addr_norm": ["12 main st", "12 main st", "12 main st", "9 elm rd"],
                       "num_first": ["12", "12", "12", "9"]})
    p = pl.DataFrame({"s1_row": [0, 0, 0, 0], "tgt_src": [2, 2, 3, 2], "tgt_row": [10, 11, 20, 12],
                      "p1": [0.95, 0.2, 0.9, 0.1]})
    s = sibling_features(p, tg)
    alias = s.filter(pl.col("tgt_row") == 11).row(0, named=True)   # alias name, same address as siblings
    other = s.filter(pl.col("tgt_row") == 12).row(0, named=True)   # different address
    assert alias["sib_addr"] > 0.8 and other["sib_addr"] < 0.5
    assert alias["sib_nf"] > 0.8 and other["sib_nf"] == 0.0
    assert {"sib_name", "sib_addr", "sib_nf", "sib_comp", "sib_n"} <= set(CTX_FEATS)


def test_sibling_features_are_identical_when_sliced():
    tg = pl.DataFrame({"tgt_src": [2, 2, 3, 2, 3], "tgt_row": [10, 11, 20, 12, 21],
                       "name_core": ["acme widgets", "zyx", "acme widgets", "other co", "acme"],
                       "name_compact": ["acmewidgets", "zyx", "acmewidgets", "otherco", "acme"],
                       "addr_norm": ["12 main st", "12 main st", "12 main st", "9 elm rd", ""],
                       "num_first": ["12", "12", "12", "9", ""]})
    p = pl.DataFrame({"s1_row": [0, 0, 0, 0, 1, 1], "tgt_src": [2, 2, 3, 2, 3, 2], "tgt_row": [10, 11, 20, 12, 21, 12],
                      "p1": [0.95, 0.2, 0.9, 0.1, 0.8, 0.4]})
    whole = sibling_features(p, tg).sort("s1_row", "tgt_src", "tgt_row")
    sliced = sibling_features(p, tg, slice_rows=1).sort("s1_row", "tgt_src", "tgt_row")
    assert whole.equals(sliced)
