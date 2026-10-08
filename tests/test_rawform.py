import math

import polars as pl

from ber.rawform import raw_pair, raw_record, score_shape, transformation


def test_raw_pair_keeps_legal_spelling_as_written():
    a, b = raw_record("Earthstone & Sons Private Limited"), raw_record("EARTHSTONE & SONS PVT LTD")
    r = dict(zip(["cf_eq", "alnum_eq", "reordered", "order", "case_same", "paren_eq", "paren_n", "legal_eq", "legal_n"], raw_pair(a, b)))
    assert r["legal_eq"] is False and r["legal_n"] == 2
    assert r["case_same"] is False
    assert raw_pair(a, raw_record("earthstone & sons private limited"))[0] is True  # casefold equal


def test_transformation_types_edits_and_templates():
    cnt, subs, legal, drop, add, first, last, _ = transformation("Café Rouge LLC", "Cafe Rouge Group Inc")
    assert cnt["accent"] == 1
    assert legal == "llc>inc"
    assert add == ["group"] and drop == []
    assert first == 1.0 and last == 0.0  # the S1's last word (its legal form) is not kept


def test_transformation_detects_doubled_letter_and_inner_swap():
    cnt, *_ = transformation("Mira Stores", "Mirra Stores")
    assert cnt["dup"] == 1
    cnt, *_ = transformation("Mira Stores", "Mria Stores")
    assert cnt["inner"] == 1  # the edit script encodes an adjacent swap as delete + insert, not as two substitutions


def test_score_shape_is_leave_one_out():
    p = pl.DataFrame({"s1_id": ["a", "a", "a"], "tgt_id": ["S2-1", "S3-2", "S2-3"], "p2": [0.95, 0.6, 0.1]})
    g = score_shape(p).sort("tgt_id")
    row = g.filter(pl.col("tgt_id") == "S2-1").row(0, named=True)
    assert row["g5_ncand"] == 2 and row["g5_n05"] == 1 and row["g5_n09"] == 0
    assert math.isclose(row["g5_pmax_other"], 0.6, rel_tol=1e-6)
    other = g.filter(pl.col("tgt_id") == "S3-2").row(0, named=True)
    assert other["g5_n09"] == 1 and other["g5_n09_other_src"] == 1 and math.isclose(other["g5_pmax_other"], 0.95, rel_tol=1e-6)
