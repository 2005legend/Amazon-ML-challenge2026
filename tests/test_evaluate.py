import polars as pl
import pytest
from ber.evaluate import macro_fbeta, blocking_ceiling

EMPTY = pl.DataFrame(schema={"s1_id": pl.Utf8, "tgt_id": pl.Utf8})


def test_statement_example():
    pred = pl.DataFrame({"s1_id": ["S1-1"] * 3, "tgt_id": ["S2-47", "S2-193", "S3-812"]})
    truth = pl.DataFrame({"s1_id": ["S1-1"] * 2, "tgt_id": ["S2-47", "S3-812"]})
    assert macro_fbeta(pred, truth, pl.Series(["S1-1"])) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    truth = pl.DataFrame({"s1_id": ["A"], "tgt_id": ["S2-1"]})
    pred = pl.DataFrame({"s1_id": ["B"], "tgt_id": ["S2-9"]})
    # A missed (0), B singleton with a prediction (0), C singleton predicted empty (1)
    assert macro_fbeta(pred, truth, pl.Series(["A", "B", "C"])) == pytest.approx(1 / 3)
    assert macro_fbeta(EMPTY, EMPTY, pl.Series(["C"])) == 1.0


def test_blocking_ceiling_counts_found_fraction():
    truth = pl.DataFrame({"s1_id": ["A", "A"], "tgt_id": ["S2-1", "S3-2"]})
    cands = pl.DataFrame({"s1_id": ["A", "A"], "tgt_id": ["S2-1", "S2-9"]})
    # perfect model on candidates predicts {S2-1}: P=1, R=0.5 -> 1.25*0.5/(0.25+0.5)
    assert blocking_ceiling(cands, truth, pl.Series(["A"])) == pytest.approx(0.8333333, abs=1e-6)
