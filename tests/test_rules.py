import polars as pl
from ber.rules import category_swap, exclude_scores


def _stats():
    rows = [("F", "club", 900, -0.2), ("F", "union", 800, -0.1), ("F", "groupe", 700, 1.5),
            ("F", "cie", 600, 0.1), ("F", "compagnie", 600, -0.1), ("F", "laxmi", 900, 0.0),
            ("F", "lakshmi", 900, 0.0), ("F", "rare", 3, 0.0)]
    return {(c, t): (df1, f) for c, t, df1, f in rows}


def test_category_swap_flags_only_category_for_category():
    st = _stats()
    assert category_swap("F", "vortex club", "vortex union", st)
    assert not category_swap("F", "vortex club", "vortex groupe", st)       # filler-like word, not a category
    assert not category_swap("F", "vortex club", "vortex club", st)         # nothing swapped
    assert not category_swap("F", "vortex club", "vortex rare", st)         # rare word, not a category
    assert not category_swap("F", "vortex cie", "vortex compagnie", st)     # abbreviation (subsequence)
    assert not category_swap("F", "sri laxmi", "sri lakshmi", st)           # spelling variant (same prefix)
    assert not category_swap("F", "club", "union", st)                      # single-word names are excluded
    assert not category_swap("G", "vortex club", "vortex union", st)        # statistics are per country


def test_exclude_scores_zeroes_only_listed_pairs():
    p = pl.DataFrame({"s1_id": ["a", "a", "b"], "tgt_id": ["x", "y", "x"], "p2": [0.9, 0.8, 0.7]})
    out = exclude_scores(p, pl.DataFrame({"s1_id": ["a"], "tgt_id": ["y"]}), "p2")
    assert out["p2"].to_list() == [0.9, 0.0, 0.7] and out.columns == p.columns
