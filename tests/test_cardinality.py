import numpy as np
import polars as pl

from ber.cardinality import KC, choose, design, pb_matrix
from ber.decide import expected_f_select


def _groups(ps):
    rows = [(f"s{g}", f"t{g}_{i}", float(v)) for g, q in enumerate(ps) for i, v in enumerate(sorted(q, reverse=True))]
    a = pl.DataFrame(rows, schema=["s1_id", "tgt_id", "p2"], orient="row")
    gid = a.select(pl.col("s1_id").rle_id())["s1_id"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(gid)) + 1, [len(gid)]]).astype(np.int64)
    return a, starts, a["p2"].to_numpy().astype(np.float64)


def test_poisson_binomial_prior_reproduces_the_expected_f_decision():
    rng = np.random.default_rng(3)
    ps = [rng.beta(0.6, 0.6, size=rng.integers(1, 13)) for _ in range(300)]
    a, starts, p = _groups(ps)
    pb = pb_matrix(starts, p)
    got = a.filter(pl.Series(choose(starts, p, pb / pb.sum(1, keepdims=True))))
    ref = expected_f_select(a, "p2")
    assert got.sort("tgt_id")["tgt_id"].to_list() == ref.sort("tgt_id")["tgt_id"].to_list()


def test_pb_matrix_rows_are_distributions_over_the_count():
    _, starts, p = _groups([[0.5, 0.5], [0.9]])
    pb = pb_matrix(starts, p)
    assert pb.shape == (2, KC)
    assert np.allclose(pb.sum(1), 1.0)
    assert np.allclose(pb[0, :3], [0.25, 0.5, 0.25])


def test_choose_follows_a_confident_cardinality_prior():
    # scores 0.9 / 0.3: if exactly one is true it is almost surely the first (with 0.6 / 0.6 taking both would win)
    _, starts, p = _groups([[0.9, 0.3]])
    for k in (0, 1, 2):
        prior = np.zeros((1, KC)); prior[0, k] = 1.0
        assert choose(starts, p, prior).sum() == k


def test_design_counts_candidate_types_per_s1():
    a = pl.DataFrame({"s1_id": ["a", "a", "b"], "tgt_id": ["S2-1", "S3-2", "S2-3"], "p2": [0.9, 0.3, 0.6],
                      "ar": ["noaddr", "same", "noaddr"], "nr": ["exact", "alias", "exact"], "label": [1, 0, 1]})
    s1 = pl.DataFrame({"s1_id": ["a", "b"], "country": ["US", "India"], "name": ["Acme Tools Inc", "Ravi"], "addr_empty": [False, True]})
    f, X, starts, p, pb = design(a, s1)
    assert f["s1_id"].to_list() == ["a", "b"]
    assert f["n_noaddr"].to_list() == [1, 1] and f["K"].to_list() == [1, 1] and f["n_s2"].to_list() == [1, 1]
    assert X.shape == (2, 12 + 14) and starts.tolist() == [0, 2, 3]
    assert np.allclose(X[0, :3], [0.9, 0.3, 0.0])
