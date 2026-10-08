import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from ber.topk import pair_dot, topk
from ber.vectorize import N_QUAD, N_TRI, doc_freq, idf_from_df, name4_counts, name_counts, tfidf, word_counts


def test_char3_counts():
    m = name_counts(pl.Series(["ab", "", "abc"]))
    assert m.shape == (3, N_TRI)
    assert m[0].sum() == 2 and m[1].nnz == 0 and m[2].sum() == 3


def test_word_counts_tokens_and_empty():
    m = word_counts(pl.Series(["main st main", "", "  x  "]))
    assert m[0].sum() == 3 and m[0].nnz == 2 and m[1].nnz == 0 and m[2].sum() == 1
    assert word_counts(pl.Series([], dtype=pl.Utf8)).shape[0] == 0
    assert name_counts(pl.Series([], dtype=pl.Utf8)).shape[0] == 0


def test_tfidf_rows_are_unit_norm_or_zero():
    m = word_counts(pl.Series(["a b", "b c", ""]))
    t = tfidf(m, idf_from_df(doc_freq(m), 3))
    norms = np.sqrt(np.asarray(t.multiply(t).sum(axis=1)).ravel())
    assert np.allclose(norms, [1, 1, 0], atol=1e-6)


def test_topk_matches_bruteforce():
    Q = sp.random(40, 60, density=0.1, format="csr", random_state=1, dtype=np.float32)
    T = sp.random(120, 60, density=0.1, format="csr", random_state=2, dtype=np.float32)
    idx, val = topk(Q, T, k=5, max_df=10**9)
    dense = (Q @ T.T).toarray()
    for i in range(40):
        want = np.sort(dense[i])[::-1][:5]
        want = want[want > 0]
        assert np.allclose(val[i][: len(want)], want, atol=1e-5)
        for j, v in zip(idx[i], val[i]):
            if j >= 0:
                assert dense[i, j] == pytest.approx(v, abs=1e-5)


def test_topk_df_cap_skips_common_features():
    T = sp.csr_matrix(np.array([[1, 1], [1, 0], [1, 0]], np.float32))
    Q = sp.csr_matrix(np.array([[1, 1]], np.float32))
    idx, val = topk(Q, T, k=3, max_df=2)
    assert set(idx[0][idx[0] >= 0].tolist()) == {0} and val[0][0] == 1.0


def test_pair_dot():
    A = sp.csr_matrix(np.array([[1, 0, 2]], np.float32))
    B = sp.csr_matrix(np.array([[0, 0, 3], [1, 1, 1]], np.float32))
    assert np.allclose(pair_dot(A, B, np.array([0, 0]), np.array([0, 1])), [6, 3])


def test_topk_max_feats_uses_only_highest_weight_features():
    Q = sp.csr_matrix(np.array([[0.9, 0.1, 0.4]], np.float32))
    T = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.float32))
    idx, _ = topk(Q, T, k=3, max_df=10**9, max_feats=2)
    assert set(idx[0][idx[0] >= 0].tolist()) == {1, 2}
    idx_all, _ = topk(Q, T, k=3, max_df=10**9)
    assert set(idx_all[0][idx_all[0] >= 0].tolist()) == {0, 1, 2}


def test_topk_is_thread_safe_on_larger_inputs():
    Q = sp.random(3000, 400, density=0.03, format="csr", random_state=3, dtype=np.float32)
    T = sp.random(5000, 400, density=0.03, format="csr", random_state=4, dtype=np.float32)
    best = (Q @ T.T).toarray().max(axis=1)
    for _ in range(3):
        _, val = topk(Q, T, k=3, max_df=10**9)
        assert np.allclose(val[:, 0], best, atol=1e-5)


def test_char4_counts():
    m = name4_counts(pl.Series(["ab", "", "abcd", "abab"]))
    assert m.shape == (4, N_QUAD)
    # padded "^^ab$" gives 2 grams, "^^abcd$" gives 4, "^^abab$" gives 4 distinct
    assert m[0].sum() == 2 and m[1].nnz == 0 and m[2].sum() == 4 and m[3].nnz == 4
    assert name4_counts(pl.Series([], dtype=pl.Utf8)).shape == (0, N_QUAD)
