"""Byte views of string columns and sparse TF-IDF matrices built by numba kernels."""
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from numba import njit

N_SYM = 38            # 0 boundary, 1-26 a-z, 27-36 0-9, 37 anything else
N_TRI = N_SYM ** 3
N_QUAD = N_SYM ** 4
N_WORD = 1 << 20


def bytes_offsets(s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = s.fill_null("").to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    arr = pc.cast(arr, pa.large_string())
    bufs = arr.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64)[arr.offset: arr.offset + len(arr) + 1].copy()
    data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
    return data, offsets


@njit(cache=True)
def _sym(b):
    if 97 <= b <= 122:
        return b - 96
    if 48 <= b <= 57:
        return b - 21
    return 37


@njit(cache=True)
def _char3_coo(data, offsets):
    n = len(offsets) - 1
    total = 0
    for i in range(n):
        total += offsets[i + 1] - offsets[i]
    rows = np.empty(total, np.int32)
    cols = np.empty(total, np.int32)
    k = 0
    for i in range(n):
        s, length = offsets[i], offsets[i + 1] - offsets[i]
        if length == 0:
            continue
        p2, p1 = 0, _sym(data[s])
        for j in range(1, length + 1):
            cur = _sym(data[s + j]) if j < length else 0
            rows[k] = i
            cols[k] = (p2 * N_SYM + p1) * N_SYM + cur
            k += 1
            p2, p1 = p1, cur
    return rows, cols


@njit(cache=True)
def _char4_coo(data, offsets):
    """Char 4-grams over the string padded as [0, 0, chars..., 0]: one gram per character."""
    n = len(offsets) - 1
    total = 0
    for i in range(n):
        total += offsets[i + 1] - offsets[i]
    rows = np.empty(total, np.int32)
    cols = np.empty(total, np.int32)
    k = 0
    for i in range(n):
        s, length = offsets[i], offsets[i + 1] - offsets[i]
        if length == 0:
            continue
        a, b, c = 0, 0, _sym(data[s])
        for j in range(1, length + 1):
            d = _sym(data[s + j]) if j < length else 0
            rows[k] = i
            cols[k] = ((a * N_SYM + b) * N_SYM + c) * N_SYM + d
            k += 1
            a, b, c = b, c, d
    return rows, cols


@njit(cache=True)
def _word_coo(data, offsets, n_buckets):
    n = len(offsets) - 1
    total = 0
    for i in range(n):
        inside = False
        for p in range(offsets[i], offsets[i + 1]):
            if data[p] == 32:
                inside = False
            elif not inside:
                inside = True
                total += 1
    rows = np.empty(total, np.int32)
    cols = np.empty(total, np.int32)
    k = 0
    for i in range(n):
        h, inside = 2166136261, False
        for p in range(offsets[i], offsets[i + 1] + 1):
            b = 32 if p == offsets[i + 1] else data[p]
            if b == 32:
                if inside:
                    rows[k] = i
                    cols[k] = h % n_buckets
                    k += 1
                h, inside = 2166136261, False
            else:
                inside = True
                h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return rows, cols


def _counts(rows, cols, n_rows, n_cols) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_rows, n_cols))
    m.sum_duplicates()
    m.sort_indices()
    return m


def name_counts(s: pl.Series) -> sp.csr_matrix:
    if len(s) == 0:
        return sp.csr_matrix((0, N_TRI), dtype=np.float32)
    data, off = bytes_offsets(s)
    return _counts(*_char3_coo(data, off), len(s), N_TRI)


def name4_counts(s: pl.Series) -> sp.csr_matrix:
    if len(s) == 0:
        return sp.csr_matrix((0, N_QUAD), dtype=np.float32)
    data, off = bytes_offsets(s)
    return _counts(*_char4_coo(data, off), len(s), N_QUAD)


def word_counts(s: pl.Series) -> sp.csr_matrix:
    if len(s) == 0:
        return sp.csr_matrix((0, N_WORD), dtype=np.float32)
    data, off = bytes_offsets(s)
    return _counts(*_word_coo(data, off, N_WORD), len(s), N_WORD)


def doc_freq(m: sp.csr_matrix) -> np.ndarray:
    return np.bincount(m.indices, minlength=m.shape[1]).astype(np.int64)


def idf_from_df(df: np.ndarray, n_docs: int) -> np.ndarray:
    return (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)


@njit(cache=True)
def _l2_rows(indptr, data):
    for i in range(len(indptr) - 1):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += data[p] * data[p]
        if acc > 0:
            inv = 1.0 / np.sqrt(acc)
            for p in range(indptr[i], indptr[i + 1]):
                data[p] *= inv


def tfidf(m: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    out = m.copy()
    out.data = ((1.0 + np.log(out.data)) * idf[out.indices]).astype(np.float32)
    _l2_rows(out.indptr, out.data)
    return out
