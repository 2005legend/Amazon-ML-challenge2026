"""Parallel sparse retrieval kernels: per-query top-k dot products and dot products of given pairs."""
import numpy as np
import scipy.sparse as sp
from numba import get_num_threads, get_thread_id, njit, prange


@njit(parallel=True, cache=False)  # thread-id intrinsics cannot be disk-cached
def _topk(q_indptr, q_indices, q_data, t_indptr, t_indices, t_data, n_targets, k, max_df, max_feats,
          max_q_nnz):
    nq = len(q_indptr) - 1
    out_idx = np.full((nq, k), -1, np.int32)
    out_val = np.zeros((nq, k), np.float32)
    if nq == 0:
        return out_idx, out_val
    # Per-thread scratch allocated once, outside the parallel loop: numba hoists allocations made
    # inside a prange body and shares them between threads, which corrupts the accumulators.
    n_threads = get_num_threads()
    acc = np.zeros((n_threads, n_targets), np.float32)
    touched = np.empty((n_threads, n_targets), np.int32)
    elig = np.empty((n_threads, max(max_q_nnz, 1)), np.int64)
    n_chunks = min(nq, 32 * n_threads)
    size = (nq + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        tid = get_thread_id()
        for q in range(c * size, min(nq, (c + 1) * size)):
            ne = 0
            for p in range(q_indptr[q], q_indptr[q + 1]):
                f = q_indices[p]
                if t_indptr[f + 1] - t_indptr[f] <= max_df:
                    elig[tid, ne] = p
                    ne += 1
            if ne > max_feats:
                for j in range(max_feats):
                    best = j
                    for z in range(j + 1, ne):
                        if q_data[elig[tid, z]] > q_data[elig[tid, best]]:
                            best = z
                    tmp = elig[tid, j]
                    elig[tid, j] = elig[tid, best]
                    elig[tid, best] = tmp
                ne = max_feats
            nt = 0
            for i in range(ne):
                p = elig[tid, i]
                f = q_indices[p]
                w = q_data[p]
                for r in range(t_indptr[f], t_indptr[f + 1]):
                    t = t_indices[r]
                    if acc[tid, t] == 0.0:
                        touched[tid, nt] = t
                        nt += 1
                    acc[tid, t] += w * t_data[r]
            filled, minpos = 0, 0
            for u in range(nt):
                t = touched[tid, u]
                v = acc[tid, t]
                acc[tid, t] = 0.0
                if filled < k:
                    out_idx[q, filled] = t
                    out_val[q, filled] = v
                    filled += 1
                    if filled == k:
                        minpos = 0
                        for z in range(1, k):
                            if out_val[q, z] < out_val[q, minpos]:
                                minpos = z
                elif v > out_val[q, minpos]:
                    out_idx[q, minpos] = t
                    out_val[q, minpos] = v
                    minpos = 0
                    for z in range(1, k):
                        if out_val[q, z] < out_val[q, minpos]:
                            minpos = z
    return out_idx, out_val


def topk(Q: sp.csr_matrix, T: sp.csr_matrix, k: int, max_df: int, max_feats: int = 1 << 30):
    """Top-k rows of T by dot product with each row of Q.

    Features present in more than max_df rows of T are skipped, and each query uses at most its
    max_feats highest-weight (rarest) remaining features, so scores are partial sums used only for
    ranking; exact cosines are recomputed later with pair_dot.
    """
    TT = T.T.tocsr()
    max_q_nnz = int(np.diff(Q.indptr).max()) if Q.shape[0] else 1
    idx, val = _topk(Q.indptr, Q.indices, Q.data, TT.indptr, TT.indices, TT.data, T.shape[0], k, max_df,
                     max_feats, max_q_nnz)
    order = np.argsort(-val, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(val, order, axis=1)


@njit(parallel=True, cache=True)
def _pair_dot(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, ia, ib):
    out = np.zeros(len(ia), np.float32)
    for i in prange(len(ia)):
        p, pe = a_indptr[ia[i]], a_indptr[ia[i] + 1]
        r, re_ = b_indptr[ib[i]], b_indptr[ib[i] + 1]
        s = 0.0
        while p < pe and r < re_:
            if a_indices[p] == b_indices[r]:
                s += a_data[p] * b_data[r]
                p += 1
                r += 1
            elif a_indices[p] < b_indices[r]:
                p += 1
            else:
                r += 1
        out[i] = s
    return out


def pair_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    A.sort_indices()
    B.sort_indices()
    return _pair_dot(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data,
                     np.asarray(ia, np.int64), np.asarray(ib, np.int64))
