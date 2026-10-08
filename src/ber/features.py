"""Pairwise similarity features for pruned candidate pairs (chunked, process pool)."""
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .folds import fold_expr
from .io import read_ground_truth
from .prepare import norm_path
from .prerank import PRE_FEATS
from .tokens import TC_FEATS, augment_features
from .vectorize import N_WORD, doc_freq, idf_from_df, word_counts

FIELDS = ["name_core", "name_compact", "name_parts", "legal", "initials", "name_indic",
          "addr_norm", "addr_words", "addr_nums", "num_first", "state", "addr_empty"]
PAIR_FEATS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "c_ratio", "c_partial", "c_jw", "c_prefix",
              "tok_jacc", "tok_soft_a", "tok_soft_b", "idf_overlap", "acro", "contain", "parts_best",
              "legal_same", "legal_conflict", "legal_missing", "n_tok_a", "n_tok_b", "len_ratio",
              "name_indic_b", "first_tok_eq", "name_exact", "name_empty_b",
              "a_tset", "a_partial", "a_ratio", "w_tset", "w_jacc", "num_jacc", "num_overlap", "num_b_in_a",
              "nf_eq", "nf_lev", "nf_contain", "nf_logdiff", "state_eq", "addr_empty_a", "addr_empty_b",
              "n_addr_tok_a", "n_addr_tok_b"]
FREQ_FEATS = ["s1_freq_a", "s1_freq_b", "tgt_freq_b"]
MODEL_FEATS = PRE_FEATS + ["p_pre", "tgt_src"] + PAIR_FEATS + FREQ_FEATS + TC_FEATS
NAN = float("nan")
_IDF = None


def fnv_bucket(tok: str) -> int:
    h = 2166136261
    for b in tok.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h % N_WORD


def _init(idf_path: str) -> None:
    global _IDF
    _IDF = np.load(idf_path)


def _w(tok: str) -> float:
    return float(_IDF[fnv_bucket(tok)]) if _IDF is not None else 1.0


def _soft(xs: list[str], ys: list[str]) -> float:
    return sum(1 for x in xs if any(JaroWinkler.normalized_similarity(x, y) >= 0.9 for y in ys)) / len(xs)


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else NAN


def _prefix(a: str, b: str) -> float:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / n if n else NAN


def pair_features(a: tuple, b: tuple) -> list[float]:
    core_a, comp_a, parts_a, legal_a, ini_a, _, addr_a, words_a, nums_a, nf_a, st_a, empty_a = a
    core_b, comp_b, parts_b, legal_b, ini_b, indic_b, addr_b, words_b, nums_b, nf_b, st_b, empty_b = b
    ta, tb = core_a.split(), core_b.split()
    sa, sb = set(ta), set(tb)
    names = bool(core_a) and bool(core_b)
    f = ([fuzz.ratio(core_a, core_b) / 100, fuzz.partial_ratio(core_a, core_b) / 100,
          fuzz.token_sort_ratio(core_a, core_b) / 100, fuzz.token_set_ratio(core_a, core_b) / 100,
          JaroWinkler.normalized_similarity(core_a, core_b), fuzz.ratio(comp_a, comp_b) / 100,
          fuzz.partial_ratio(comp_a, comp_b) / 100, JaroWinkler.normalized_similarity(comp_a, comp_b),
          _prefix(comp_a, comp_b)] if names else [NAN] * 9)
    f.append(_jacc(sa, sb))
    f += [_soft(ta, tb), _soft(tb, ta)] if names else [NAN, NAN]
    f.append(sum(_w(t) for t in sa & sb) / sum(_w(t) for t in sa | sb) if names else NAN)
    f.append(1.0 if (len(ini_a) >= 2 and ini_a == comp_b) or (len(ini_b) >= 2 and ini_b == comp_a) else 0.0)
    f.append(1.0 if min(len(comp_a), len(comp_b)) >= 4 and (comp_a in comp_b or comp_b in comp_a) else 0.0)
    pa = parts_a.split("|") if parts_a else [core_a]
    pb = parts_b.split("|") if parts_b else [core_b]
    f.append(max(fuzz.token_set_ratio(x, y) for x in pa for y in pb) / 100 if names else NAN)
    la, lb = set(legal_a.split()), set(legal_b.split())
    f += [1.0 if la and la == lb else 0.0, 1.0 if la and lb and not la & lb else 0.0,
          1.0 if bool(la) != bool(lb) else 0.0]
    f += [float(len(ta)), float(len(tb)),
          min(len(core_a), len(core_b)) / max(len(core_a), len(core_b)) if names else NAN]
    f += [float(indic_b), 1.0 if ta and tb and ta[0] == tb[0] else 0.0,
          1.0 if names and core_a == core_b else 0.0, 0.0 if core_b else 1.0]
    f += ([fuzz.token_set_ratio(addr_a, addr_b) / 100, fuzz.partial_ratio(addr_a, addr_b) / 100,
           fuzz.ratio(addr_a, addr_b) / 100] if addr_a and addr_b else [NAN] * 3)
    f.append(fuzz.token_set_ratio(words_a, words_b) / 100 if words_a and words_b else NAN)
    f.append(_jacc(set(words_a.split()), set(words_b.split())))
    na, nb = set(nums_a.split()), set(nums_b.split())
    f += [_jacc(na, nb), float(len(na & nb)), len(na & nb) / len(nb) if nb else NAN]
    if nf_a and nf_b:
        f += [1.0 if nf_a == nf_b else 0.0, float(Levenshtein.distance(nf_a, nf_b)),
              1.0 if nf_a.startswith(nf_b) or nf_b.startswith(nf_a) or nf_a.endswith(nf_b) or nf_b.endswith(nf_a)
              else 0.0, math.log1p(abs(int(nf_a[:15]) - int(nf_b[:15])))]
    else:
        f += [NAN] * 4
    f.append((1.0 if st_a == st_b else 0.0) if st_a and st_b else NAN)
    f += [float(empty_a), float(empty_b), float(len(addr_a.split())), float(len(addr_b.split()))]
    return f


def _chunk(args) -> np.ndarray:
    A, B = args
    return np.asarray([pair_features(a, b) for a, b in zip(A, B)], dtype=np.float32).reshape(-1, len(PAIR_FEATS))


def build_name_idf(work_dir: Path, split: str) -> Path:
    names = pl.concat([pl.read_parquet(norm_path(work_dir, split, s), columns=["name_core"])
                       for s in (1, 2, 3)])["name_core"]
    path = Path(work_dir) / split / "name_idf.npy"
    np.save(path, idf_from_df(doc_freq(word_counts(names)), len(names)))
    return path


def build_features(split: str, work_dir: Path, data_dir: Path, n_jobs: int,
                   chunk: int = 400_000, sub: int = 20_000) -> None:
    idf_path = build_name_idf(work_dir, split)
    norm = {s: pl.read_parquet(norm_path(work_dir, split, s), columns=["entity_id", "country"] + FIELDS)
            for s in (1, 2, 3)}
    f_s1 = norm[1].group_by("country", "name_core").len("n")
    f_tg = pl.concat([norm[s].select("country", "name_core") for s in (2, 3)]).group_by("country", "name_core").len("n")
    gt = None
    if split == "train":
        gt = read_ground_truth(Path(data_dir) / "train" / "train_ground_truth.tsv").with_columns(
            pl.lit(1, pl.Int8).alias("label"))
    out_dir = Path(work_dir) / split / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(str(idf_path),)) if n_jobs > 1 else None
    if ex is None:
        _init(str(idf_path))
    try:
        for f in sorted((Path(work_dir) / split / "cands").glob("*.parquet")):
            c = pl.read_parquet(f)
            if c.height == 0:
                continue
            tgt = int(c["tgt_src"][0])
            for part_no, start in enumerate(range(0, c.height, chunk)):
                part = c.slice(start, chunk)
                ia, ib = part["s1_row"].to_numpy(), part["tgt_row"].to_numpy()
                A = norm[1].select(pl.col(FIELDS).gather(ia)).rows()
                B = norm[tgt].select(pl.col(FIELDS).gather(ib)).rows()
                tasks = [(A[i:i + sub], B[i:i + sub]) for i in range(0, len(A), sub)]
                X = np.vstack(list(ex.map(_chunk, tasks)) if ex else [_chunk(t) for t in tasks])
                meta = pl.DataFrame({"s1_id": norm[1]["entity_id"].gather(ia),
                                     "tgt_id": norm[tgt]["entity_id"].gather(ib),
                                     "country": norm[1]["country"].gather(ia),
                                     "core_a": norm[1]["name_core"].gather(ia),
                                     "core_b": norm[tgt]["name_core"].gather(ib)})
                df = part.hstack(meta).hstack(pl.DataFrame(X, schema=PAIR_FEATS))
                df = (df.join(f_s1.rename({"name_core": "core_a", "n": "s1_freq_a"}), on=["country", "core_a"], how="left")
                        .join(f_s1.rename({"name_core": "core_b", "n": "s1_freq_b"}), on=["country", "core_b"], how="left")
                        .join(f_tg.rename({"name_core": "core_b", "n": "tgt_freq_b"}), on=["country", "core_b"], how="left")
                        .with_columns(pl.col(FREQ_FEATS).fill_null(0).cast(pl.Float32))
                        .drop("core_a", "core_b"))
                if gt is not None:
                    df = (df.join(gt, on=["s1_id", "tgt_id"], how="left")
                            .with_columns(pl.col("label").fill_null(0), fold_expr().alias("fold")))
                df.write_parquet(out_dir / f"{f.stem}_{part_no:03d}.parquet")
    finally:
        if ex is not None:
            ex.shutdown()
    augment_features(work_dir, split, n_jobs)
