"""Candidate generation: TF-IDF top-k over name, address and combined views, in both directions."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from .evaluate import macro_fbeta
from .io import read_ground_truth
from .prepare import norm_path
from .topk import pair_dot, topk
from .vectorize import doc_freq, idf_from_df, name4_counts, tfidf, word_counts

VIEWS = ("n", "a", "na")
FWD_BIT = {"n": 1, "a": 2, "na": 4}
REV_BIT = {"n": 8, "a": 16, "na": 32}
EXACT_BIT = 64
CANDS_SCHEMA = {"s1_row": pl.Int32, "tgt_row": pl.Int32, "tgt_src": pl.Int8, "cos_n": pl.Float32,
                "cos_a": pl.Float32, "cos_na": pl.Float32, "via": pl.Int8}


@dataclass
class BlockConfig:
    """Top-k per view and direction; df_frac/max_feats override max_df_frac/unlimited per view.

    Defaults come from the calibration in work/NOTES.md (India train, 50k S1 queries): char-4-gram
    names, cheap capped name/address views, and a combined view limited to each query's 10 rarest
    features, searched forward (S1 -> S2/S3) and in reverse (S2/S3 -> S1).
    """
    k_fwd: dict = field(default_factory=lambda: {"n": 15, "a": 15, "na": 30})
    k_rev: dict = field(default_factory=lambda: {"na": 3})
    max_df_frac: float = 0.01
    exact_max_group: int = 5
    df_frac: dict = field(default_factory=lambda: {"n": 0.003, "a": 0.003, "na": 0.01})
    max_feats: dict = field(default_factory=lambda: {"na": 10})

    def cap(self, view: str, n_targets: int) -> int:
        return max(1000, int(self.df_frac.get(view, self.max_df_frac) * n_targets))

    def feats(self, view: str) -> int:
        return self.max_feats.get(view, 1 << 30)


DEFAULT = BlockConfig()


def safe(country: str) -> str:
    return re.sub(r"\W+", "_", country)


def cands_path(work_dir: Path, split: str, country: str, tgt_src: int) -> Path:
    return Path(work_dir) / split / "cands_raw" / f"{safe(country)}_s{tgt_src}.parquet"


def _views(frames: dict[int, pl.DataFrame]) -> dict[int, dict[str, sp.csr_matrix]]:
    counts = {s: {"n": name4_counts(f["name_compact"]), "a": word_counts(f["addr_norm"])} for s, f in frames.items()}
    n_docs = sum(f.height for f in frames.values())
    out = {}
    idf = {v: idf_from_df(sum(doc_freq(counts[s][v]) for s in frames), n_docs) for v in ("n", "a")}
    for s in frames:
        n = tfidf(counts[s]["n"], idf["n"])
        a = tfidf(counts[s]["a"], idf["a"])
        na = sp.hstack([n * np.float32(np.sqrt(0.5)), a * np.float32(np.sqrt(0.5))], format="csr")
        out[s] = {"n": n, "a": a, "na": na}
    return out


def _pairs_from_topk(idx: np.ndarray, reverse: bool):
    q = np.repeat(np.arange(idx.shape[0], dtype=np.int32), idx.shape[1])
    t = idx.ravel()
    keep = t >= 0
    q, t = q[keep], t[keep]
    return (t, q) if reverse else (q, t)


def _exact_pairs(s1: pl.DataFrame, tg: pl.DataFrame, max_group: int):
    a = s1.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("i"), "name_core").filter(pl.col("name_core") != "")
    b = tg.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("j"), "name_core").filter(pl.col("name_core") != "")
    a = a.filter(pl.len().over("name_core") <= max_group)
    b = b.filter(pl.len().over("name_core") <= 4 * max_group)
    j = a.join(b, on="name_core")
    return j["i"].to_numpy(), j["j"].to_numpy()


def _union(parts, n_tgt: int):
    keys = np.concatenate([s.astype(np.int64) * n_tgt + t for s, t, _ in parts])
    flags = np.concatenate([np.full(len(s), f, np.int8) for s, t, f in parts])
    order = np.argsort(keys, kind="stable")
    keys, flags = keys[order], flags[order]
    uniq, start = np.unique(keys, return_index=True)
    via = np.bitwise_or.reduceat(flags, start) if len(uniq) else np.zeros(0, np.int8)
    return (uniq // n_tgt).astype(np.int32), (uniq % n_tgt).astype(np.int32), via


def build_candidates(split: str, work_dir: Path, cfg: BlockConfig = DEFAULT) -> None:
    cols = ["row", "country", "name_core", "name_compact", "addr_norm"]
    norm = {s: pl.read_parquet(norm_path(work_dir, split, s), columns=cols) for s in (1, 2, 3)}
    countries = sorted(set().union(*(set(norm[s]["country"].unique().to_list()) for s in (1, 2, 3))))
    for country in countries:
        frames = {s: norm[s].filter(pl.col("country") == country) for s in (1, 2, 3)}
        if frames[1].height == 0:
            continue
        mats = _views(frames)
        for tgt in (2, 3):
            n1, nt = frames[1].height, frames[tgt].height
            path = cands_path(work_dir, split, country, tgt)
            path.parent.mkdir(parents=True, exist_ok=True)
            if nt == 0:
                pl.DataFrame(schema=CANDS_SCHEMA).write_parquet(path)
                continue
            parts = []
            for v, k in cfg.k_fwd.items():
                idx, _ = topk(mats[1][v], mats[tgt][v], k, cfg.cap(v, nt), cfg.feats(v))
                parts.append((*_pairs_from_topk(idx, reverse=False), FWD_BIT[v]))
            for v, k in cfg.k_rev.items():
                idx, _ = topk(mats[tgt][v], mats[1][v], k, cfg.cap(v, n1), cfg.feats(v))
                parts.append((*_pairs_from_topk(idx, reverse=True), REV_BIT[v]))
            parts.append((*_exact_pairs(frames[1], frames[tgt], cfg.exact_max_group), EXACT_BIT))
            si, ti, via = _union(parts, nt)
            out = pl.DataFrame({
                "s1_row": frames[1]["row"].to_numpy()[si],
                "tgt_row": frames[tgt]["row"].to_numpy()[ti],
                "tgt_src": np.full(len(si), tgt, np.int8),
                "cos_n": pair_dot(mats[1]["n"], mats[tgt]["n"], si, ti),
                "cos_a": pair_dot(mats[1]["a"], mats[tgt]["a"], si, ti),
                "cos_na": pair_dot(mats[1]["na"], mats[tgt]["na"], si, ti),
                "via": via,
            })
            out.write_parquet(path)


def load_candidates(work_dir: Path, split: str, name: str = "cands_raw") -> pl.DataFrame:
    files = sorted((Path(work_dir) / split / name).glob("*.parquet"))
    cands = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    ids = {s: pl.read_parquet(norm_path(work_dir, split, s), columns=["row", "entity_id", "country"])
           for s in (1, 2, 3)}
    s1 = ids[1].rename({"row": "s1_row", "entity_id": "s1_id"})
    tg = pl.concat([ids[s].with_columns(pl.lit(s, pl.Int8).alias("tgt_src")) for s in (2, 3)])
    tg = tg.rename({"row": "tgt_row", "entity_id": "tgt_id"}).drop("country")
    return cands.join(s1, on="s1_row").join(tg, on=["tgt_src", "tgt_row"])


def gt_rows(work_dir: Path, data_dir: Path) -> pl.DataFrame:
    """Train ground truth with integer keys: s1_id, tgt_id, country, s1_row, tgt_src, tgt_row."""
    gt = read_ground_truth(Path(data_dir) / "train" / "train_ground_truth.tsv")
    s1 = pl.read_parquet(norm_path(work_dir, "train", 1), columns=["row", "entity_id", "country"]).rename(
        {"row": "s1_row", "entity_id": "s1_id"})
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, "train", s), columns=["row", "entity_id"])
                      .with_columns(pl.lit(s, pl.Int8).alias("tgt_src")) for s in (2, 3)])
    tg = tg.rename({"row": "tgt_row", "entity_id": "tgt_id"})
    return gt.join(s1, on="s1_id").join(tg, on="tgt_id")


def recall_report(work_dir: Path, data_dir: Path, name: str = "cands_raw", s1_subset=None) -> dict:
    """Pair recall and perfect-matcher F0.5 ceiling, reading candidate files one at a time on int keys."""
    gtr = gt_rows(work_dir, data_dir)
    s1 = pl.read_parquet(norm_path(work_dir, "train", 1), columns=["row", "entity_id", "country"])
    if s1_subset is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_subset.implode()))
        gtr = gtr.filter(pl.col("s1_id").is_in(s1_subset.implode()))
    keep_rows = s1["row"]
    keys = ["s1_row", "tgt_src", "tgt_row"]
    hits, n_pairs = [], 0
    for f in sorted((Path(work_dir) / "train" / name).glob("*.parquet")):
        c = pl.read_parquet(f, columns=keys)
        if s1_subset is not None:
            c = c.filter(pl.col("s1_row").is_in(keep_rows.implode()))
        n_pairs += c.height
        hits.append(gtr.join(c, on=keys, how="semi").select("s1_id", "tgt_id"))
    found = pl.concat(hits).unique()
    gtc = gtr.join(found.with_columns(pl.lit(1).alias("hit")), on=["s1_id", "tgt_id"], how="left").with_columns(
        pl.col("hit").fill_null(0))
    report = {"n_s1": s1.height, "pairs": n_pairs, "cands_per_s1": n_pairs / max(1, s1.height),
              "pair_recall": gtc["hit"].mean(),
              "ceiling_f05": macro_fbeta(found, gtr.select("s1_id", "tgt_id"), s1["entity_id"])}
    for country, recall in gtc.group_by("country").agg(pl.col("hit").mean()).iter_rows():
        report[f"recall_{country}"] = recall
    return report
