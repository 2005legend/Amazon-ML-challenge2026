"""Stage-2 features: within-S1 context, competition for each target record, and sibling agreement."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from .model import table_files
from .prepare import norm_path

GROUP_FEATS = ["c_rank_s1", "c_rank_s1src", "c_max_s1", "c_2nd_s1", "c_gap_s1", "c_sum_s1", "c_n05_s1",
               "c_max_s1src", "c_max_tgt", "c_rank_tgt", "c_margin_tgt", "c_n05_tgt", "c_n_tgt"]
SIB_FEATS = ["sib_name", "sib_addr", "sib_nf", "sib_comp", "sib_n"]
CTX_FEATS = ["p1"] + GROUP_FEATS + SIB_FEATS
NAN = float("nan")


def group_features(p: pl.DataFrame, score: str = "p1") -> pl.DataFrame:
    s = pl.col(score)
    s1, s1src, tg = ["s1_row"], ["s1_row", "tgt_src"], ["tgt_src", "tgt_row"]
    out = p.with_columns(
        s.rank("ordinal", descending=True).over(s1).cast(pl.Int32).alias("c_rank_s1"),
        s.rank("ordinal", descending=True).over(s1src).cast(pl.Int32).alias("c_rank_s1src"),
        s.max().over(s1).alias("c_max_s1"),
        (s.max().over(s1) - s).alias("c_gap_s1"),
        s.sum().over(s1).alias("c_sum_s1"),
        (s > 0.5).sum().over(s1).cast(pl.Int32).alias("c_n05_s1"),
        s.max().over(s1src).alias("c_max_s1src"),
        s.max().over(tg).alias("c_max_tgt"),
        s.rank("ordinal", descending=True).over(tg).cast(pl.Int32).alias("c_rank_tgt"),
        (s > 0.5).sum().over(tg).cast(pl.Int32).alias("c_n05_tgt"),
        pl.len().over(tg).cast(pl.Int32).alias("c_n_tgt"),
    )
    second_s1 = pl.when(pl.col("c_rank_s1") > 1).then(s).max().over(s1).fill_null(0.0)
    second_tg = pl.when(pl.col("c_rank_tgt") > 1).then(s).max().over(tg).fill_null(0.0)
    return out.with_columns(
        second_s1.alias("c_2nd_s1"),
        pl.when(pl.col("c_rank_tgt") == 1).then(s - second_tg).otherwise(s - pl.col("c_max_tgt"))
          .alias("c_margin_tgt"),
    )


def _sib_chunk(rows):
    out = []
    for core_b, comp_b, addr_b, nf_b, core_s, comp_s, addr_s, nf_s in rows:
        out.append((fuzz.token_set_ratio(core_b, core_s) / 100 if core_b and core_s else NAN,
                    fuzz.token_set_ratio(addr_b, addr_s) / 100 if addr_b and addr_s else NAN,
                    (1.0 if nf_b == nf_s else 0.0) if nf_b and nf_s else NAN,
                    fuzz.partial_ratio(comp_b, comp_s) / 100 if comp_b and comp_s else NAN))
    return out


SIM_SCHEMA = {"s_name": pl.Float64, "s_addr": pl.Float64, "s_nf": pl.Float64, "s_comp": pl.Float64}
_SIB_OUT = (("s_name", "sib_name"), ("s_addr", "sib_addr"), ("s_nf", "sib_nf"), ("s_comp", "sib_comp"))


def sibling_features(p: pl.DataFrame, tg_fields: pl.DataFrame, score: str = "p1", min_sib: float = 0.3,
                     min_pair: float = 0.01, n_jobs: int = 1, sub: int = 50_000,
                     slice_rows: int = 2_000_000) -> pl.DataFrame:
    """Similarity of each candidate to its S1's top-2 other candidates, weighted by their scores.

    Works on slices of (pair, sibling) rows so memory stays bounded; per-slice maxima and counts
    are recombined exactly (max of maxima, sum of counts).
    """
    keys = ["s1_row", "tgt_src", "tgt_row"]
    sibs = (p.filter(pl.col(score) >= min_sib).sort(["s1_row", score], descending=[False, True])
              .group_by("s1_row", maintain_order=True).head(2)
              .select("s1_row", pl.col("tgt_src").alias("sib_src"), pl.col("tgt_row").alias("sib_row"),
                      pl.col(score).alias("sib_p")))
    pairs = (p.filter(pl.col(score) >= min_pair).select(keys).join(sibs, on="s1_row")
              .filter(~((pl.col("sib_src") == pl.col("tgt_src")) & (pl.col("sib_row") == pl.col("tgt_row")))))
    f = tg_fields.select("tgt_src", "tgt_row", "name_core", "name_compact", "addr_norm", "num_first")
    fs = f.rename({"tgt_src": "sib_src", "tgt_row": "sib_row", "name_core": "core_s", "name_compact": "comp_s",
                   "addr_norm": "addr_s", "num_first": "nf_s"})
    parts = []
    ex = ProcessPoolExecutor(n_jobs) if n_jobs > 1 else None
    try:
        for start in range(0, pairs.height, slice_rows):
            j = (pairs.slice(start, slice_rows).join(f, on=["tgt_src", "tgt_row"])
                      .join(fs, on=["sib_src", "sib_row"]))
            rows = j.select("name_core", "name_compact", "addr_norm", "num_first", "core_s", "comp_s", "addr_s",
                            "nf_s").rows()
            tasks = [rows[i:i + sub] for i in range(0, len(rows), sub)]
            chunks = ex.map(_sib_chunk, tasks) if ex is not None else map(_sib_chunk, tasks)
            sim = pl.DataFrame([x for part in chunks for x in part], schema=SIM_SCHEMA, orient="row")
            jj = j.select(keys + ["sib_p"]).hstack(sim)
            parts.append(jj.group_by(keys).agg(
                *[(pl.col(c) * pl.col("sib_p")).max().alias(n) for c, n in _SIB_OUT],
                pl.len().cast(pl.Int32).alias("sib_n")))
    finally:
        if ex is not None:
            ex.shutdown()
    if parts:
        agg = pl.concat(parts).group_by(keys).agg(*[pl.col(n).max() for _, n in _SIB_OUT],
                                                  pl.col("sib_n").sum().cast(pl.Int32))
    else:
        agg = pl.DataFrame(schema={**{k: p.schema[k] for k in keys}, **{n: pl.Float64 for _, n in _SIB_OUT},
                                   "sib_n": pl.Int32})
    return p.select(keys).join(agg, on=keys, how="left")


def _tg_fields(work_dir: Path, split: str) -> pl.DataFrame:
    cols = ["row", "name_core", "name_compact", "addr_norm", "num_first"]
    return pl.concat([pl.read_parquet(norm_path(work_dir, split, s), columns=cols)
                        .with_columns(pl.lit(s, pl.Int8).alias("tgt_src")).rename({"row": "tgt_row"})
                      for s in (2, 3)])


def build_stage2_tables(split: str, work_dir: Path, n_jobs: int) -> None:
    keys = ["s1_row", "tgt_src", "tgt_row"]
    files = table_files(work_dir, split, "p1")
    p = pl.concat([pl.read_parquet(f, columns=keys + ["p1"]) for f in files])
    ctx = group_features(p).join(sibling_features(p, _tg_fields(work_dir, split), n_jobs=n_jobs),
                                 on=keys, how="left")
    out_dir = Path(work_dir) / split / "feats2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in table_files(work_dir, split, "feats"):
        feats = pl.read_parquet(f)
        feats.join(ctx, on=keys, how="left").write_parquet(out_dir / f.name)
