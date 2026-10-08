"""Same-address recovery of exact copies the similarity search never retrieved.

For names shared by 100-300 records the top-k search overflows, so exact copies at the exact same address can be
missed. A deterministic key adds, for every S1, the targets with the same (country, core name, first street number)
and an identical normalised address (token-set similarity 1.0), in name groups of at most 600 targets, when exactly
one S1 claims the target and the target is not already matched. Holdout: 96% precise. The added pairs are also
added to the candidate set, since matches must be a subset of the candidates.
"""
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

from .prepare import norm_path

C = ["entity_id", "country", "name", "name_core", "addr_norm", "num_first", "state", "addr_empty"]


def same_address_pairs(work_dir: Path, split: str, cand: pl.DataFrame, cap: int = 600) -> pl.DataFrame:
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=C)
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=C) for k in (2, 3)]).filter(pl.col("name_core").str.len_chars() > 0)
    nt = tg.group_by("country", "name_core").len("nt")
    q = s1.join(nt, on=["country", "name_core"]).filter(pl.col("nt") <= cap)
    x = q.select(pl.col("entity_id").alias("s1_id"), "country", "name_core", "num_first", pl.col("addr_norm").alias("a1")).join(
        tg.filter(~pl.col("addr_empty")).select(pl.col("entity_id").alias("tgt_id"), "country", "name_core", "num_first", pl.col("addr_norm").alias("a2")),
        on=["country", "name_core", "num_first"])
    x = x.join(cand, on=["s1_id", "tgt_id"], how="anti")
    return x.with_columns(pl.Series("asim", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0))


def recovered(x: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
    sel = x.filter(pl.col("asim") >= 0.999).filter(pl.len().over("tgt_id") == 1)
    return sel.join(matched.select("tgt_id").unique(), on="tgt_id", how="anti").select("s1_id", "tgt_id")
