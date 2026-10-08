"""TSV reading/writing. Every file is tab-separated, UTF-8, with no quoting."""
from pathlib import Path

import polars as pl

_SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, has_header=True,
                     schema_overrides={c: pl.Utf8 for c in _SRC_COLS},
                     empty_string_is_null=False)
    return df.rename({"business_name": "name", "business_address": "address"})


def read_ground_truth(path: Path) -> pl.DataFrame:
    gt = pl.read_csv(path, separator="\t", quote_char=None, has_header=True,
                     schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
                     empty_string_is_null=False)
    return (gt.rename({"source1_entity_id": "s1_id"})
              .with_columns(pl.col("matched_entity_ids").str.split(",").alias("tgt_id"))
              .explode("tgt_id", empty_as_null=True)
              .filter(pl.col("tgt_id").is_not_null() & (pl.col("tgt_id") != ""))
              .select("s1_id", "tgt_id"))


def write_id_lists(path: Path, s1_ids: pl.Series, pairs: pl.DataFrame, col: str) -> None:
    """One row per S1 id (in the given order); ids comma-joined, deduplicated, empty when none."""
    lists = (pairs.select("s1_id", "tgt_id")
                  .group_by("s1_id", maintain_order=True)
                  .agg(pl.col("tgt_id").unique(maintain_order=True))
                  .with_columns(pl.col("tgt_id").list.join(",").alias(col))
                  .select("s1_id", col))
    out = (pl.DataFrame({"s1_id": s1_ids})
             .join(lists, on="s1_id", how="left", maintain_order="left")
             .with_columns(pl.col(col).fill_null(""))
             .rename({"s1_id": "source1_entity_id"}))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")
