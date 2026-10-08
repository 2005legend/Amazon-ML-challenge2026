"""Deterministic assignment of Source-1 entities to folds A (40%), B (40%) and holdout H (20%)."""
import polars as pl


def fold_expr(col: str = "s1_id") -> pl.Expr:
    n = pl.col(col).str.slice(3).cast(pl.Int64) % 1_000_003
    h = (n * 2654435761) % 4294967296 % 10
    return pl.when(h < 4).then(pl.lit("A")).when(h < 8).then(pl.lit("B")).otherwise(pl.lit("H"))
