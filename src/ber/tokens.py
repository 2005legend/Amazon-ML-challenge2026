"""Token-change features: which words differ between two names, and whether they are category words or fillers.

Word statistics are computed per country from the split's own names (no labels), so they carry over to countries
the model never saw in training. A "category" word is common in that country's S1 names and not over-represented
in S2/S3 names (e.g. club, union, care, clinic); a "filler" word is over-represented in S2/S3 names because noise
or decoys add it (e.g. holding, services, center).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from .model import table_files
from .prepare import norm_path

TC_FEATS = ["tc_n_ua", "tc_n_ub", "tc_c_ua", "tc_c_ub", "tc_fmin_ua", "tc_fmax_ua", "tc_fmin_ub", "tc_fmax_ub",
            "tc_share_ua", "tc_share_ub"]


def unmatched(a: str, b: str, thr: int = 80) -> tuple[list[str], list[str]]:
    """Tokens of each name with no fuzzy counterpart (rapidfuzz ratio >= thr) in the other name."""
    ta, tb = (a or "").split(), (b or "").split()
    ua = [x for x in ta if not any(fuzz.ratio(x, y) >= thr for y in tb)]
    ub = [y for y in tb if not any(fuzz.ratio(x, y) >= thr for x in ta)]
    return ua, ub


def _doc_tokens(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select("country", pl.col("name_core").str.split(" ").list.unique().alias("t"))
              .explode("t", empty_as_null=True).drop_nulls().filter(pl.col("t") != "")
              .group_by("country", "t").len())


def token_stats(s1: pl.DataFrame, tg: pl.DataFrame, min_share: float = 0.0005, max_f: float = 0.3) -> pl.DataFrame:
    """Per (country, token): share1 = log(S1 doc share), f = log target/S1 doc-frequency ratio (size-adjusted)."""
    n = s1.group_by("country").len("n1").join(tg.group_by("country").len("nt"), on="country", how="full",
                                                coalesce=True).fill_null(1)
    d = (_doc_tokens(s1).rename({"len": "df1"})
         .join(_doc_tokens(tg).rename({"len": "dft"}), on=["country", "t"], how="full", coalesce=True)
         .fill_null(0).join(n, on="country"))
    f = (pl.col("dft") + 1).log() - (pl.col("df1") + 1).log() - (pl.col("nt") / pl.col("n1")).log()
    share = (pl.col("df1") / pl.col("n1")).log()
    return d.select("country", "t", "df1", share.alias("share1"), f.alias("f"),
                    ((pl.col("df1") / pl.col("n1") >= min_share) & (f < max_f)).alias("cat"))


def _diff_chunk(rows):
    return [tuple(" ".join(x) for x in unmatched(a, b)) for a, b in rows]


def token_change_features(pairs: pl.DataFrame, stats: pl.DataFrame, n_jobs: int = 1,
                          sub: int = 50_000) -> pl.DataFrame:
    """pairs: country, core_a, core_b -> one row per pair with TC_FEATS (NaN where no token differs)."""
    rows = list(zip(pairs["core_a"].to_list(), pairs["core_b"].to_list()))
    tasks = [rows[i:i + sub] for i in range(0, len(rows), sub)]
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            res = [x for part in ex.map(_diff_chunk, tasks) for x in part]
    else:
        res = [x for t in tasks for x in _diff_chunk(t)]
    base = pairs.select("country").with_row_index("i").with_columns(
        pl.Series("ua", [x for x, _ in res], dtype=pl.Utf8), pl.Series("ub", [y for _, y in res], dtype=pl.Utf8))

    def side(col: str, tag: str) -> pl.DataFrame:
        e = (base.select("i", "country", pl.col(col).str.split(" ").alias("t"))
                 .explode("t", empty_as_null=True).drop_nulls().filter(pl.col("t") != "")
                 .join(stats, on=["country", "t"], how="left"))
        return e.group_by("i").agg(pl.len().alias(f"tc_n_{tag}"), pl.col("cat").fill_null(False).sum().alias(f"tc_c_{tag}"),
                                   pl.col("f").min().alias(f"tc_fmin_{tag}"), pl.col("f").max().alias(f"tc_fmax_{tag}"),
                                   pl.col("share1").max().alias(f"tc_share_{tag}"))

    out = base.select("i").join(side("ua", "ua"), on="i", how="left").join(side("ub", "ub"), on="i", how="left")
    counts = ["tc_n_ua", "tc_n_ub", "tc_c_ua", "tc_c_ub"]
    return (out.sort("i").with_columns(pl.col(counts).fill_null(0))
               .select([pl.col(c).cast(pl.Float32).fill_null(float("nan")) for c in TC_FEATS]))


def augment_features(work_dir: Path, split: str, n_jobs: int) -> None:
    """Add (or replace) TC_FEATS columns in every work/<split>/feats file."""
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["row", "country", "name_core"])
    tgs = {s: pl.read_parquet(norm_path(work_dir, split, s), columns=["row", "country", "name_core"]) for s in (2, 3)}
    stats = token_stats(s1.drop("row"), pl.concat([t.drop("row") for t in tgs.values()]))
    a = s1.select(pl.col("row").alias("s1_row"), pl.col("name_core").alias("core_a"))
    b = pl.concat([t.select(pl.lit(s, pl.Int8).alias("tgt_src"), pl.col("row").alias("tgt_row"),
                            pl.col("name_core").alias("core_b")) for s, t in tgs.items()])
    for f in table_files(work_dir, split, "feats"):
        df = pl.read_parquet(f).drop(TC_FEATS, strict=False)
        names = (df.select("s1_row", pl.col("tgt_src").cast(pl.Int8), "tgt_row", "country")
                   .join(a, on="s1_row", how="left", maintain_order="left")
                   .join(b, on=["tgt_src", "tgt_row"], how="left", maintain_order="left"))
        df.hstack(token_change_features(names, stats, n_jobs=n_jobs)).write_parquet(f)
