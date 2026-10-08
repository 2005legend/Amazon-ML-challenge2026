"""Post-model rule: reject category-word swaps at the same address.

A candidate whose name differs from the S1 name by exactly one word, where both words are common *category*
words of that country's S1 names (club/union/ecole, care/clinic...), at the same address and house number, is a
sister organisation, not a noisy copy. On the training holdout such pairs are true 1.5% of the time; the model
already rejects them for US/India but trusts them for vocabularies it never saw (France). Spelling variants
(same two-letter prefix) and abbreviations (one word a subsequence of the other, cie/compagnie) are exempt.
"""
from pathlib import Path

import polars as pl

from .model import table_files
from .prepare import norm_path
from .tokens import token_stats, unmatched


def _subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(c in it for c in a)


def category_swap(country: str, a: str, b: str, stats: dict, min_df1: int = 500, max_f: float = 0.3) -> bool:
    ta, tb = (a or "").split(), (b or "").split()
    if len(ta) < 2 or len(tb) < 2:
        return False
    ua, ub = unmatched(a, b)
    if len(ua) != 1 or len(ub) != 1:
        return False
    o, i = ua[0], ub[0]
    if len(o) < 3 or len(i) < 3 or o[:2] == i[:2] or _subseq(o, i) or _subseq(i, o):
        return False
    so, si = stats.get((country, o)), stats.get((country, i))
    return bool(so and si and so[0] >= min_df1 and si[0] >= min_df1 and so[1] < max_f and si[1] < max_f)


def exclude_scores(pairs: pl.DataFrame, excl: pl.DataFrame, score: str) -> pl.DataFrame:
    """Set `score` to 0 for the (s1_id, tgt_id) pairs listed in `excl`."""
    x = excl.select("s1_id", "tgt_id").unique().with_columns(pl.lit(True).alias("_x"))
    return (pairs.join(x, on=["s1_id", "tgt_id"], how="left", maintain_order="left")
                 .with_columns(pl.when(pl.col("_x")).then(0.0).otherwise(pl.col(score)).cast(pairs.schema[score]).alias(score))
                 .drop("_x"))


def catswap_pairs(work_dir: Path, split: str) -> pl.DataFrame:
    """All candidate pairs of `split` that are category swaps at the same address."""
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["row", "country", "name_core"])
    tgs = {s: pl.read_parquet(norm_path(work_dir, split, s), columns=["row", "country", "name_core"]) for s in (2, 3)}
    st = token_stats(s1.drop("row"), pl.concat([t.drop("row") for t in tgs.values()]))
    stats = {(c, t): (d, f) for c, t, d, f in st.select("country", "t", "df1", "f").iter_rows()}
    same = (pl.col("a_tset").fill_nan(0) >= 0.95) & (pl.col("nf_eq").fill_nan(0) == 1)
    cols = ["s1_id", "tgt_id", "country", "s1_row", "tgt_src", "tgt_row"]
    p = pl.concat([pl.read_parquet(f, columns=cols + ["a_tset", "nf_eq"]).filter(same).select(cols)
                   for f in table_files(work_dir, split, "feats")])
    a = s1.select(pl.col("row").alias("s1_row"), pl.col("name_core").alias("ca"))
    b = pl.concat([t.select(pl.lit(s, pl.Int8).alias("tgt_src"), pl.col("row").alias("tgt_row"), pl.col("name_core").alias("cb"))
                   for s, t in tgs.items()])
    p = p.with_columns(pl.col("tgt_src").cast(pl.Int8)).join(a, on="s1_row").join(b, on=["tgt_src", "tgt_row"])
    flag = [category_swap(c, x, y, stats) for c, x, y in zip(p["country"].to_list(), p["ca"].to_list(), p["cb"].to_list())]
    return p.filter(pl.Series(flag)).select("s1_id", "tgt_id")
