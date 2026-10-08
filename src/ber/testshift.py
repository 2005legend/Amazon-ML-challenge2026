"""Test-set density correction.

The test split has about 1.9x as many unmatched records per S1 as train, so token statistics computed on test
distort the token-change features: decoy words start to look like filler. For every test country that also exists
in train, the token-change features are recomputed with TRAIN statistics (the distribution the models were fitted
on); a country unseen in train keeps its own test statistics. Stages 1, 2 (fold and A+B models) and 3 then rescore
the test pairs with the existing models, in a separate directory so the uncorrected tables stay untouched.
"""
import shutil
from pathlib import Path

import polars as pl

from .context import CTX_FEATS, build_stage2_tables
from .features import MODEL_FEATS
from .model import load, predict_stage, table_files
from .prepare import norm_path
from .stage3 import score_split, stage3_context
from .tokens import TC_FEATS, token_change_features, token_stats


def _stats(work_dir: Path, split: str) -> pl.DataFrame:
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["country", "name_core"])
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, split, s), columns=["country", "name_core"]) for s in (2, 3)])
    return token_stats(s1, tg)


def reference_stats(work_dir: Path) -> pl.DataFrame:
    st_tr, st_te = _stats(work_dir, "train"), _stats(work_dir, "test")
    seen = st_tr["country"].unique().to_list()
    return pl.concat([st_tr, st_te.filter(~pl.col("country").is_in(seen))])


def rescore_test(work_dir: Path, tc_dir: Path, n_jobs: int) -> None:
    work_dir, tc_dir = Path(work_dir), Path(tc_dir)
    (tc_dir / "test").mkdir(parents=True, exist_ok=True)
    if (tc_dir / "models").exists():
        shutil.rmtree(tc_dir / "models")
    shutil.copytree(work_dir / "models", tc_dir / "models")
    for s in (1, 2, 3):
        shutil.copy2(norm_path(work_dir, "test", s), tc_dir / "test" / norm_path(work_dir, "test", s).name)
    if (tc_dir / "test" / "feats").exists():
        shutil.rmtree(tc_dir / "test" / "feats")
    shutil.copytree(work_dir / "test" / "feats", tc_dir / "test" / "feats")
    stats = reference_stats(work_dir)
    s1 = pl.read_parquet(norm_path(work_dir, "test", 1), columns=["row", "name_core"])
    a = s1.select(pl.col("row").alias("s1_row"), pl.col("name_core").alias("core_a"))
    b = pl.concat([pl.read_parquet(norm_path(work_dir, "test", s), columns=["row", "name_core"])
                     .select(pl.lit(s, pl.Int8).alias("tgt_src"), pl.col("row").alias("tgt_row"), pl.col("name_core").alias("core_b")) for s in (2, 3)])
    for f in table_files(tc_dir, "test", "feats"):
        df = pl.read_parquet(f).drop(TC_FEATS, strict=False)
        names = (df.select("s1_row", pl.col("tgt_src").cast(pl.Int8), "tgt_row", "country")
                   .join(a, on="s1_row", how="left", maintain_order="left").join(b, on=["tgt_src", "tgt_row"], how="left", maintain_order="left"))
        df.hstack(token_change_features(names, stats, n_jobs=n_jobs)).write_parquet(f)
    m = lambda n: load(tc_dir / "models" / f"{n}.txt")
    predict_stage(tc_dir, "test", "feats", MODEL_FEATS, {"A": m("m1_A"), "B": m("m1_B")}, "p1")
    build_stage2_tables("test", tc_dir, n_jobs)
    predict_stage(tc_dir, "test", "feats2", MODEL_FEATS + CTX_FEATS, {"A": m("m2_A"), "B": m("m2_B")}, "p2")
    score_split(tc_dir, "test", "p2ab", m("m2_AB"))
    score_split(tc_dir, "test", "p3", m("m3_AB"), stage3_context(tc_dir, "test", n_jobs))
