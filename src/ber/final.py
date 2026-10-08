"""Final submission: stacker on the density-corrected base scores, generator rules, expected-F0.5 decision,
same-address recovery, outputs and validation. Inputs: stage-3 scores (ber.stage3) and the corrected test tables
(ber.testshift)."""
from pathlib import Path

import polars as pl

from .blocking import load_candidates
from .genrules import apply_rules, test_flags
from .io import write_id_lists
from .model import load_table
from .package import validate
from .prepare import norm_path
from .rawform import TYPE_COLS, pair_types, raw_name_features, score_shape, template_features, template_tables
from .recovery import recovered, same_address_pairs
from .rules import catswap_pairs, exclude_scores
from .cardinality import fit_cardinality, prior_decision
from .neural import G9, load_g9
from .rawform import G5, G8
from .stacker import FULL, add_groups, oof_scores, stack_rows, stacked_scores
from .stage3 import base_scores


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s1_traits(work_dir: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(norm_path(work_dir, split, 1), columns=["entity_id", "country", "name", "addr_empty"]).rename({"entity_id": "s1_id"})


def run_final(work_dir: Path, tc_dir: Path, output_dir: Path, data_dir: Path, n_jobs: int, use_shape: bool, use_card: bool = False, use_neural: bool = False,
              card_all: bool = False, gate: str = "full") -> int:
    work_dir, tc_dir, output_dir = Path(work_dir), Path(tc_dir), Path(output_dir)
    h = exclude_scores(base_scores(work_dir, "train"), catswap_pairs(work_dir, "train"), "p2")
    t = base_scores(tc_dir, "test")
    log("base scores: holdout", h.height, "test", t.height)
    hf = load_table(work_dir, "train", "feats", TYPE_COLS, pl.col("fold") == "H")
    tf = load_table(work_dir, "test", "feats", TYPE_COLS)
    ty_h, ty_t = pair_types(hf, h), pair_types(tf, t)
    train = stack_rows(h, pl.scan_parquet(str(work_dir / "train" / "feats2" / "*.parquet")).filter(pl.col("fold") == "H"),
                       ty_h, raw_name_features(work_dir, "train", h, n_jobs))
    test = stack_rows(t, pl.scan_parquet(str(tc_dir / "test" / "feats2" / "*.parquet")), ty_t, raw_name_features(work_dir, "test", t, n_jobs))
    log("stacker rows: holdout", train.height, "test", test.height)
    tables = template_tables(work_dir, n_jobs)
    log("template tables", {k: len(v) for k, v in tables.items()})
    shape = (lambda x: score_shape(x.select("s1_id", "tgt_id", "p2"))) if use_shape else (lambda x: None)
    train = add_groups(train, shape(train), template_features(work_dir, "train", h, tables, n_jobs))
    test = add_groups(test, shape(test), template_features(work_dir, "test", t, tables, n_jobs))
    extra = G9 if use_neural else []
    if use_neural:
        train, test = (x.join(load_g9(work_dir, s), on=["s1_id", "tgt_id"], how="left") for x, s in ((train, "H"), (test, "test")))
        log("neural G9 features joined; test null share", float(test["g9_logit"].is_null().mean()))
    seen = pl.read_parquet(norm_path(work_dir, "train", 1), columns=["country"])["country"].unique().to_list()
    scores = stacked_scores(train, test, t, use_shape, seen, work_dir / "models", extra)
    scores.write_parquet(tc_dir / "test" / "stacked.parquet")
    log("stacked scores written; training countries", seen)
    last = None
    if use_card:
        cols = FULL + (G5 if use_shape else []) + G8 + extra
        oof = oof_scores(train, cols).join(train.select("s1_id", "tgt_id", "label"), on=["s1_id", "tgt_id"])
        types = lambda ty: ty.select("s1_id", "tgt_id", "ar", "nr")
        b = fit_cardinality(work_dir, oof, types(ty_h), s1_traits(work_dir, "train"), work_dir / "models")
        prior_countries = s1_traits(work_dir, "test")["country"].unique().to_list() if card_all else seen
        last = prior_decision(b, types(ty_t), s1_traits(work_dir, "test"), prior_countries)
        log("cardinality prior trained on", oof["s1_id"].n_unique(), "holdout S1s")
    pred = apply_rules(scores, test_flags(work_dir, gate), last)
    log("rule chain + decision:", pred.height, "pairs")
    cands = load_candidates(work_dir, "test", "cands").select("s1_id", "tgt_id")
    add = recovered(same_address_pairs(work_dir, "test", cands), pred)
    log("same-address recovery:", add.height, "pairs")
    matches = pl.concat([pred, add]).unique()
    cand_all = pl.concat([cands, add]).unique()
    s1_ids = pl.read_parquet(norm_path(work_dir, "test", 1), columns=["entity_id"])["entity_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    m, c = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    write_id_lists(c, s1_ids, cand_all, "candidate_entity_ids")
    write_id_lists(m, s1_ids, matches, "matched_entity_ids")
    log("written", m, c)
    return validate(m, c, data_dir)
