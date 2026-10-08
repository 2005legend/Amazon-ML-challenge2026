"""Rules recovered from how the data generator builds copies and lookalikes.

Every statistic is computed per country from the split's own candidate pairs (or, for the decoy vocabulary, from
training folds A+B), so a country that never appears in training is handled without naming it.

Exclusions (score set to 0):
  category swap      same address, one category word swapped for another (rules.catswap_pairs), minus swaps into
                     words that are added >= 5x as often as dropped (noise suffixes, not categories)
  decoy word         target = S1 name + one word w that appears >= 1000x at a neighbouring street number and rarely at
                     the same address (log ratio >= 3.5), at a different first number
  decoy vocabulary   added words that are < 1% true in >= 300 training (A+B) pairs
  number gate        a country whose same-street copies show no digit-typo signature (ratio < 0.05) cannot have true
                     copies whose address numbers are disjoint from the S1's
Boosts (score raised to >= 0.95):
  noise addition     same address + number, target adds a word that is far more common there than at a neighbour
  dual-use word      same-address additions of words that are decoys elsewhere, where the estimated noise share is
                     >= 0.8 and the model's acceptance is >= 0.3 below it
  abbreviation       same-address abbreviation swaps in a direction the model rejects while accepting the reverse
  noise-suffix swap  same-address swaps into a noise suffix, top claimant only, where acceptance < noise share - 0.05
"""
from pathlib import Path

import polars as pl

from .decide import decide
from .model import load_table
from .prepare import norm_path
from .rules import catswap_pairs, exclude_scores

D = {"score": "p2", "rule": "expected_f", "tau": None}
FC = ["s1_id", "tgt_id", "country", "a_tset", "nf_eq", "num_overlap", "num_jacc", "tc_n_ua", "tc_n_ub"]
nz = lambda c: pl.col(c).fill_nan(None)
SAME = (nz("a_tset") >= 0.95) & (pl.col("nf_eq") == 1)
NEIGH = (nz("a_tset") >= 0.8) & (pl.col("nf_eq") == 0)
SAMEADDR = (nz("a_tset") >= 0.95) & ((pl.col("nf_eq") == 1) | nz("nf_eq").is_null())


# ------------------------------------------------------------------ helpers
def boost(pairs: pl.DataFrame, sel: pl.DataFrame, score: str = "p2", floor: float = 0.95) -> pl.DataFrame:
    return pairs.join(sel.with_columns(pl.lit(True).alias("_b")), on=["s1_id", "tgt_id"], how="left").with_columns(
        pl.when(pl.col("_b")).then(pl.max_horizontal(score, pl.lit(floor))).otherwise(pl.col(score)).alias(score)).drop("_b")


def cores(work_dir: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["entity_id", "name_core"]).rename({"entity_id": "s1_id", "name_core": "c1"})
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=["entity_id", "name_core"]) for k in (2, 3)]).rename({"entity_id": "tgt_id", "name_core": "c2"})
    return s1, tg


def edits(work_dir: Path, split: str, feats: pl.DataFrame) -> pl.DataFrame:
    """Adds the words the target adds to / drops from the S1 core name (as lists)."""
    s1, tg = cores(work_dir, split)
    return feats.join(s1, on="s1_id").join(tg, on="tgt_id").with_columns(
        pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).alias("add"),
        pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).alias("drop")).drop("c1", "c2")


def one_word_additions(x: pl.DataFrame) -> pl.DataFrame:
    return x.filter((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0)).with_columns(pl.col("add").list.first().alias("w"))


def rel(a: str, b: str) -> str:
    """Relation between two street numbers: eq / indel / sub / transpose / near<=4 / near<=20 / far."""
    if not a or not b:
        return "na"
    if a == b:
        return "eq"
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        if any(l[:i] + l[i + 1:] == s for i in range(len(l))):
            return "indel"
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return "sub"
        if len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]:
            return "transpose"
    try:
        x = abs(int(a) - int(b))
        return "near<=4" if x <= 4 else "near<=20" if x <= 20 else "far"
    except ValueError:
        return "far"


def is_abbrev(short: str, long: str) -> bool:
    if not short or not long or len(short) >= len(long) or short[0] != long[0] or len(short) > 5:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


# ------------------------------------------------------------------ exclusions
def refined_catswap(work_dir: Path, split: str, cs: pl.DataFrame) -> pl.DataFrame:
    """Category-swap flags minus swaps into asymmetric replacement words (added >= 5x as often as dropped, >= 200)."""
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["entity_id", "country", "name_core"]).rename({"entity_id": "s1_id", "name_core": "c1"})
    _, tg = cores(work_dir, split)
    x = cs.join(s1, on="s1_id").join(tg, on="tgt_id").with_columns(
        pl.col("c2").str.split(" ").list.set_difference(pl.col("c1").str.split(" ")).list.first().alias("add"),
        pl.col("c1").str.split(" ").list.set_difference(pl.col("c2").str.split(" ")).list.first().alias("drop"))
    a = x.group_by("country", pl.col("add").alias("w")).len("na")
    d = x.group_by("country", pl.col("drop").alias("w")).len("nd")
    m = a.join(d, on=["country", "w"], how="full", coalesce=True).fill_null(0).with_columns(((pl.col("na") + 1) / (pl.col("nd") + 1)).alias("ratio"))
    asym = m.filter((pl.col("na") >= 200) & (pl.col("ratio") >= 5))
    return x.join(asym.select("country", pl.col("w").alias("add")), on=["country", "add"], how="anti").select("s1_id", "tgt_id")


def decoy_word_pairs(x: pl.DataFrame, min_d: int = 1000, llr: float = 3.5) -> pl.DataFrame:
    one = one_word_additions(x)
    st = one.group_by("country", "w").agg(NEIGH.fill_null(False).sum().alias("D"), SAME.fill_null(False).sum().alias("S")).with_columns(
        ((pl.col("D") + 1) / (pl.col("S") + 1)).log().alias("llr"))
    dw = st.filter((pl.col("D") >= min_d) & (pl.col("llr") >= llr))
    return one.join(dw.select("country", "w"), on=["country", "w"]).filter(pl.col("nf_eq") == 0).select("s1_id", "tgt_id").unique()


def decoy_vocabulary(work_dir: Path) -> pl.DataFrame:
    """Per-country added words that are < 1% true among >= 300 training (folds A+B) one-word additions."""
    tr = load_table(work_dir, "train", "feats", ["s1_id", "tgt_id", "country", "label"], pl.col("fold").is_in(["A", "B"]))
    e = one_word_additions(edits(work_dir, "train", tr))
    return e.group_by("country", "w").agg(pl.len(), pl.col("label").mean().alias("true")).filter((pl.col("len") >= 300) & (pl.col("true") < 0.01))


def vocab_pairs(x: pl.DataFrame, voc: pl.DataFrame) -> pl.DataFrame:
    y = x.filter(pl.col("drop").list.len() <= 1).explode("add", empty_as_null=True)
    return y.join(voc.select("country", pl.col("w").alias("add")), on=["country", "add"]).select("s1_id", "tgt_id").unique()


def gate_countries(work_dir: Path, split: str, feats: pl.DataFrame, max_ratio: float = 0.05) -> list[str]:
    """Countries whose exact-name same-street pairs show no digit-typo signature (one-digit indel with |diff| > 20)
    relative to the neighbour signature (one-digit substitution or |diff| <= 20)."""
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["entity_id", "num_first"]).rename({"entity_id": "s1_id", "num_first": "f1"})
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=["entity_id", "num_first"]) for k in (2, 3)]).rename({"entity_id": "tgt_id", "num_first": "f2"})
    x = feats.filter((nz("a_tset") >= 0.8) & (pl.col("nf_eq") == 0) & (pl.col("tc_n_ua") == 0) & (pl.col("tc_n_ub") == 0))
    x = x.join(s1, on="s1_id").join(tg, on="tgt_id").with_columns(
        pl.struct("f1", "f2").map_elements(lambda r: rel(r["f1"], r["f2"]), return_dtype=pl.String).alias("rel"),
        ((pl.col("f1").cast(pl.Int64, strict=False) - pl.col("f2").cast(pl.Int64, strict=False)).abs() > 20).alias("big"))
    g = x.group_by("country").agg(((pl.col("rel") == "indel") & pl.col("big")).sum().alias("typo"),
                                  pl.col("rel").is_in(["sub", "near<=4", "near<=20"]).sum().alias("neigh"))
    g = g.with_columns((pl.col("typo") / pl.col("neigh")).alias("ratio"))
    print("street-number typo signature ratio per country:", sorted(g.select("country", "ratio").rows()), flush=True)
    return g.filter(pl.col("ratio") < max_ratio)["country"].to_list()


def numgate_pairs(feats: pl.DataFrame, countries: list[str]) -> pl.DataFrame:
    return feats.filter(pl.col("country").is_in(countries) & (pl.col("nf_eq") == 0) & (pl.col("num_overlap") == 0)
                        & nz("num_jacc").is_not_null()).select("s1_id", "tgt_id").unique()


def released_gate(work_dir: Path, split: str, gated: pl.DataFrame, mode: str = "narrow") -> pl.DataFrame:
    """Narrow the number gate to digit-level relations. The gate's evidence (no one-digit-indel typo signature) says
    nothing about wholesale renumbering: on the training holdout, exact-name pairs with the same legal form whose street
    numbers are unrelated ('far') are 82-91% true and one-digit indels 94-98%. Those relations are released unless the
    legal forms conflict; substitutions / neighbouring numbers / transpositions (clone-like) stay gated.
    mode 'open' releases every relation except legal-form conflicts (leaving the decision to the model)."""
    s1 = pl.read_parquet(norm_path(work_dir, split, 1), columns=["entity_id", "num_first"]).rename({"entity_id": "s1_id", "num_first": "f1"})
    tg = pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=["entity_id", "num_first"]) for k in (2, 3)]).rename({"entity_id": "tgt_id", "num_first": "f2"})
    lf = load_table(work_dir, split, "feats", ["s1_id", "tgt_id", "legal_conflict"])
    x = gated.join(s1, on="s1_id").join(tg, on="tgt_id").join(lf, on=["s1_id", "tgt_id"], how="left").with_columns(
        pl.struct("f1", "f2").map_elements(lambda r: rel(r["f1"], r["f2"]), return_dtype=pl.String).alias("rel"))
    if mode == "open":
        return x.filter(pl.col("legal_conflict") == 1).select("s1_id", "tgt_id")
    return x.filter(~pl.col("rel").is_in(["far", "indel"]) | (pl.col("legal_conflict") == 1)).select("s1_id", "tgt_id")


# ------------------------------------------------------------------ boosts
def noise_addition_pairs(x: pl.DataFrame) -> pl.DataFrame:
    one = one_word_additions(x)
    st = one.group_by("country", "w").agg(NEIGH.fill_null(False).sum().alias("D"), SAME.fill_null(False).sum().alias("S")).with_columns(
        ((pl.col("D") + 1) / (pl.col("S") + 1)).log().alias("llr"))
    nw = st.filter((pl.col("S") >= 1000) & (pl.col("llr") <= -1))
    return one.filter(SAME).join(nw.select("country", "w"), on=["country", "w"]).select("s1_id", "tgt_id").unique()


def _noise_share(one: pl.DataFrame, extra_aggs: list) -> pl.DataFrame:
    st = one.group_by("country", "w").agg([SAME.sum().alias("S"), NEIGH.sum().alias("D")] + extra_aggs)
    r0 = st.filter(pl.col("D") >= 5000).group_by("country").agg((pl.col("S") / pl.col("D")).median().alias("r0"))
    return st.join(r0, on="country", how="left"), r0


def dualuse_pairs(x: pl.DataFrame, acc: pl.DataFrame) -> pl.DataFrame:
    one = one_word_additions(x).join(acc, on=["s1_id", "tgt_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    st, r0 = _noise_share(one, [pl.col("acc").filter(SAME).mean().alias("acc_S")])
    st = st.with_columns((1 - pl.col("r0") * pl.col("D") / pl.col("S")).alias("ns"))
    sel = st.filter((pl.col("D") >= 1000) & (pl.col("ns") >= 0.8) & (pl.col("acc_S") < pl.col("ns") - 0.3))
    print("dual-use words boosted at the same address:", sel.select("country", "w", "S", "D").rows(), flush=True)
    return one.filter(SAME).join(sel.select("country", "w"), on=["country", "w"]).select("s1_id", "tgt_id").unique()


def abbrev_pairs(x: pl.DataFrame, acc: pl.DataFrame) -> pl.DataFrame:
    sw = x.filter((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 1) & SAME).with_columns(
        pl.col("add").list.first().alias("a"), pl.col("drop").list.first().alias("d"))
    sw = sw.with_columns(pl.struct("a", "d").map_elements(lambda r: is_abbrev(r["d"], r["a"]) or is_abbrev(r["a"], r["d"]), return_dtype=pl.Boolean).alias("abbr"))
    sw = sw.filter(pl.col("abbr")).join(acc, on=["s1_id", "tgt_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    g = sw.group_by("country", "d", "a").agg(pl.len().alias("n"), pl.col("acc").mean().alias("r"))
    rev = g.select("country", pl.col("a").alias("d"), pl.col("d").alias("a"), pl.col("n").alias("n_rev"), pl.col("r").alias("r_rev"))
    sel = g.join(rev, on=["country", "d", "a"]).filter((pl.col("r") < 0.5) & (pl.col("r_rev") >= 0.9) & (pl.col("n_rev") >= 100))
    print("one-way abbreviation directions boosted:", sel.select("country", "d", "a").rows(), flush=True)
    return sw.join(sel.select("country", "d", "a"), on=["country", "d", "a"]).select("s1_id", "tgt_id")


def noise_suffixes(x: pl.DataFrame) -> pl.DataFrame:
    st, _ = _noise_share(one_word_additions(x), [])
    st = st.with_columns((1 - pl.col("r0").fill_null(0) * pl.col("D") / pl.col("S")).alias("ns"))
    return st.filter((pl.col("S") >= 500) & (pl.col("ns") >= 0.8)).select("country", "w", "ns")


def swap_pairs(x: pl.DataFrame, scores: pl.DataFrame, acc: pl.DataFrame) -> pl.DataFrame:
    nw = noise_suffixes(x)
    sw = x.filter(SAMEADDR & (pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 1)).with_columns(pl.col("add").list.first().alias("w"))
    sw = sw.join(nw, on=["country", "w"]).join(scores, on=["s1_id", "tgt_id"])
    sw = sw.join(acc, on=["s1_id", "tgt_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    g = sw.group_by("country", "w").agg(pl.len().alias("n"), pl.col("acc").mean().alias("acc_sw"), pl.col("ns").first())
    sel = g.filter((pl.col("n") >= 200) & (pl.col("acc_sw") < pl.col("ns") - 0.05))
    print("noise-suffix swap classes boosted:", sel.select("country", "w").sort("country", "w").rows(), flush=True)
    return sw.join(sel.select("country", "w"), on=["country", "w"]).select("s1_id", "tgt_id")


def top1(pairs: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Keep pairs whose S1 is the target's highest-scoring candidate."""
    r = scores.with_columns(pl.col("p2").rank("ordinal", descending=True).over("tgt_id").alias("rk")).filter(pl.col("rk") == 1)
    return pairs.join(r.select("s1_id", "tgt_id"), on=["s1_id", "tgt_id"])


# ------------------------------------------------------------------ test-time chain
def test_flags(work_dir: Path, gate: str = "full") -> dict:
    """Label-free exclusion / boost pair sets for the test split (vocabulary from training folds A+B)."""
    feats = load_table(work_dir, "test", "feats", FC)
    x = edits(work_dir, "test", feats)
    cs = refined_catswap(work_dir, "test", catswap_pairs(work_dir, "test"))
    gate_pairs = numgate_pairs(feats, gate_countries(work_dir, "test", feats))
    if gate in ("narrow", "open"):
        gate_pairs = released_gate(work_dir, "test", gate_pairs, gate)
    return {"catswap": cs, "decoyword": decoy_word_pairs(x), "decoyvocab": vocab_pairs(x, decoy_vocabulary(work_dir)),
            "numgate": gate_pairs, "noiseword": noise_addition_pairs(x), "edits": x}


def apply_rules(scores: pl.DataFrame, flags: dict, last=None) -> pl.DataFrame:
    """Exclusions, then boosts in three passes (each later boost family is judged against the previous decisions).
    `last` optionally replaces the final decision (scores -> selected pairs), e.g. cardinality.prior_decision."""
    acc = lambda p: p.select("s1_id", "tgt_id").with_columns(pl.lit(1).alias("acc"))
    t = scores.select("s1_id", "tgt_id", "p2")
    ex = pl.concat([flags[k] for k in ("catswap", "decoyword", "decoyvocab", "numgate")]).unique()
    base = exclude_scores(t, ex, "p2")
    nb = flags["noiseword"]
    p10 = decide(boost(base, nb.join(ex, on=["s1_id", "tgt_id"], how="anti")), "p2", D)
    x = flags["edits"]
    b11 = pl.concat([nb, dualuse_pairs(x, acc(p10)), abbrev_pairs(x, acc(p10))]).unique().join(ex, on=["s1_id", "tgt_id"], how="anti")
    p11 = decide(boost(base, b11), "p2", D)
    sp = top1(swap_pairs(x, t, acc(p11)), t).join(ex, on=["s1_id", "tgt_id"], how="anti")
    final = boost(base, pl.concat([b11, sp]).unique())
    return (last(final) if last is not None else decide(final, "p2", D)).select("s1_id", "tgt_id")
