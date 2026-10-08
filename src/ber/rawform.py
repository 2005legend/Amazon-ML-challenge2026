"""Pair descriptors computed from the RAW records, which normalisation discards, plus structural pair types.

Copies are made from the owner's raw string with a few character edits; a different business with the same name
was written independently, so its raw form (legal-form spelling, casing, brackets, word order, edit script) and the
kind of transformation from the S1 differ.

types      address relation x name relation x whether the S1 is the target's top claimant (R33)
raw name   14 descriptors: raw equality, word order, casing, brackets, legal words as written, edit-script counts
templates  copy-vs-decoy log-likelihood ratios of the S1 -> target transformation (legal-form rewrite as written,
           added and dropped words, character substitutions), learned on a 20% sample of training folds A+B,
           plus typed edit counts (accent, punctuation, doubled letter, transposition, keyboard neighbour, ...)
shape      leave-one-out summary of the S1's OTHER candidate scores (never the pair itself)
"""
import math
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from .prepare import norm_path

# ------------------------------------------------------------------ structural types
_nz = lambda c: pl.col(c).fill_nan(None)
TYPE_COLS = ["s1_id", "tgt_id", "country", "a_tset", "nf_eq", "addr_empty_b", "addr_empty_a", "tc_n_ua", "tc_n_ub", "name_exact", "acro", "n_tset", "tgt_src"]
AR_EXPR = (pl.when(pl.col("addr_empty_b") == 1).then(pl.lit("noaddr"))
             .when(pl.col("addr_empty_a") == 1).then(pl.lit("s1noaddr"))
             .when((_nz("a_tset") >= 0.95) & (pl.col("nf_eq") == 1)).then(pl.lit("same"))
             .when((_nz("a_tset") >= 0.95) & _nz("nf_eq").is_null()).then(pl.lit("same_nonum"))
             .when((_nz("a_tset") >= 0.8) & (pl.col("nf_eq") == 0)).then(pl.lit("diffnum"))
             .when(pl.col("nf_eq") == 1).then(pl.lit("num_only"))
             .otherwise(pl.lit("other")))
_ua, _ub = pl.col("tc_n_ua"), pl.col("tc_n_ub")
NR_EXPR = (pl.when(pl.col("name_exact") == 1).then(pl.lit("exact"))
             .when((_ua == 0) & (_ub == 0)).then(pl.lit("fuzzy_eq"))
             .when(pl.col("acro") == 1).then(pl.lit("acro"))
             .when(_nz("n_tset") < 0.5).then(pl.lit("alias"))
             .when((_ua == 0) & (_ub == 1)).then(pl.lit("add1"))
             .when((_ua == 0) & (_ub >= 2)).then(pl.lit("add2+"))
             .when((_ua == 1) & (_ub == 0)).then(pl.lit("drop1"))
             .when((_ua >= 2) & (_ub == 0)).then(pl.lit("drop2+"))
             .when((_ua == 1) & (_ub == 1)).then(pl.lit("swap1"))
             .otherwise(pl.lit("multi")))
AR = ["noaddr", "s1noaddr", "same", "same_nonum", "diffnum", "num_only", "other"]
NR = ["exact", "fuzzy_eq", "acro", "alias", "add1", "add2+", "drop1", "drop2+", "swap1", "multi"]
TR = ["t1", "t2+"]


def pair_types(feats: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    x = feats.join(scores, on=["s1_id", "tgt_id"], how="inner")
    x = x.with_columns(pl.col("p2").rank("ordinal", descending=True).over("tgt_id").alias("rk_t"))
    return x.with_columns(AR_EXPR.alias("ar"), NR_EXPR.alias("nr"),
                          pl.when(pl.col("rk_t") == 1).then(pl.lit("t1")).otherwise(pl.lit("t2+")).alias("tr")).select("s1_id", "tgt_id", "ar", "nr", "tr")


# ------------------------------------------------------------------ raw name descriptors
LEGAL = {"pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "pc", "pllc",
         "lp", "llc.", "plc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "selarl", "opc", "ltda", "pa", "psc"}
TOK = re.compile(r"[^\w]+", re.UNICODE)
PAREN = re.compile(r"\(([^)]*)\)")
G2 = ["rn_cf_eq", "rn_alnum_eq", "rn_reordered", "rn_order_frac", "rn_case_same", "rn_paren_eq", "rn_paren_n", "rn_legal_eq", "rn_legal_n",
      "rn_ins", "rn_del", "rn_sub", "rn_edge_ops", "rn_ops"]


def raw_record(name: str) -> tuple:
    name = name or ""
    cf = name.casefold()
    toks = [t for t in TOK.split(cf) if t]
    case = "U" if name.isupper() else "L" if name.islower() else "T" if name.istitle() else "M"
    paren = "|".join(p.strip().casefold() for p in PAREN.findall(name))
    legal = " ".join(t for t in toks if t in LEGAL)
    return cf, " ".join(toks), case, paren, legal


def order_frac(a: list, b: list) -> float:
    common = [t for t in a if t in set(b)]
    if len(common) < 2:
        return np.nan
    pos = {t: i for i, t in enumerate(b)}
    seq = [pos[t] for t in common]
    return sum(1 for i in range(len(seq) - 1) if seq[i] < seq[i + 1]) / (len(seq) - 1)


def raw_pair(a: tuple, b: tuple) -> tuple:
    ta, tb = a[1].split(), b[1].split()
    ops = Levenshtein.editops(a[1], b[1])
    ins = sum(1 for o in ops if o.tag == "insert"); dele = sum(1 for o in ops if o.tag == "delete"); sub = sum(1 for o in ops if o.tag == "replace")
    edge = sum(1 for o in ops if o.src_pos < 2 or o.src_pos >= len(a[1]) - 2)
    return (a[0] == b[0], a[1] == b[1], sorted(ta) == sorted(tb) and ta != tb, order_frac(ta, tb), a[2] == b[2],
            (a[3] == b[3]) if (a[3] or b[3]) else None, bool(a[3]) + bool(b[3]),
            (a[4] == b[4]) if (a[4] and b[4]) else None, bool(a[4]) + bool(b[4]), ins, dele, sub, edge, len(ops))


def _raw_chunk(rows):
    return [(s, t) + raw_pair(a, b) for s, t, a, b in rows]


def names(work_dir: Path, split: str, ids: pl.Series, srcs: tuple) -> dict:
    return dict(pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=["entity_id", "name"]) for k in srcs])
                .filter(pl.col("entity_id").is_in(ids.implode())).iter_rows())


def _pooled(fn, rows, n_jobs, initializer=None, initargs=()):
    chunks = [rows[i:i + 100_000] for i in range(0, len(rows), 100_000)]
    with ProcessPoolExecutor(n_jobs, initializer=initializer, initargs=initargs) as ex:
        return [r for part in ex.map(fn, chunks) for r in part]


def raw_name_features(work_dir: Path, split: str, pairs: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    n1 = {k: raw_record(v) for k, v in names(work_dir, split, pairs["s1_id"].unique(), (1,)).items()}
    n2 = {k: raw_record(v) for k, v in names(work_dir, split, pairs["tgt_id"].unique(), (2, 3)).items()}
    rows = [(s, t, n1[s], n2[t]) for s, t in pairs.select("s1_id", "tgt_id").iter_rows()]
    res = _pooled(_raw_chunk, rows, n_jobs)
    return pl.DataFrame(res, schema=["s1_id", "tgt_id"] + G2, orient="row", infer_schema_length=None)


# ------------------------------------------------------------------ copy-vs-decoy templates
VOW = set("aeiouy")
_KB = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KPOS = {c: (r, i) for r, row in enumerate(_KB) for i, c in enumerate(row)}
G8 = ["g8_accent", "g8_punct", "g8_dup", "g8_transp", "g8_vowel_sub", "g8_cons_sub", "g8_kbd_sub", "g8_digit_ops", "g8_inner_ops",
      "g8_first_kept", "g8_last_kept", "g8_lcs_minus_set", "g8_legal_llr", "g8_sub_llr", "g8_drop_llr", "g8_add_llr", "g8_drop_min", "g8_add_min", "g8_copy_llr"]
_TABLES = None


def _base(c: str) -> str:
    return unicodedata.normalize("NFKD", c)[0] if c else c


def _base_str(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def _kbd(a: str, b: str) -> bool:
    if a in KPOS and b in KPOS:
        (r1, i1), (r2, i2) = KPOS[a], KPOS[b]
        return abs(r1 - r2) <= 1 and abs(i1 - i2) <= 1
    return False


def _lcs(x: list, y: list) -> int:
    if not x or not y:
        return 0
    prev = [0] * (len(y) + 1)
    for i in range(len(x)):
        cur = [0] * (len(y) + 1)
        for j in range(len(y)):
            cur[j + 1] = prev[j] + 1 if x[i] == y[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[-1]


def transformation(n1: str, n2: str) -> tuple:
    """Typed edit counts and template keys of the raw-name transformation S1 -> target."""
    a, b = (n1 or "").casefold(), (n2 or "").casefold()
    cnt = dict(accent=0, punct=0, dup=0, transp=0, vowel=0, cons=0, kbd=0, digit=0, inner=0)
    subs, prev = [], None
    for o in Levenshtein.editops(a, b):
        ca = a[o.src_pos] if o.tag != "insert" and o.src_pos < len(a) else ""
        cb = b[o.dest_pos] if o.tag != "delete" and o.dest_pos < len(b) else ""
        ch = ca or cb
        if 2 <= o.src_pos < len(a) - 2:
            cnt["inner"] += 1
        if ch.isdigit():
            cnt["digit"] += 1
        if not ch.isalnum():
            cnt["punct"] += 1; prev = o; continue
        if o.tag == "replace":
            if _base(ca) == _base(cb):
                cnt["accent"] += 1
            else:
                subs.append(ca + ">" + cb)
                if prev is not None and prev.tag == "replace" and prev.src_pos == o.src_pos - 1 and a[prev.src_pos] == cb and b[prev.dest_pos] == ca:
                    cnt["transp"] += 1
                elif ca in VOW and cb in VOW:
                    cnt["vowel"] += 1
                else:
                    cnt["cons"] += 1
                if _kbd(ca, cb):
                    cnt["kbd"] += 1
        elif o.tag == "insert":
            if (o.src_pos < len(a) and a[o.src_pos] == cb) or (o.src_pos > 0 and a[o.src_pos - 1] == cb):
                cnt["dup"] += 1
        else:
            if (o.src_pos + 1 < len(a) and a[o.src_pos + 1] == ca) or (o.src_pos > 0 and a[o.src_pos - 1] == ca):
                cnt["dup"] += 1
        prev = o
    ba = [_base_str(t) for t in TOK.split(a) if t]; bb = [_base_str(t) for t in TOK.split(b) if t]
    la = " ".join(t for t in ba if t in LEGAL); lb = " ".join(t for t in bb if t in LEGAL)
    sa, sb = set(ba), set(bb)
    drop = [t for t in ba if t not in sb and t not in LEGAL]; add = [t for t in bb if t not in sa and t not in LEGAL]
    jac = len(sa & sb) / max(len(sa | sb), 1)
    first = float(bool(ba) and ba[0] in sb); last = float(bool(ba) and ba[-1] in sb)
    lcs_minus = (_lcs(ba, bb) / max(len(ba), len(bb), 1)) - jac
    return cnt, subs, la + ">" + lb, drop, add, first, last, lcs_minus


def _count_chunk(rows):
    c = {k: (Counter(), Counter()) for k in ("legal", "sub", "drop", "add")}
    for n1, n2, y in rows:
        _, subs, lg, drop, add, *_ = transformation(n1, n2)
        c["legal"][y][lg] += 1
        c["sub"][y].update(subs); c["drop"][y].update(drop); c["add"][y].update(add)
    return [c]


def template_tables(work_dir: Path, n_jobs: int, min_count: int = 5) -> dict:
    """Smoothed log P(template | copy) - log P(template | not a copy) from a 20% S1 sample of training folds A+B."""
    ab = (pl.scan_parquet(str(Path(work_dir) / "train" / "feats2" / "*.parquet"))
            .filter((pl.col("fold") != "H") & (pl.col("s1_id").hash(11) % 5 == 0)).select("s1_id", "tgt_id", "label").collect())
    n1 = names(work_dir, "train", ab["s1_id"].unique(), (1,)); n2 = names(work_dir, "train", ab["tgt_id"].unique(), (2, 3))
    rows = [(n1.get(s, ""), n2.get(t, ""), int(y)) for s, t, y in ab.iter_rows()]
    tot = {k: (Counter(), Counter()) for k in ("legal", "sub", "drop", "add")}
    for c in _pooled(_count_chunk, rows, n_jobs):
        for k in tot:
            tot[k][0].update(c[k][0]); tot[k][1].update(c[k][1])
    npos = sum(r[2] for r in rows); nneg = len(rows) - npos
    return {k: {key: math.log((c1[key] + 1) / (npos + 2)) - math.log((c0[key] + 1) / (nneg + 2)) for key in set(c0) | set(c1) if c0[key] + c1[key] >= min_count}
            for k, (c0, c1) in tot.items()}


def _init_tables(tables):
    global _TABLES
    _TABLES = tables


def _template_chunk(rows):
    out = []
    for s, t, n1, n2 in rows:
        cnt, subs, lg, drop, add, first, last, lcsm = transformation(n1, n2)
        l_legal = _TABLES["legal"].get(lg, 0.0)
        l_sub = sum(_TABLES["sub"].get(k, 0.0) for k in subs)
        l_drop = [_TABLES["drop"].get(k, 0.0) for k in drop]; l_add = [_TABLES["add"].get(k, 0.0) for k in add]
        out.append((s, t, cnt["accent"], cnt["punct"], cnt["dup"], cnt["transp"], cnt["vowel"], cnt["cons"], cnt["kbd"], cnt["digit"], cnt["inner"],
                    first, last, lcsm, l_legal, l_sub, sum(l_drop), sum(l_add), min(l_drop) if l_drop else 0.0, min(l_add) if l_add else 0.0,
                    l_legal + l_sub + sum(l_drop) + sum(l_add)))
    return out


def template_features(work_dir: Path, split: str, pairs: pl.DataFrame, tables: dict, n_jobs: int) -> pl.DataFrame:
    n1 = names(work_dir, split, pairs["s1_id"].unique(), (1,)); n2 = names(work_dir, split, pairs["tgt_id"].unique(), (2, 3))
    rows = [(s, t, n1.get(s, ""), n2.get(t, "")) for s, t in pairs.select("s1_id", "tgt_id").iter_rows()]
    res = _pooled(_template_chunk, rows, n_jobs, _init_tables, (tables,))
    return pl.DataFrame(res, schema=["s1_id", "tgt_id"] + G8, orient="row").with_columns([pl.col(c).cast(pl.Float32) for c in G8])


# ------------------------------------------------------------------ S1 score shape (leave-one-out)
G5 = ["g5_n05", "g5_n09", "g5_sum", "g5_ent", "g5_n09_same_src", "g5_n09_other_src", "g5_ncand", "g5_pmax_other"]


def score_shape(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_id, tgt_id, p2 -> summaries of the S1's other candidates, excluding the pair itself."""
    c = pl.col("p").clip(1e-6, 1 - 1e-6)
    src = pl.col("tgt_id").str.slice(0, 2).replace_strict({"S2": 2, "S3": 3}, default=0)
    x = pairs.select("s1_id", "tgt_id", pl.col("p2").alias("p")).with_columns(src.alias("src"))
    x = x.with_columns((pl.col("p") > 0.5).cast(pl.Int32).alias("h5"), (pl.col("p") > 0.9).cast(pl.Int32).alias("h9"),
                       (-(c * c.log() + (1 - c) * (1 - c).log())).alias("ent"))
    x = x.with_columns((pl.col("h5").sum().over("s1_id") - pl.col("h5")).alias("g5_n05"),
                       (pl.col("h9").sum().over("s1_id") - pl.col("h9")).alias("g5_n09"),
                       (pl.col("p").sum().over("s1_id") - pl.col("p")).alias("g5_sum"),
                       (pl.col("ent").sum().over("s1_id") - pl.col("ent")).alias("g5_ent"),
                       (pl.col("h9").sum().over("s1_id", "src") - pl.col("h9")).alias("g5_n09_same_src"),
                       (pl.len().over("s1_id") - 1).alias("g5_ncand"),
                       pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("rk"), pl.col("p").max().over("s1_id").alias("pmax"))
    x = x.with_columns((pl.col("g5_n09") - pl.col("g5_n09_same_src")).alias("g5_n09_other_src"))
    second = x.filter(pl.col("rk") == 2).select("s1_id", pl.col("p").alias("p2nd"))
    x = x.join(second, on="s1_id", how="left").with_columns(
        pl.when(pl.col("rk") == 1).then(pl.col("p2nd")).otherwise(pl.col("pmax")).fill_null(0).alias("g5_pmax_other"))
    return x.select(["s1_id", "tgt_id"] + G5).with_columns([pl.col(c).cast(pl.Float32) for c in G5])
