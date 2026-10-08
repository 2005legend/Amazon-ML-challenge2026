"""G9: a character-level cross-encoder over the raw records, trained from scratch (no pretrained weights) on
training folds A+B only, on the GPU when available.

Input per pair: [CLS] S1 name (40) | S1 address (72) | target name (40) | target address (72), raw case kept, one
token per character (the 252 most frequent training characters, others UNK). Positions restart in every slot, so
character i of the S1 name lines up with character i of the target name. Addresses longer than 72 characters keep
their first 55 and last 16 characters (the ZIP / PIN code survives). A 4-layer pre-norm transformer (d = 128,
0.87M parameters) reads the concatenation; the head sees the CLS state and the mean state.

Training pairs: A+B pairs whose stage-2 out-of-fold score is in [0.005, 0.995] (hard) plus a 15% hash sample of the
rest; 5% of A+B S1s are held out to pick the checkpoint. The holdout fold H and the test pairs are only scored.
The stacker uses the logit, its rank inside the S1 and its margin over the S1's best other candidate (G9).
GPU training is not bit-reproducible (atomic adds); the trained weights are saved in <work>/nn/model.pt.
"""
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from .prepare import norm_path

NAME_L, ADDR_L = 40, 72
REC = NAME_L + ADDR_L
T = 1 + 2 * REC
PAD, UNK, CLS, CUT = 0, 1, 2, 3
G9 = ["g9_logit", "g9_rank", "g9_gap"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def nn_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "nn"


# ------------------------------------------------------------------ encoding
def fixed_width() -> list:
    a = pl.col("address").fill_null("")
    a = pl.when(a.str.len_chars() > ADDR_L).then(a.str.slice(0, ADDR_L - 17) + "\x1f" + a.str.slice(-16, 16)).otherwise(a)
    return [pl.col("name").fill_null("").str.slice(0, NAME_L).str.pad_end(NAME_L, "\x00").alias("n"), a.str.pad_end(ADDR_L, "\x00").alias("a")]


def build_vocab(work_dir: Path, sample: int = 400_000, size: int = 252) -> np.ndarray:
    c = Counter()
    for k in (1, 2, 3):
        s = pl.read_parquet(norm_path(work_dir, "train", k), columns=["name", "address"]).sample(sample, seed=k)
        for col in ("name", "address"):
            for v in s[col].drop_nulls().to_list():
                c.update(v)
    lut = np.full(65536, UNK, np.uint8)
    lut[0], lut[0x1F] = PAD, CUT
    for i, (ch, _) in enumerate(c.most_common(size)):
        if ord(ch) < 65536 and ch not in ("\x00", "\x1f"):
            lut[ord(ch)] = 4 + i
    return lut


def encode(df: pl.DataFrame, lut: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """name, address -> uint8 [n, REC]."""
    out = np.empty((df.height, REC), np.uint8)
    for i in range(0, df.height, chunk):
        s = df.slice(i, chunk).select(fixed_width())
        for col, lo, L in (("n", 0, NAME_L), ("a", NAME_L, ADDR_L)):
            cp = np.frombuffer("".join(s[col].to_list()).encode("utf-32-le"), np.uint32).reshape(-1, L)
            out[i:i + s.height, lo:lo + L] = lut[np.minimum(cp, 65535)]
    return out


def entity_table(work_dir: Path, split: str, ids: pl.Series, lut: np.ndarray):
    need = pl.DataFrame({"entity_id": ids.unique()})
    ent = pl.concat([pl.read_parquet(norm_path(work_dir, split, k), columns=["entity_id", "name", "address"]).join(need, on="entity_id")
                     for k in (1, 2, 3)]).with_row_index("row")
    return encode(ent, lut), ent.select("entity_id", "row")


def attach_rows(pairs: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    return (pairs.join(ids.rename({"entity_id": "s1_id", "row": "r1"}), on="s1_id", how="left")
            .join(ids.rename({"entity_id": "tgt_id", "row": "r2"}), on="tgt_id", how="left"))


def batch_x(arr: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    x = np.empty((len(r1), T), np.uint8)
    x[:, 0] = CLS
    x[:, 1:1 + REC] = arr[r1]
    x[:, 1 + REC:] = arr[r2]
    return x


def prep(work_dir: Path) -> None:
    d = nn_dir(work_dir); d.mkdir(parents=True, exist_ok=True)
    if (d / "vocab.npy").exists():                        # keep the vocabulary the shipped weights were trained with
        lut = np.load(d / "vocab.npy")
    else:
        lut = build_vocab(work_dir); np.save(d / "vocab.npy", lut)
    p = pl.scan_parquet(str(Path(work_dir) / "train" / "p2" / "*.parquet")).select("s1_id", "tgt_id", "country", "label", "fold", "p2").collect()
    hard = pl.col("p2").is_between(0.005, 0.995)
    tr = p.filter(pl.col("fold") != "H").filter(hard | (pl.struct("s1_id", "tgt_id").hash(17) % 100 < 15)).with_columns(
        hard.alias("hard"), (pl.col("s1_id").hash(5) % 20 == 0).alias("val"))
    hp = p.filter(pl.col("fold") == "H").select("s1_id", "tgt_id", "country", "label", "p2")
    arr, ids = entity_table(work_dir, "train", pl.concat([tr["s1_id"], tr["tgt_id"], hp["s1_id"], hp["tgt_id"]]), lut)
    np.save(d / "ent_train.npy", arr)
    attach_rows(tr, ids).write_parquet(d / "train_pairs.parquet")
    attach_rows(hp, ids).write_parquet(d / "H_pairs.parquet")
    log("neural prep: train pairs", tr.height, "hard", int(tr["hard"].sum()), "| H pairs", hp.height, "| entities", arr.shape[0])


# ------------------------------------------------------------------ model
def make_model(d: int = 128, layers: int = 4, heads: int = 4, ff: int = 512):
    import torch
    from torch import nn

    class PairNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(256, d, padding_idx=PAD)
            s.seg = nn.Embedding(5, d)
            s.pos = nn.Embedding(ADDR_L, d)
            seg = [0] + [1] * NAME_L + [2] * ADDR_L + [3] * NAME_L + [4] * ADDR_L
            pos = [0] + list(range(NAME_L)) + list(range(ADDR_L)) + list(range(NAME_L)) + list(range(ADDR_L))
            s.register_buffer("segi", torch.tensor(seg), persistent=False)
            s.register_buffer("posi", torch.tensor(pos), persistent=False)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            s.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            s.norm = nn.LayerNorm(d)
            s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(s, x):
            mask = x == PAD
            h = s.emb(x) + s.seg(s.segi) + s.pos(s.posi)
            h = s.norm(s.enc(h, src_key_padding_mask=mask))
            keep = (~mask).unsqueeze(-1).to(h.dtype)
            mean = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
            return s.head(torch.cat([h[:, 0], mean], -1)).squeeze(-1)

    return PairNet()


def auc(y, s) -> float:
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def predict(model, arr, r1, r2, bs: int = 2048) -> np.ndarray:
    import torch
    model.eval(); out = np.empty(len(r1), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(r1), bs):
            x = torch.from_numpy(batch_x(arr, r1[i:i + bs], r2[i:i + bs])).cuda(non_blocking=True).long()
            out[i:i + bs] = model(x).float().cpu().numpy()
    return out


def train(work_dir: Path, epochs: float = 2.0, bs: int = 512, lr: float = 1e-3, layers: int = 4, seed: int = 7, tag: str = "") -> None:
    """Model <work>/nn/model{tag}.pt; the submitted G9 averages the logits of the listed models (see load_g9)."""
    import torch
    d = nn_dir(work_dir)
    torch.manual_seed(seed); np.random.seed(seed)
    arr = np.load(d / "ent_train.npy")
    tp = pl.read_parquet(d / "train_pairs.parquet")
    tr, va = tp.filter(~pl.col("val")), tp.filter(pl.col("val"))
    r1, r2, y = tr["r1"].to_numpy(), tr["r2"].to_numpy(), tr["label"].to_numpy().astype(np.float32)
    v1, v2, vy, vh = va["r1"].to_numpy(), va["r2"].to_numpy(), va["label"].to_numpy(), va["hard"].to_numpy()
    model = make_model(layers=layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps = int(math.ceil(len(y) / bs) * epochs); warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 1 - s / steps))
    scaler = torch.amp.GradScaler(); lossf = torch.nn.BCEWithLogitsLoss()
    step, best = 0, -1.0
    while step < steps:
        perm = np.random.permutation(len(y))
        for i in range(0, len(perm) - bs + 1, bs):
            if step >= steps:
                break
            j = perm[i:i + bs]
            x = torch.from_numpy(batch_x(arr, r1[j], r2[j])).cuda(non_blocking=True).long()
            t = torch.from_numpy(y[j]).cuda(non_blocking=True)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 3000 == 0 or step == steps:
                s = predict(model, arr, v1, v2)
                ah = auc(vy[vh], s[vh])
                log(f"neural step {step}/{steps}: val AUC {auc(vy, s):.5f} hard-AUC {ah:.5f}")
                if ah > best:
                    best = ah; torch.save(model.state_dict(), d / f"model{tag}.pt")


def score(work_dir: Path, tc_dir: Path, split: str, layers: int = 4, tag: str = "") -> pl.DataFrame:
    """Neural logits for the holdout pairs (split 'H') or every test pair (from the corrected test tables)."""
    import torch
    d = nn_dir(work_dir)
    model = make_model(layers=layers).cuda(); model.load_state_dict(torch.load(d / f"model{tag}.pt", weights_only=True))
    if split == "H":
        arr, p = np.load(d / "ent_train.npy"), pl.read_parquet(d / "H_pairs.parquet")
    else:
        p = pl.scan_parquet(str(Path(tc_dir) / "test" / "p3" / "*.parquet")).select("s1_id", "tgt_id").collect()
        arr, ids = entity_table(work_dir, "test", pl.concat([p["s1_id"], p["tgt_id"]]), np.load(d / "vocab.npy"))
        p = attach_rows(p, ids)
    out = p.select("s1_id", "tgt_id").with_columns(pl.Series("g9_logit", predict(model, arr, p["r1"].to_numpy(), p["r2"].to_numpy())))
    out.write_parquet(d / f"g9{tag}_{split}.parquet")
    log("neural scores", tag or "model", split, out.height)
    return out


def g9_context(g: pl.DataFrame) -> pl.DataFrame:
    """s1_id, tgt_id, g9_logit -> + rank inside the S1 (1 = best) and margin over the best other candidate."""
    g = g.with_columns(pl.col("g9_logit").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32).alias("g9_rank"))
    top2 = g.group_by("s1_id").agg(pl.col("g9_logit").top_k(2).alias("t"))
    g = g.join(top2, on="s1_id").with_columns(
        pl.when(pl.col("g9_logit") >= pl.col("t").list.get(0)).then(pl.col("g9_logit") - pl.col("t").list.get(1, null_on_oob=True))
        .otherwise(pl.col("g9_logit") - pl.col("t").list.get(0)).fill_null(10.0).cast(pl.Float32).alias("g9_gap"))
    return g.drop("t")


MODELS = (("", 2.0, 7), ("_b", 4.0, 11))   # (tag, epochs, seed): model A and model B; G9 = mean of their logits


def load_g9(work_dir: Path, split: str, tags: tuple = tuple(m[0] for m in MODELS)) -> pl.DataFrame:
    """G9 logit = mean of the listed models' logits (files <work>/nn/g9{tag}_{split}.parquet), then rank/margin context."""
    g = None
    for i, tag in enumerate(tags):
        f = pl.read_parquet(nn_dir(work_dir) / f"g9{tag}_{split}.parquet").rename({"g9_logit": f"l{i}"})
        g = f if g is None else g.join(f, on=["s1_id", "tgt_id"])
    g = g.with_columns((pl.sum_horizontal([f"l{i}" for i in range(len(tags))]) / len(tags)).alias("g9_logit"))
    return g9_context(g.select("s1_id", "tgt_id", "g9_logit"))
