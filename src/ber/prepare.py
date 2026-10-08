"""Normalize every source file of a split into Parquet, in parallel chunks."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from .io import read_ground_truth, read_source
from .normalize import normalize_address, normalize_name
from .translit import Transliterator, learn_component_map, learn_token_map

NAME_COLS = ["name_core", "name_compact", "name_parts", "legal", "initials", "name_indic"]
ADDR_COLS = ["addr_norm", "addr_words", "addr_nums", "num_first", "state", "addr_indic", "addr_empty"]
INDIC_CLASS = "[ऀ-ൿ]"
_T: Transliterator | None = None


def norm_path(work_dir: Path, split: str, src: int) -> Path:
    return Path(work_dir) / split / f"norm_s{src}.parquet"


def _init(translit_path: str) -> None:
    global _T
    _T = Transliterator.load(translit_path)


def _normalize_chunk(args):
    names, addrs, countries = args
    return ([normalize_name(n, _T.token) for n in names],
            [normalize_address(a, c, _T.component) for a, c in zip(addrs, countries)])


def learn_translit(data_dir: Path, out_path: Path) -> Transliterator:
    gt = read_ground_truth(Path(data_dir) / "train" / "train_ground_truth.tsv")
    s1 = read_source(Path(data_dir) / "train" / "train_source1.tsv")
    frames = []
    for src in (2, 3):
        s = read_source(Path(data_dir) / "train" / f"train_source{src}.tsv").filter(
            pl.col("name").str.contains(INDIC_CLASS) | pl.col("address").str.contains(INDIC_CLASS))
        frames.append(gt.join(s, left_on="tgt_id", right_on="entity_id")
                        .join(s1, left_on="s1_id", right_on="entity_id", suffix="_s1")
                        .select("name_s1", "name", "address_s1", "address", "country"))
    j = pl.concat(frames)
    t = Transliterator(learn_token_map(zip(j["name_s1"].to_list(), j["name"].to_list())),
                       learn_component_map(zip(j["country"].to_list(), j["address_s1"].to_list(),
                                               j["address"].to_list())))
    t.save(out_path)
    return t


NORM_SCHEMA = {**{c: pl.Utf8 for c in NAME_COLS + ADDR_COLS},
               "name_indic": pl.Boolean, "addr_indic": pl.Boolean, "addr_empty": pl.Boolean}


def _normalize_frame(df: pl.DataFrame, translit_path: Path, n_jobs: int, chunk: int = 50_000,
                     window: int = 1_000_000) -> pl.DataFrame:
    """Normalize in bounded windows so peak memory stays well below the 16 GB budget."""
    def run(mapper) -> pl.DataFrame:
        parts = []
        for w0 in range(0, df.height, window):
            w = df.slice(w0, window)
            tasks = [(w["name"][i:i + chunk].to_list(), w["address"][i:i + chunk].to_list(),
                      w["country"][i:i + chunk].to_list()) for i in range(0, w.height, chunk)]
            results = list(mapper(_normalize_chunk, tasks))
            names = pl.DataFrame([r for res in results for r in res[0]], schema=NAME_COLS, orient="row")
            addrs = pl.DataFrame([r for res in results for r in res[1]], schema=ADDR_COLS, orient="row")
            parts.append(names.hstack(addrs).cast(NORM_SCHEMA))
        return pl.concat(parts) if parts else pl.DataFrame(schema=NORM_SCHEMA)

    if n_jobs == 1:
        _init(str(translit_path))
        return df.hstack(run(map))
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(str(translit_path),)) as ex:
        return df.hstack(run(ex.map))


def prepare_split(split: str, data_dir: Path, work_dir: Path, n_jobs: int) -> None:
    work_dir = Path(work_dir)
    translit_path = work_dir / "translit.json"
    if not translit_path.exists():
        if split != "train":
            raise FileNotFoundError("run `prepare --split train` first: it learns translit.json")
        work_dir.mkdir(parents=True, exist_ok=True)
        learn_translit(data_dir, translit_path)
    for src in (1, 2, 3):
        df = read_source(Path(data_dir) / split / f"{split}_source{src}.tsv")
        df = df.with_row_index("row").with_columns(pl.col("row").cast(pl.Int32))
        out = _normalize_frame(df.select("entity_id", "country", "row", "name", "address"),
                               translit_path, n_jobs)
        path = norm_path(work_dir, split, src)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)
