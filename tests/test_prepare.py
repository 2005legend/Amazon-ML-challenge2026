import polars as pl
from ber.prepare import norm_path, prepare_split


def test_prepare_train_then_test(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    prepare_split("test", tiny_data, work, n_jobs=1)
    assert (work / "translit.json").exists()
    s2 = pl.read_parquet(norm_path(work, "train", 2))
    assert s2.height == 4 and s2["row"].to_list() == [0, 1, 2, 3]
    r = s2.filter(pl.col("entity_id") == "S2-2").row(0, named=True)
    assert "marketing" in r["name_core"].split() and r["legal"] == "ltd pvt"
    assert r["state"] == "mh" and r["name_core"].isascii()
    s3 = pl.read_parquet(norm_path(work, "train", 3))
    assert s3.filter(pl.col("entity_id") == "S3-1")["addr_empty"].item() is True
    fr = pl.read_parquet(norm_path(work, "test", 1)).filter(pl.col("country") == "France").row(0, named=True)
    assert fr["state"] == "naq" and fr["addr_norm"].startswith("175 blvd du president roosevelt")
