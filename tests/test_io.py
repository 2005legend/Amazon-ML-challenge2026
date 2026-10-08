import polars as pl
from ber.io import read_source, read_ground_truth, write_id_lists


def test_read_source_keeps_empty_strings(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                 'S2-1\tA "quoted", name\t\tIndia\n', encoding="utf-8")
    df = read_source(p)
    assert df.columns == ["entity_id", "name", "address", "country"]
    assert df.row(0) == ("S2-1", 'A "quoted", name', "", "India")


def test_read_ground_truth_explodes_and_drops_empty(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5,S3-7\nS1-2\t\n", encoding="utf-8")
    gt = read_ground_truth(p)
    assert gt.sort("tgt_id").rows() == [("S1-1", "S2-5"), ("S1-1", "S3-7")]


def test_write_id_lists_one_row_per_s1(tmp_path):
    p = tmp_path / "out.tsv"
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1"], "tgt_id": ["S2-5", "S3-7", "S2-5"]})
    write_id_lists(p, pl.Series(["S1-1", "S1-2"]), pairs, "matched_entity_ids")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-5,S3-7", "S1-2\t"]
