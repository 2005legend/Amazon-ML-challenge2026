import shutil

import polars as pl
import pytest
from ber import config
from ber.blocking import BlockConfig, build_candidates
from ber.package import validate, write_outputs
from ber.prepare import prepare_split
from ber.prerank import apply_prerank

SMALL = BlockConfig({"n": 2, "a": 2, "na": 2}, {"n": 1, "a": 1, "na": 1}, 0.01, 5)


@pytest.fixture
def test_work(tiny_data, tmp_path):
    work = tmp_path / "work"
    prepare_split("train", tiny_data, work, n_jobs=1)
    prepare_split("test", tiny_data, work, n_jobs=1)
    build_candidates("test", work, SMALL)
    apply_prerank("test", work, None)
    return work


def test_writer_includes_every_s1(test_work, tmp_path):
    matches = pl.DataFrame({"s1_id": ["S1-5"], "tgt_id": ["S2-5"]})
    m, c = write_outputs(test_work, tmp_path / "output", matches)
    rows = dict(line.split("\t") for line in m.read_text(encoding="utf-8").splitlines()[1:])
    assert set(rows) == {"S1-1", "S1-2", "S1-3", "S1-4", "S1-5"}
    assert rows["S1-5"] == "S2-5" and rows["S1-4"] == ""


def test_match_outside_candidates_is_rejected(test_work, tmp_path):
    with pytest.raises(ValueError):
        write_outputs(test_work, tmp_path / "output", pl.DataFrame({"s1_id": ["S1-4"], "tgt_id": ["S2-1"]}))


def test_written_files_pass_validator_rules(test_work, tiny_data, tmp_path):
    utils = tiny_data.parent / "utils"
    utils.mkdir(exist_ok=True)
    shutil.copy(config.DATA_DIR.parent / "utils" / "validate_submission.py", utils / "validate_submission.py")
    m, c = write_outputs(test_work, tmp_path / "output", pl.DataFrame({"s1_id": ["S1-5"], "tgt_id": ["S2-5"]}))
    assert validate(m, c, tiny_data) == 0


def test_build_zip_layout(tmp_path):
    import zipfile
    from ber.package import build_zip
    root = tmp_path / "business_entity_resolution"
    (root / "src" / "ber").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "ber" / "cli.py").write_text("x = 1\n")
    (root / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    for name in ("README.md", "requirements.txt", "pytest.ini", "pyproject.toml"):
        (root / name).write_text(name)
    out = root / "output"
    out.mkdir()
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        (out / name).write_text("h\n")
    doc = root / "Documentation_template.md"
    doc.write_text("# doc\n")
    z = build_zip(root, out, doc, "team_x")
    names = set(zipfile.ZipFile(z).namelist())
    assert z.name == "team_x_submission.zip"
    assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
            "code/business_entity_resolution/pyproject.toml",
            "code/business_entity_resolution/src/ber/cli.py",
            "code/business_entity_resolution/tests/test_x.py"} <= names
