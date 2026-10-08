"""Write the two submission files and check them with the official validator."""
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl

from .blocking import load_candidates
from .io import write_id_lists
from .prepare import norm_path


def write_outputs(work_dir: Path, output_dir: Path, matches: pl.DataFrame) -> tuple[Path, Path]:
    s1_ids = pl.read_parquet(norm_path(work_dir, "test", 1), columns=["entity_id"])["entity_id"]
    cands = load_candidates(work_dir, "test", "cands").select("s1_id", "tgt_id")
    outside = matches.select("s1_id", "tgt_id").join(cands, on=["s1_id", "tgt_id"], how="anti")
    if outside.height:
        raise ValueError(f"{outside.height} matches are not in the candidate set")
    out = Path(output_dir)
    match_path, cand_path = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    write_id_lists(cand_path, s1_ids, cands, "candidate_entity_ids")
    write_id_lists(match_path, s1_ids, matches.select("s1_id", "tgt_id"), "matched_entity_ids")
    return match_path, cand_path


def validate(match_path: Path, cand_path: Path, data_dir: Path) -> int:
    validator = Path(data_dir).parent / "utils" / "validate_submission.py"
    cmd = [sys.executable, str(validator), "--matching", str(match_path), "--candidate", str(cand_path),
           "--test-dir", str(Path(data_dir) / "test"), "--check-ids"]
    return subprocess.run(cmd, check=False).returncode


def build_zip(project_root: Path, output_dir: Path, doc_path: Path, team: str) -> Path:
    project_root = Path(project_root)
    zip_path = project_root.parent / f"{team}_submission.zip"
    code = "code/business_entity_resolution"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(output_dir) / name, f"output/{name}")
        for name in ("README.md", "requirements.txt", "pytest.ini", "pyproject.toml"):
            z.write(project_root / name, f"{code}/{name}")
        for sub in ("src", "tests"):
            for f in sorted((project_root / sub).rglob("*.py")):
                z.write(f, f"{code}/{f.relative_to(project_root).as_posix()}")
        for f in sorted((project_root / "weights").glob("*")) if (project_root / "weights").exists() else []:
            z.write(f, f"{code}/weights/{f.name}")
        z.write(doc_path, "Documentation_template.md")
    return zip_path
