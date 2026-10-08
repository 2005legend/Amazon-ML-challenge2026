"""Central configuration: paths, seeds and tunable hyperparameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR",
    ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))
TC_DIR = Path(os.environ.get("BER_TC_DIR", WORK_DIR / "tc"))  # density-corrected test tables (ber.testshift)

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 2)
BETA = 0.5
SOURCES = (1, 2, 3)


def source_path(split: str, src: int) -> Path:
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def work_path(split: str, name: str) -> Path:
    path = WORK_DIR / split
    path.mkdir(parents=True, exist_ok=True)
    return path / name
