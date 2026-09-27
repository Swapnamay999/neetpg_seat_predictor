from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_PATH: list[Path] = list((BASE_DIR / "docs").glob("*.pdf"))
DATA_PATH = BASE_DIR / "data"

MODEL_PATH = BASE_DIR / "model"

