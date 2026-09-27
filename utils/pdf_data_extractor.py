from pandas.core.frame import DataFrame
from pathlib import Path
import re
import subprocess

import pandas as pd
import tabula
from tqdm import tqdm

DOCS_PATH: list[Path] = list(
    (Path(__file__).resolve().parent.parent / "docs").glob("*.pdf")
)

DATA_PATH = Path(__file__).resolve().parent.parent/"data"


def extract_pdf_title(pdf_path: Path) -> str:
    """Extracts the title/counselling round from the top of page 1."""
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-", "-f", "1", "-l", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    # Page 1 top lines:
    # line 0: "PROVISIONAL SEAT ALLOTMENT RESULT ROUND-2"
    # line 1: "(WB PG MEDICAL COUNSELLING 2025)"
    # line 2-3: Legend (UR=Unreserved, EWS=Economically Weaker Section...)

    title_parts = lines[:2]
    combined_title = " ".join(title_parts)

    # Create a safe, clean filename (e.g., WB_PG_MEDICAL_COUNSELLING_2025_ROUND_2)
    clean_name = re.sub(r"[^\w\s-]", "", combined_title)
    clean_name = re.sub(r"[-\s]+", "_", clean_name).strip("_")
    return clean_name


def read_pdf_to_dataframe(file_path: Path) -> pd.DataFrame:
    dataframes = tabula.read_pdf(file_path, pages="all", lattice=True)

    cleaned_dfs: list[pd.DataFrame] = []

    for df in tqdm(dataframes, desc="Processing pages"):
        df.columns = [" ".join(col.split()) for col in df.columns]

        df = df.map(
            lambda x: " ".join(x.split()) if isinstance(x, str) else x
        )
        cleaned_dfs.append(df)

    if not cleaned_dfs:
        return pd.DataFrame()

    output_dataframe = pd.concat(cleaned_dfs, ignore_index=True)
    print(output_dataframe.describe())
    print(output_dataframe.info())
    return output_dataframe

def save_output_to_csv(file_paths:list[Path])->None:

    for file_path in tqdm(file_paths):
        file_name: str = extract_pdf_title(file_path)
        df: DataFrame = read_pdf_to_dataframe(file_path)
        df.to_csv(DATA_PATH/f"{file_name}.csv")

if __name__ == "__main__":
    save_output_to_csv(DOCS_PATH)
