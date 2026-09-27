import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DATA_PATH


def add_counselling_year(file_path:Path)->pd.DataFrame:
    
    counselling_year: int = int(file_path.stem.split('_')[-1])
    counselling_round : int = int(file_path.stem.split('_')[5])

    df = pd.read_csv(file_path)
    df["YEAR"] = counselling_year
    df["ROUND"] = counselling_round

    return df

def filter_dataframe(df:pd.DataFrame)->pd.DataFrame:

    df = df[df["STATUS"]!="Not Upgraded"]

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean and normalize string columns
    str_cols = [
        "INSTITUTE",
        "COURSE",
        "ALLOTTED QUOTA",
        "ALLOTTED CATEGORY",
        "CANDIDATE CATEGORY",
    ]
    for col in str_cols:
        if col in df.columns:
            # Collapse whitespace and strip
            df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            # Normalize 'Non- Creamy' spacing to 'Non-Creamy'
            df[col] = df[col].str.replace(r"Non-\s+", "Non-", regex=True)

    # Standardize quota naming variations
    quota_mapping = {
        "In-service Quota": "In-Service",
        "In-service DNB Quota": "In-Service DNB",
    }
    df["ALLOTTED QUOTA"] = df["ALLOTTED QUOTA"].replace(quota_mapping)

    # 2. Ensure ALL INDIA RANK is clean integer
    df["ALL INDIA RANK"] = pd.to_numeric(df["ALL INDIA RANK"], errors="coerce")
    df = df.dropna(subset=["ALL INDIA RANK"])
    df["ALL INDIA RANK"] = df["ALL INDIA RANK"].astype(int)

    return df


def aggregate_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "YEAR",
        "ROUND",
        "INSTITUTE",
        "COURSE",
        "ALLOTTED QUOTA",
        "CANDIDATE CATEGORY",
        "ALLOTTED CATEGORY",
    ]

    cutoffs_df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            OPENING_RANK=("ALL INDIA RANK", "min"),
            CLOSING_RANK=("ALL INDIA RANK", "max"),
            SEATS_ALLOTTED=("ALL INDIA RANK", "count"),
        )
    )

    # Sort for consistency
    cutoffs_df = cutoffs_df.sort_values(
        by=["YEAR", "ROUND", "CLOSING_RANK"]
    ).reset_index(drop=True)

    return cutoffs_df


if __name__ == "__main__":
    all_data_files = sorted(Path(DATA_PATH).glob("PROVI*.csv"))

    processed_dfs: list[pd.DataFrame] = []
    for file_path in all_data_files:
        print(f"Processing: {file_path.name}")
        df = add_counselling_year(file_path)
        df = filter_dataframe(df)
        df = clean_dataframe(df)
        processed_dfs.append(df)

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    cutoffs_df = aggregate_cutoffs(combined_df)

    output_path = Path(DATA_PATH) / "processed_cutoffs.csv"
    cutoffs_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully created {output_path} with {len(cutoffs_df)} aggregated cutoff rows.")
    print(cutoffs_df.head())