import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import MODEL_PATH

# Standard quota aliases
QUOTA_ALIASES = {
    "open": ["Open Quota", "State Quota"],
    "open quota": ["Open Quota", "State Quota"],
    "state quota": ["Open Quota", "State Quota"],
    "state": ["Open Quota", "State Quota"],
    "management": ["Private Management Quota"],
    "private management quota": ["Private Management Quota"],
    "private": ["Private Management Quota"],
    "in-service": ["In-Service"],
    "in-service dnb": ["In-Service DNB"],
    "nri": ["NRI Quota"],
    "nri quota": ["NRI Quota"],
}

# Standard category aliases
CATEGORY_ALIASES = {
    "ur": "General",
    "general": "General",
    "gen": "General",
    "sc": "SC",
    "st": "ST",
    "ews": "EWS",
    "obc-a": "OBC-A (Non-Creamy Layer)",
    "obca": "OBC-A (Non-Creamy Layer)",
    "obc-b": "OBC-B (Non-Creamy Layer)",
    "obcb": "OBC-B (Non-Creamy Layer)",
    "obc": "OBC (Non-Creamy Layer)",
    "general pwd": "General PwD",
    "ur pwd": "General PwD",
    "sc pwd": "SC PwD",
}


def normalize_quota(quota_str: str) -> list[str]:
    cleaned = quota_str.strip().lower()
    return QUOTA_ALIASES.get(cleaned, [quota_str])


def normalize_category(cat_str: str) -> str:
    cleaned = cat_str.strip().lower()
    return CATEGORY_ALIASES.get(cleaned, cat_str)


def compute_confidence(air: float, q10: float, q50: float, q90: float) -> float:
    """
    Computes a continuous, calibrated confidence score (0% to 100%)
    that candidate with rank `air` will be allotted a seat where:
    - q10 is the conservative cutoff (alpha=0.10, ~90% chance candidate beats it)
    - q50 is the median cutoff (alpha=0.50, ~50% chance candidate beats it)
    - q90 is the generous cutoff (alpha=0.90, ~10% chance candidate beats it)
    """
    # Enforce strict monotonic order
    q10 = max(1.0, q10)
    q50 = max(q10 + 1.0, q50)
    q90 = max(q50 + 1.0, q90)

    if air <= q10:
        # Better than the strict 10th percentile cutoff -> 90% to 99%
        prob = 0.90 + 0.09 * (1.0 - (air / q10))
    elif air <= q50:
        # Between 10th percentile and median -> 50% to 90%
        prob = 0.50 + 0.40 * ((q50 - air) / (q50 - q10))
    elif air <= q90:
        # Between median and generous 90th percentile -> 10% to 50%
        prob = 0.10 + 0.40 * ((q90 - air) / (q90 - q50))
    else:
        # Beyond 90th percentile cutoff -> tail decay (< 10%)
        excess = (air - q90) / q90
        prob = max(0.01, 0.10 * np.exp(-1.5 * excess))

    return round(float(prob * 100), 1)


def get_safety_badge(confidence: float) -> str:
    if confidence >= 80.0:
        return "Very Safe (High Chance)"
    elif confidence >= 50.0:
        return "Realistic (Moderate Chance)"
    elif confidence >= 25.0:
        return "Borderline (Competitive)"
    else:
        return "Dream (Low Chance)"


class SeatPredictor:
    def __init__(self, model_dir: Path | None = None):
        model_dir = model_dir or Path(MODEL_PATH)
        bundle_file = model_dir / "seat_predictor_bundle.joblib"
        catalog_file = model_dir / "seat_catalog.joblib"

        if not bundle_file.exists() or not catalog_file.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {model_dir}. Please run 'python src/train.py' first."
            )

        bundle = joblib.load(bundle_file)
        self.models = bundle["models"]
        self.feature_cols = bundle["feature_cols"]
        self.cat_categories = bundle["categorical_categories"]
        self.catalog: pd.DataFrame = joblib.load(catalog_file)

    def predict(
        self,
        air: int,
        candidate_category: str,
        allotted_quota: str,
        round_no: int = 1,
        min_confidence: float = 0.50,
        sort_by: str = "cutoff",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Predicts eligible college + branch combinations with dynamic confidence filtering.

        Parameters:
            air: Candidate All India Rank.
            candidate_category: General, SC, ST, EWS, OBC-A, OBC-B, etc.
            allotted_quota: Open Quota, In-Service, Private Management Quota, NRI Quota.
            round_no: Counselling Round (1, 2, or 3).
            min_confidence: Minimum probability threshold (e.g. 0.50 for 50%).
            sort_by: 'cutoff' (ranks most competitive seats first) or 'confidence' (safest first).
            top_n: Limit number of returned rows.
        """
        norm_cat = normalize_category(candidate_category)
        norm_quotas = normalize_quota(allotted_quota)

        # Filter available seats in the catalog
        matching_seats = self.catalog[
            (self.catalog["CANDIDATE CATEGORY"] == norm_cat)
            & (self.catalog["ALLOTTED QUOTA"].isin(norm_quotas))
        ].copy()

        # If strict match has 0 rows, fallback to candidate category
        if matching_seats.empty:
            matching_seats = self.catalog[
                (self.catalog["CANDIDATE CATEGORY"] == norm_cat)
            ].copy()

        if matching_seats.empty:
            print(f"Warning: No historical seats found for Category='{norm_cat}' and Quota='{allotted_quota}'")
            return pd.DataFrame()

        # Build feature matrix for model prediction
        query_df = pd.DataFrame({
            "INSTITUTE": matching_seats["INSTITUTE"].values,
            "COURSE": matching_seats["COURSE"].values,
            "CANDIDATE CATEGORY": matching_seats["CANDIDATE CATEGORY"].values,
            "ALLOTTED QUOTA": matching_seats["ALLOTTED QUOTA"].values,
            "ROUND": round_no,
        })

        # Match categorical dtypes from training
        for col in self.feature_cols:
            query_df[col] = pd.Categorical(
                query_df[col],
                categories=self.cat_categories[col]
            )

        # Predict cutoffs from quantile models
        pred_q10 = self.models["q_ambitious"].predict(query_df)
        pred_q50 = self.models["q_median"].predict(query_df)
        pred_q90 = self.models["q_safe"].predict(query_df)

        # Calculate confidence for each seat
        confidences = [
            compute_confidence(air, q10, q50, q90)
            for q10, q50, q90 in zip(pred_q10, pred_q50, pred_q90)
        ]

        results_df = pd.DataFrame({
            "INSTITUTE": matching_seats["INSTITUTE"].values,
            "COURSE": matching_seats["COURSE"].values,
            "CONFIDENCE (%)": confidences,
            "STATUS": [get_safety_badge(c) for c in confidences],
            "EST_CUTOFF (Median)": np.round(pred_q50).astype(int),
            "SAFE_CUTOFF (90%)": np.round(pred_q90).astype(int),
            "AMBITIOUS_CUTOFF (10%)": np.round(pred_q10).astype(int),
            "HIST_AVG_RANK": np.round(matching_seats["HISTORICAL_AVG_RANK"].values).astype(int),
            "HIST_TOTAL_SEATS": matching_seats["TOTAL_SEATS_ALLOTTED"].values,
        })

        # Drop duplicate college+branch offerings, keeping highest confidence
        results_df = results_df.sort_values(by="CONFIDENCE (%)", ascending=False)
        results_df = results_df.drop_duplicates(subset=["INSTITUTE", "COURSE"], keep="first")

        # Filter by minimum confidence threshold
        min_conf_pct = min_confidence * 100.0 if min_confidence <= 1.0 else min_confidence
        filtered_df = results_df[results_df["CONFIDENCE (%)"] >= min_conf_pct].copy()

        # Sorting: 'cutoff' puts top competitive branches first; 'confidence' puts safest first
        if sort_by == "cutoff":
            filtered_df = filtered_df.sort_values(
                by=["EST_CUTOFF (Median)", "CONFIDENCE (%)"],
                ascending=[True, False]
            ).reset_index(drop=True)
        else:
            filtered_df = filtered_df.sort_values(
                by=["CONFIDENCE (%)", "EST_CUTOFF (Median)"],
                ascending=[False, True]
            ).reset_index(drop=True)

        if top_n is not None and top_n > 0:
            filtered_df = filtered_df.head(top_n)

        return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Predict NEET PG seat allotment chances for West Bengal State Counselling."
    )
    parser.add_argument("--air", type=int, default=3500, help="Candidate All India Rank (AIR)")
    parser.add_argument(
        "--category",
        type=str,
        default="General",
        help="Candidate Category (e.g. General, SC, ST, EWS, OBC-A, OBC-B)",
    )
    parser.add_argument(
        "--quota",
        type=str,
        default="Open Quota",
        help="Allotted Quota (e.g. Open Quota, In-Service, Private Management Quota, NRI Quota)",
    )
    parser.add_argument("--round", type=int, default=2, help="Counselling Round (1, 2, or 3)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Minimum confidence threshold (e.g. 0.50 for >= 50%%)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="cutoff",
        choices=["cutoff", "confidence"],
        help="Sort order: 'cutoff' (top/competitive branches first) or 'confidence' (safest first)",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of top results to display")

    args = parser.parse_args()

    predictor = SeatPredictor()
    results = predictor.predict(
        air=args.air,
        candidate_category=args.category,
        allotted_quota=args.quota,
        round_no=args.round,
        min_confidence=args.min_confidence,
        sort_by=args.sort_by,
        top_n=args.top,
    )

    print("\n" + "=" * 80)
    print(f" NEET PG SEAT PREDICTIONS (AIR: {args.air:,} | Category: {args.category} | Quota: {args.quota} | Round: {args.round})")
    print(f" Showing seats with Confidence >= {args.min_confidence * 100:.0f}%")
    print("=" * 80)

    if results.empty:
        print("No colleges found meeting the confidence threshold. Try lowering --min-confidence or checking higher rounds.")
    else:
        print(f"Found {len(results)} matching seat options:\n")
        display_cols = [
            "INSTITUTE",
            "COURSE",
            "CONFIDENCE (%)",
            "STATUS",
            "EST_CUTOFF (Median)",
            "SAFE_CUTOFF (90%)",
        ]
        print(results[display_cols].to_string(index=False))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
