import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MODEL_PATH
from src.predict import SeatPredictor
from src.districts import ALL_DISTRICTS, get_district
from src.cli import VALID_CATEGORIES, VALID_QUOTAS

# Page configuration
st.set_page_config(
    page_title="WB NEET PG Seat Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-mod {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    """Cache the predictor instance to avoid reloading models on every interaction."""
    return SeatPredictor()


@st.cache_data
def load_metrics():
    metrics_file = Path(MODEL_PATH) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}


predictor = load_predictor()
metrics_data = load_metrics()

# Available choices
all_courses = sorted(predictor.catalog["COURSE"].unique())

# -------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stethoscope.png", width=64)
    st.header("🎯 Candidate Profile")

    air_input = st.number_input(
        "All India Rank (AIR)",
        min_value=1,
        max_value=250000,
        value=3500,
        step=50,
        help="Enter your NEET PG All India Rank as printed on your score card.",
    )

    category_input = st.selectbox(
        "Candidate Category",
        options=VALID_CATEGORIES,
        index=0,
        help="Select your reservation category in West Bengal counselling.",
    )

    quota_input = st.selectbox(
        "Quota Eligibility",
        options=VALID_QUOTAS,
        index=0,
        help="Select your quota (e.g. Open/State Quota, In-Service, Management Quota).",
    )

    round_input = st.radio(
        "Counselling Round",
        options=[1, 2, 3],
        index=1,
        horizontal=True,
        help="Cutoffs typically expand in later rounds as seats shuffle.",
    )

    st.markdown("---")
    st.subheader("🔍 Filters & Sorting")

    selected_districts = st.multiselect(
        "Filter by District(s)",
        options=ALL_DISTRICTS,
        default=[],
        placeholder="All West Bengal Districts",
        help="Select specific districts like Kolkata, Darjeeling, etc.",
    )

    selected_courses = st.multiselect(
        "Filter Specialities / Branches",
        options=all_courses,
        default=[],
        placeholder="All Medical Specialities",
        help="Select branches of interest (e.g. Medicine, Surgery, Radio-Diagnosis).",
    )

    confidence_threshold = st.slider(
        "Minimum Confidence Threshold (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="Only display seats where your probability of admission meets this threshold.",
    )

    sort_order = st.radio(
        "Sort Options By",
        options=["Most Competitive (Cutoff Rank)", "Safest First (Highest Confidence)"],
        index=0,
    )

    top_limit = st.slider(
        "Max Results to Display",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
    )

# -------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------
st.markdown('<div class="main-title">🩺 West Bengal NEET PG Seat Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered seat allotment prediction using Quantile LightGBM models trained on official 2024 & 2025 counselling data.</div>',
    unsafe_allow_html=True,
)

# Run Inference
sort_by_param = "cutoff" if "Competitive" in sort_order else "confidence"
results_df = predictor.predict(
    air=air_input,
    candidate_category=category_input,
    allotted_quota=quota_input,
    round_no=round_input,
    min_confidence=confidence_threshold / 100.0,
    sort_by=sort_by_param,
    top_n=None,
)

# Apply District filter if any selected
if selected_districts and not results_df.empty:
    results_df = results_df[results_df["DISTRICT"].isin(selected_districts)]

# Apply Course filter if any selected
if selected_courses and not results_df.empty:
    results_df = results_df[results_df["COURSE"].isin(selected_courses)]

# Compute KPI Metrics
total_seats = len(results_df)
safe_seats = len(results_df[results_df["CONFIDENCE (%)"] >= 80.0]) if not results_df.empty else 0
realistic_seats = len(results_df[(results_df["CONFIDENCE (%)"] >= 50.0) & (results_df["CONFIDENCE (%)"] < 80.0)]) if not results_df.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Eligible Seats Found", f"{total_seats:,}", help=f"Seats with confidence ≥ {confidence_threshold}%")
col2.metric("🟢 Very Safe (≥ 80%)", f"{safe_seats:,}", help="High probability allotments")
col3.metric("🟡 Realistic (50% - 79%)", f"{realistic_seats:,}", help="Competitive target seats")
col4.metric("Candidate AIR", f"{air_input:,}", f"{category_input} | R{round_input}")

# TABS
tab1, tab2, tab3 = st.tabs(["📋 Seat Recommendations", "📊 Visual Analytics", "📈 Model Metrics & Limitations"])

# -------------------------------------------------------------
# TAB 1: SEAT RECOMMENDATIONS TABLE
# -------------------------------------------------------------
with tab1:
    if results_df.empty:
        st.warning(
            f"No seats found matching your criteria with Confidence ≥ {confidence_threshold}%.\n\n"
            "**Try:**\n"
            "- Lowering the confidence threshold slider in the sidebar.\n"
            "- Checking a later counselling round (e.g. Round 2 or Round 3).\n"
            "- Removing any restrictive district or course filters."
        )
    else:
        st.caption(f"Displaying top {min(top_limit, len(results_df))} of {len(results_df)} matching seat combinations.")

        display_df = results_df.head(top_limit)[[
            "INSTITUTE",
            "DISTRICT",
            "COURSE",
            "CONFIDENCE (%)",
            "STATUS",
            "EST_CUTOFF (Median)",
            "SAFE_CUTOFF (90%)",
            "AMBITIOUS_CUTOFF (10%)",
        ]].copy()

        # Format column names for presentation
        display_df.columns = [
            "Medical Institute",
            "District",
            "Speciality / Course",
            "Confidence",
            "Safety Tier",
            "Median Cutoff",
            "Safe Cutoff (90%)",
            "Ambitious Cutoff (10%)",
        ]

        # Interactive Dataframe with formatting
        st.dataframe(
            display_df.style.format({
                "Confidence": "{:.1f}%",
                "Median Cutoff": "{:,}",
                "Safe Cutoff (90%)": "{:,}",
                "Ambitious Cutoff (10%)": "{:,}",
            }),
            use_container_width=True,
            hide_index=True,
            height=480,
        )

        # Download CSV Button
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Prediction List as CSV",
            data=csv_data,
            file_name=f"neetpg_wb_predictions_air_{air_input}_round_{round_input}.csv",
            mime="text/csv",
        )

# -------------------------------------------------------------
# TAB 2: VISUAL ANALYTICS
# -------------------------------------------------------------
with tab2:
    if results_df.empty:
        st.info("No data available to plot. Adjust your filters to generate recommendations.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Safety Tier Distribution Donut Chart
        with chart_col1:
            st.subheader("🎯 Safety Tier Distribution")
            status_counts = results_df["STATUS"].value_counts().reset_index()
            status_counts.columns = ["Safety Tier", "Seats Count"]

            color_map = {
                "Very Safe (High Chance)": "#22C55E",
                "Realistic (Moderate Chance)": "#EAB308",
                "Borderline (Competitive)": "#F97316",
                "Dream (Low Chance)": "#EF4444",
            }

            fig_donut = px.pie(
                status_counts,
                values="Seats Count",
                names="Safety Tier",
                hole=0.45,
                color="Safety Tier",
                color_discrete_map=color_map,
            )
            fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_donut, use_container_width=True)

        # Chart 2: District-wise Seat Availability
        with chart_col2:
            st.subheader("🗺️ Seats by District")
            district_counts = (
                results_df["DISTRICT"]
                .value_counts()
                .head(8)
                .reset_index()
            )
            district_counts.columns = ["District", "Available Seats"]

            fig_dist = px.bar(
                district_counts,
                x="Available Seats",
                y="District",
                orientation="h",
                color="Available Seats",
                color_continuous_scale="Blues",
            )
            fig_dist.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Chart 3: Candidate AIR vs Predicted Cutoff for Top 10 Options
        st.subheader("📊 Your AIR vs Predicted Cutoffs (Top 10 Choices)")
        st.caption("Green region shows how much your rank beats the expected closing cutoff.")

        top10_plot = results_df.head(10).copy()
        top10_plot["SEAT_LABEL"] = (
            top10_plot["INSTITUTE"].str.split().str[:3].str.join(" ")
            + " - "
            + top10_plot["COURSE"].str.replace("M.D. ", "").str.replace("M.S. ", "")
        )

        fig_cutoffs = go.Figure()

        # Median Cutoff Bar
        fig_cutoffs.add_trace(
            go.Bar(
                name="Predicted Median Cutoff",
                x=top10_plot["EST_CUTOFF (Median)"],
                y=top10_plot["SEAT_LABEL"],
                orientation="h",
                marker_color="#93C5FD",
            )
        )

        # Candidate AIR Line
        fig_cutoffs.add_vline(
            x=air_input,
            line_width=3,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Your AIR ({air_input:,})",
            annotation_position="top right",
        )

        fig_cutoffs.update_layout(
            barmode="group",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Rank (Lower rank is better)",
            height=380,
            margin=dict(t=30, b=30, l=20, r=20),
        )
        st.plotly_chart(fig_cutoffs, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MODEL METRICS & TRANSPARENCY
# -------------------------------------------------------------
with tab3:
    st.subheader("🔬 Machine Learning Model Card")
    st.write(
        """
        This seat predictor is built using **LightGBM (Light Gradient Boosting Machine)** equipped with 
        **Quantile Loss (Pinball Loss)** to evaluate cutoff distributions rather than arbitrary classification.
        """
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("5-Fold Cross-Val R²", f"{metrics_data.get('cv_r2_mean', 0.8399):.4f}", "± 0.0044")
    m2.metric("Mean Adjusted R²", f"{metrics_data.get('cv_adj_r2_mean', 0.8389):.4f}", "Robust CV")
    m3.metric("Out-of-Time R² (2024➔2025)", f"{metrics_data.get('temporal_r2', 0.7389):.4f}", "Future Test")
    m4.metric("Mean Rank MAE", f"±{metrics_data.get('cv_mae_mean', 14596.9):,.0f} ranks", "Across all tiers")

    st.markdown("---")
    st.subheader("⚙️ How the Confidence Score is Calculated")
    st.markdown(
        r"""
        Unlike naive models that treat seat prediction as a single-label problem, this engine models the **empirical cutoff distribution** for each seat:
        - **$q_{0.1}$ (10th percentile cutoff)**: If your AIR is better than this strict cutoff, you have **$\ge 90\%$ confidence (Very Safe)**.
        - **$q_{0.5}$ (Median expected cutoff)**: If your AIR is close to this rank, you have **$\approx 50\%$ confidence (Realistic)**.
        - **$q_{0.9}$ (90th percentile cutoff)**: If your AIR only qualifies in generous years, you have **$10\% - 49\%$ confidence (Ambitious)**.
        """
    )

    st.markdown("---")
    st.subheader("⚠️ Limitations & Disclaimer")
    st.warning(
        """
        **Please Read Before Choice Filling:**
        1. **Seat Matrix Changes**: Medical colleges occasionally add new PG seats or lose recognition. Cutoffs adjust accordingly.
        2. **Candidate Preference Shifts**: Macro trends (e.g. rising popularity of Radio-Diagnosis or Emergency Medicine) shift cutoffs between years.
        3. **Service Bond Policies**: State rural service bond rules directly impact how top candidates rank state government colleges vs private/DNB institutes.
        4. **Not Official Counselling**: This tool is an independent AI decision-support utility. Always consult the official **West Bengal Health & Family Welfare Department (WBMCC)** notifications for official rounds.
        """
    )
