# 🩺 West Bengal NEET PG Seat Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20Quantile-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

An intelligent, machine-learning-powered seat allotment prediction engine for **West Bengal NEET PG State Medical Counselling**. Trained on official round-wise allotment records across 2024 and 2025, this tool predicts the probability of securing every available postgraduate medical seat (MD/MS/Diploma/NBEMS) based on a candidate's **All India Rank (AIR)**, **Category**, **Quota**, and **Round**.

---

## 🌟 Key Features

- **Empirical Cutoff Modeling**: Solves the multi-class probability dilution problem by using **Multi-Quantile LightGBM Regression** ($\alpha = 0.1, 0.5, 0.9$) on historical closing ranks.
- **Dynamic Calibrated Confidence Scores (0% – 100%)**:
  - 🟢 **Very Safe ($\ge 80\%$)**: Better than strict 10th percentile cutoffs.
  - 🟡 **Realistic / Target ($50\% - 79\%$)**: Within expected median cutoff bounds.
  - 🟠 **Borderline / Competitive ($25\% - 49\%$)**: Near upper-tail historical cutoffs.
  - 🔴 **Dream / Reach ($< 25\%$)**: Unlikely based on past patterns.
- **Branch Classification**: One-click toggle between **Clinical** (e.g. Medicine, Surgery, Pediatrics, Radio-Diagnosis) and **Non-Clinical / Para-Clinical** (e.g. Pathology, Pharmacology, Microbiology, Anatomy).
- **Column-Level Filters**: Instant multi-select and search for specific **Institute Names**, **Specialities / Subjects**, and **Branch Types** directly above the recommendation table.
- **District-Specific Filtering**: Comprehensive mapping of all **64 medical colleges and hospitals** across West Bengal's **21 administrative districts**.
- **Interactive Web App**: Modern Streamlit interface with Plotly analytics (safety distribution donut charts, clinical vs non-clinical charts, cutoff vs AIR comparisons, district availability bars).
- **Rich Terminal CLI**: Interactive CLI powered by **Typer**, **Tabulate**, and **Rich**.
- **High Predictive Power**: Mean 5-Fold Cross-Validation **$R^2 = 0.8399$** (Adjusted $R^2 = 0.8389$) and Out-of-Time Temporal **$R^2 = 0.7389$** on unseen future counselling rounds.

---

## 🏗️ Architecture & Directory Structure

```text
neetpg_seat_predictor/
├── app.py                     # Streamlit web application
├── main.py                    # Root CLI entrypoint
├── config.py                  # Global path and directory configurations
├── requirements.txt           # Deployment and runtime dependencies
├── data/
│   ├── processed_cutoffs.csv  # 4,277 consolidated cutoff records (2024 & 2025)
│   └── PROVISIONAL_*.csv      # Raw student allotment records
├── model/
│   ├── seat_predictor_bundle.joblib  # Trained LightGBM multi-quantile models
│   ├── seat_catalog.joblib           # 2,129 category/quota specific seats
│   └── metrics.json                  # Model validation and evaluation scores
├── src/
│   ├── cli.py                 # Typer + Tabulate CLI application
│   ├── districts.py           # Complete WB institute-to-district mapping
│   ├── predict.py             # Inference engine and confidence calculator
│   ├── prepare_data.py        # Data cleaning, normalization, and aggregation
│   └── train.py               # Model training & cross-validation pipeline
└── utils/
    └── pdf_data_extractor.py  # Tabula-py PDF table extraction utility
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Swapnamay999/neetpg_seat_predictor.git
cd neetpg_seat_predictor

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 Terminal CLI Usage

Run the CLI using `python main.py`:

### Interactive Wizard
```bash
python main.py predict --interactive
```
Walks you step-by-step through entering your AIR, Category, Quota, Round, and District.

### Direct Prediction Queries
```bash
# Predict Round 2 open quota options for AIR 3,000 with >= 50% confidence
python main.py predict --air 3000 --category General --quota "Open Quota" --round 2 --min-confidence 50

# Filter specifically for Surgery seats in Kolkata
python main.py predict --air 3000 --category General --round 2 --district Kolkata --course Surgery --top 10

# Export results to CSV
python main.py predict --air 15000 --category SC --round 2 --export my_seats.csv
```

### View Model Validation & Catalog
```bash
# Display R², Adjusted R², RMSE, and cross-validation metrics
python main.py metrics

# Inspect the 21 West Bengal districts and their constituent medical colleges
python main.py districts

# View seat database summary
python main.py catalog
```

---

## 🔬 Model Performance & Methodology

### Multi-Quantile LightGBM Architecture
Rather than treating seat allotment as a naive multi-class classification problem (which dilutes probabilities across 466 options), we model the **closing rank threshold** for every seat under quantile loss (Pinball Loss):

$$L_\alpha(y, \hat{y}) = \max(\alpha(y - \hat{y}), (\alpha - 1)(y - \hat{y}))$$

| Model | Quantile ($\alpha$) | Role | Confidence Threshold |
| :--- | :---: | :--- | :---: |
| **Ambitious** | `0.10` | Conservative Cutoff Bound | Candidate AIR $\le q_{0.1} \implies \ge 90\%$ |
| **Median** | `0.50` | Expected Median Cutoff | Candidate AIR $\approx q_{0.5} \implies \approx 50\%$ |
| **Safe** | `0.90` | Generous Cutoff Bound | Candidate AIR $\approx q_{0.9} \implies \approx 10\%$ |

### Validation Results

| Protocol | Metric | Value | Interpretation |
| :--- | :--- | :---: | :--- |
| **Temporal Split (2024 $\to$ 2025)** | $R^2$ Score | **`0.7389`** | Real-world future year prediction accuracy |
| **Temporal Split (2024 $\to$ 2025)** | Adjusted $R^2$ | **`0.7383`** | Penalized for feature count |
| **5-Fold Cross Validation** | Mean $R^2$ | **`0.8399` $\pm$ `0.0044`** | Variance explained across full dataset |
| **5-Fold Cross Validation** | Mean Adjusted $R^2$ | **`0.8389`** | Cross-validated goodness-of-fit |
| **5-Fold Cross Validation** | Mean Rank MAE | **`14,596.9`** | Average deviation across all ranks (1 to 140k+) |

---

## ☁️ Deploying to Streamlit Cloud (Free Tier)

This repository is ready for one-click continuous deployment on **Streamlit Community Cloud**:

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete NEET PG seat predictor"
   git push origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"Create app"** ➔ **"Yup, I have an app"**.
4. Set:
   - **Repository**: `Swapnamay999/neetpg_seat_predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy!**
6. *Continuous Deployment*: Every future `git push origin main` will automatically update the live site.

---

## ⚠️ Disclaimer

> [!WARNING]
> This application is an independent decision-support tool powered by machine learning on historical allotment patterns. Cutoffs vary each year depending on seat matrix additions/deletions, bond policy changes, and candidate preferences. Always refer to official notifications from the **West Bengal Medical Counselling Committee (WBMCC)** for binding counselling decisions.

---

## 📄 License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE v3](LICENSE).
