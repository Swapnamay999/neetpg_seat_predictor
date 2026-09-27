import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tabulate import tabulate

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import MODEL_PATH
from src.predict import SeatPredictor, CATEGORY_ALIASES, QUOTA_ALIASES
from src.districts import ALL_DISTRICTS, INSTITUTE_TO_DISTRICT, get_district

app = typer.Typer(
    name="neetpg-predict",
    help="🩺 West Bengal NEET PG Seat Allotment Predictor CLI",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()

# Pre-defined choices for validation
VALID_CATEGORIES = [
    "General",
    "SC",
    "ST",
    "EWS",
    "OBC-A",
    "OBC-B",
    "OBC",
    "General PwD",
    "SC PwD",
]

VALID_QUOTAS = [
    "Open Quota",
    "In-Service",
    "Private Management Quota",
    "In-Service DNB",
    "NRI Quota",
]


def format_status(status_str: str) -> str:
    """Adds terminal colors to status badges."""
    if "Very Safe" in status_str:
        return f"[bold green]🟢 {status_str}[/bold green]"
    elif "Realistic" in status_str:
        return f"[bold yellow]🟡 {status_str}[/bold yellow]"
    elif "Borderline" in status_str:
        return f"[bold magenta]🟠 {status_str}[/bold magenta]"
    else:
        return f"[bold red]🔴 {status_str}[/bold red]"


@app.command(name="predict", help="🔍 Predict eligible seats based on your AIR and profile.")
def predict(
    air: Optional[int] = typer.Option(
        None,
        "--air",
        "-a",
        help="All India Rank (AIR)",
    ),
    category: str = typer.Option(
        "General",
        "--category",
        "-c",
        help="Candidate Category (e.g. General, SC, ST, EWS, OBC-A, OBC-B)",
    ),
    quota: str = typer.Option(
        "Open Quota",
        "--quota",
        "-q",
        help="Allotted Quota (e.g. Open Quota, In-Service, Private Management Quota)",
    ),
    round_no: int = typer.Option(
        2,
        "--round",
        "-r",
        help="Counselling Round (1, 2, or 3)",
    ),
    min_confidence: float = typer.Option(
        50.0,
        "--min-confidence",
        "-m",
        help="Minimum confidence threshold percentage (e.g. 50.0 for >= 50%)",
    ),
    district: Optional[str] = typer.Option(
        None,
        "--district",
        "-d",
        help="Filter results by district (e.g. 'Kolkata', 'Darjeeling', 'Burdwan', 'Nadia')",
    ),
    sort_by: str = typer.Option(
        "cutoff",
        "--sort-by",
        "-s",
        help="Sort by 'cutoff' (top competitive branches first) or 'confidence' (safest first)",
    ),
    top: int = typer.Option(
        25,
        "--top",
        "-n",
        help="Maximum number of seat recommendations to display",
    ),
    course_filter: Optional[str] = typer.Option(
        None,
        "--course",
        help="Filter results by course substring (e.g. 'Medicine', 'Surgery', 'Radio')",
    ),
    college_filter: Optional[str] = typer.Option(
        None,
        "--college",
        help="Filter results by college substring (e.g. 'Kolkata', 'Burdwan')",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        "-e",
        help="Optional path to export predictions as a CSV file",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive questionnaire wizard",
    ),
):
    """
    Run NEET PG Seat Prediction with beautiful tabular output.
    """
    # Interactive wizard if requested or if AIR wasn't provided
    if interactive or air is None:
        console.print(
            Panel(
                "[bold cyan]🩺 West Bengal NEET PG Seat Allotment Predictor[/bold cyan]\n"
                "[dim]Answer a few quick questions to estimate your allotment chances.[/dim]",
                border_style="cyan",
            )
        )
        if air is None:
            air = typer.prompt("👉 Enter your All India Rank (AIR)", type=int)

        rprint(f"[dim]Available categories: {', '.join(VALID_CATEGORIES)}[/dim]")
        category = typer.prompt("👉 Enter your Category", default=category)

        rprint(f"[dim]Available quotas: {', '.join(VALID_QUOTAS)}[/dim]")
        quota = typer.prompt("👉 Enter your Quota", default=quota)

        round_no = typer.prompt("👉 Select Counselling Round (1, 2, 3)", default=round_no, type=int)
        min_confidence = typer.prompt("👉 Minimum Confidence %", default=min_confidence, type=float)

        rprint(f"[dim]Common districts: Kolkata, North 24 Parganas, Paschim Bardhaman, Darjeeling (or Enter for all)[/dim]")
        dist_input = typer.prompt("👉 Filter by District (optional, press Enter to skip)", default="")
        if dist_input.strip():
            district = dist_input.strip()

    # Convert percentage to decimal if > 1.0
    conf_decimal = min_confidence / 100.0 if min_confidence > 1.0 else min_confidence

    with console.status("[bold cyan]Loading model and predicting seat cutoffs...[/bold cyan]"):
        try:
            predictor = SeatPredictor()
            df = predictor.predict(
                air=air,
                candidate_category=category,
                allotted_quota=quota,
                round_no=round_no,
                min_confidence=conf_decimal,
                district=district,
                sort_by=sort_by,
                top_n=None,  # Filter before truncating
            )
        except Exception as e:
            console.print(f"[bold red]Error running prediction:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Apply optional course or college substring filters
    if course_filter and not df.empty:
        df = df[df["COURSE"].str.contains(course_filter, case=False, na=False)]
    if college_filter and not df.empty:
        df = df[df["INSTITUTE"].str.contains(college_filter, case=False, na=False)]

    # Limit to top N
    df_top = df.head(top).copy()

    # Header Card
    console.print()
    header_parts = [
        f"[bold]AIR:[/bold] [yellow]{air:,}[/yellow]",
        f"[bold]Category:[/bold] [cyan]{category}[/cyan]",
        f"[bold]Quota:[/bold] [green]{quota}[/green]",
        f"[bold]Round:[/bold] [magenta]{round_no}[/magenta]",
        f"[bold]Threshold:[/bold] [yellow]≥ {min_confidence:.0f}%[/yellow]",
    ]
    if district:
        header_parts.append(f"[bold]District:[/bold] [yellow]{district}[/yellow]")
    header_parts.append(f"[bold]Sort:[/bold] [cyan]{sort_by.capitalize()}[/cyan]")

    header_text = "   ".join(header_parts)
    console.print(Panel(header_text, title="🎯 Prediction Parameters", border_style="blue"))

    if df_top.empty:
        console.print(
            Panel(
                f"[yellow]No seats found with Confidence ≥ {min_confidence:.0f}%.[/yellow]\n\n"
                "[dim]💡 Suggestions:\n"
                "  • Try lowering the threshold: [cyan]--min-confidence 30[/cyan]\n"
                "  • Check a later round: [cyan]--round 3[/cyan]\n"
                "  • Remove district or course filters[/dim]",
                title="Result",
                border_style="yellow",
            )
        )
        return

    # Build Tabulate Data
    table_data = []
    for idx, row in df_top.iterrows():
        table_data.append([
            len(table_data) + 1,
            row["INSTITUTE"],
            row["DISTRICT"],
            row["COURSE"],
            f"{row['CONFIDENCE (%)']:.1f}%",
            row["STATUS"],
            f"{row['EST_CUTOFF (Median)']:,}",
            f"{row['SAFE_CUTOFF (90%)']:,}",
        ])

    headers = [
        "#",
        "Institute",
        "District",
        "Course / Speciality",
        "Confidence",
        "Safety Status",
        "Est. Cutoff",
        "Safe Cutoff",
    ]

    # Render with tabulate (rounded_outline style)
    rendered_table = tabulate(
        table_data,
        headers=headers,
        tablefmt="rounded_outline",
        colalign=("center", "left", "left", "left", "right", "left", "right", "right"),
    )
    console.print(rendered_table)

    # Summary Stats
    total_found = len(df)
    very_safe_count = len(df[df["CONFIDENCE (%)"] >= 80.0])
    realistic_count = len(df[(df["CONFIDENCE (%)"] >= 50.0) & (df["CONFIDENCE (%)"] < 80.0)])

    summary_text = (
        f"📊 [bold]Found {total_found} eligible seats with ≥ {min_confidence:.0f}% confidence.[/bold] "
        f"(Showing top {len(df_top)})\n"
        f"   🟢 [green]{very_safe_count} Very Safe[/green]   "
        f"   🟡 [yellow]{realistic_count} Realistic/Target[/yellow]"
    )
    console.print(Panel(summary_text, border_style="dim"))

    # Export to CSV if requested
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export, index=False)
        console.print(f"[bold green]✔ Exported {len(df)} predictions to:[/bold green] {export}")


@app.command(name="metrics", help="📈 View LightGBM model evaluation metrics and validation results.")
def show_metrics():
    """
    Display model performance (R², RMSE, MAE, CV scores).
    """
    metrics_file = Path(MODEL_PATH) / "metrics.json"
    if not metrics_file.exists():
        console.print("[bold red]Metrics file not found. Run 'python src/train.py' first.[/bold red]")
        raise typer.Exit(code=1)

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    table = Table(title="🚀 LightGBM Model Performance", border_style="cyan")
    table.add_column("Evaluation Protocol", style="bold white")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Interpretation", style="dim")

    table.add_row(
        "Temporal Split (2024 ➔ 2025)",
        "R² Score",
        f"{metrics.get('temporal_r2', 0):.4f}",
        "Out-of-time future prediction accuracy",
    )
    table.add_row(
        "Temporal Split (2024 ➔ 2025)",
        "Adjusted R²",
        f"{metrics.get('temporal_adj_r2', 0):.4f}",
        "Penalized for feature count",
    )
    table.add_row(
        "Temporal Split (2024 ➔ 2025)",
        "RMSE",
        f"{metrics.get('temporal_rmse', 0):,.1f}",
        "Standard deviation of rank residuals",
    )
    table.add_row(
        "Temporal Split (2024 ➔ 2025)",
        "MAE",
        f"{metrics.get('temporal_mae', 0):,.1f}",
        "Average absolute error in rank",
    )
    table.add_section()
    table.add_row(
        "5-Fold Cross Validation",
        "Mean R²",
        f"{metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}",
        "Overall variance explained across all folds",
    )
    table.add_row(
        "5-Fold Cross Validation",
        "Mean Adjusted R²",
        f"{metrics.get('cv_adj_r2_mean', 0):.4f}",
        "Robust cross-validated metric",
    )
    table.add_row(
        "5-Fold Cross Validation",
        "Mean MAE",
        f"{metrics.get('cv_mae_mean', 0):,.1f}",
        "Average cross-validated rank error",
    )

    console.print(table)


@app.command(name="catalog", help="📋 Inspect unique colleges, courses, categories, and quotas in data.")
def show_catalog():
    """
    Display statistics about the historical seat catalog.
    """
    catalog_file = Path(MODEL_PATH) / "seat_catalog.joblib"
    if not catalog_file.exists():
        console.print("[bold red]Catalog file not found. Run 'python src/train.py' first.[/bold red]")
        raise typer.Exit(code=1)

    import joblib
    catalog: pd.DataFrame = joblib.load(catalog_file)

    n_institutes = catalog["INSTITUTE"].nunique()
    n_courses = catalog["COURSE"].nunique()
    n_categories = catalog["CANDIDATE CATEGORY"].nunique()
    n_quotas = catalog["ALLOTTED QUOTA"].nunique()
    total_seat_combos = len(catalog)

    summary_panel = (
        f"[bold]Total Unique Seats:[/bold] [yellow]{total_seat_combos:,}[/yellow]\n"
        f"[bold]Medical Colleges:[/bold]   [cyan]{n_institutes}[/cyan]\n"
        f"[bold]PG Courses/Branches:[/bold][green]{n_courses}[/green]\n"
        f"[bold]Quotas Supported:[/bold]   [magenta]{n_quotas}[/magenta]\n"
        f"[bold]Categories:[/bold]         [blue]{n_categories}[/blue]"
    )
    console.print(Panel(summary_panel, title="🏛️ WB NEET PG Seat Catalog Summary", border_style="cyan"))


@app.command(name="districts", help="🗺️ List all West Bengal districts and their medical institutions.")
def list_districts():
    """
    List all West Bengal districts and the medical colleges in each.
    """
    from collections import defaultdict
    by_district = defaultdict(list)
    for inst, dist in sorted(INSTITUTE_TO_DISTRICT.items()):
        by_district[dist].append(inst)

    table = Table(title="🗺️ West Bengal Medical Institutions by District", border_style="cyan")
    table.add_column("District", style="bold cyan", no_wrap=True)
    table.add_column("Count", style="yellow", justify="center", no_wrap=True)
    table.add_column("Medical Colleges & Hospitals", style="white")

    for dist in sorted(by_district.keys()):
        inst_list = "\n".join(f"• {inst}" for inst in sorted(by_district[dist]))
        table.add_row(dist, str(len(by_district[dist])), inst_list)
        table.add_section()

    console.print(table)


if __name__ == "__main__":
    app()

