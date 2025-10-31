"""Generate exploratory insights and visualisations for ACECQA education and care services.

This module focuses on three analysis pillars recommended in the assignment brief:

1. **Service quality** – distribution of ratings by state and service type, including the
   relationship between ratings and proximity to public transport.
2. **Accessibility and coverage** – spatial distribution of services and identification of
   potential gaps in access to transport infrastructure.
3. **Operational trends** – capacity, approval timelines, and operating hour signals by
   location and quality segment.

The script expects the CSV file `Education-services-with-station-access_loc.csv` to be stored
inside the repository's `data/` directory. Visualisations are exported to the `outputs/`
folder so they can be used directly in presentation slides.

Usage
-----
Run the module as a script once the dataset has been downloaded:

```
python -m analysis.acecqa_analysis
```

All plots will be saved into the `outputs/` directory. The script prints a short textual
summary to the console that can be used to guide narration in the accompanying video.

The implementation is intentionally defensive – the data dictionary can vary slightly
between state downloads, so column names are detected via pattern matching rather than hard
coded references. When a required column is missing the script raises a clear error message
explaining how to resolve the issue.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, MutableMapping, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:  # GeoPandas is optional – some environments may not have GEOS/PROJ installed.
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover - only triggered when GeoPandas is unavailable.
    gpd = None
    Point = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DATA_FILENAME = "Education-services-with-station-access_loc.csv"


@dataclass
class ColumnHints:
    """Container for flexible column lookups used across the analysis pipeline."""

    state: tuple[str, ...] = ("state", "jurisdiction")
    service_type: tuple[str, ...] = ("service type", "service_category")
    overall_rating: tuple[str, ...] = ("overall rating", "overallrating")
    quality_area_prefix: tuple[str, ...] = ("quality area", "qualityarea")
    approval_date: tuple[str, ...] = ("approval date", "granted")
    capacity: tuple[str, ...] = ("capacity", "places")
    latitude: tuple[str, ...] = ("latitude", "lat", "y coordinate")
    longitude: tuple[str, ...] = ("longitude", "lon", "lng", "x coordinate")
    suburb: tuple[str, ...] = ("suburb", "town")
    postcode: tuple[str, ...] = ("postcode", "postal")
    train_distance_km: tuple[str, ...] = ("train distance", "distance to train")
    bus_distance_km: tuple[str, ...] = ("bus distance", "distance to bus")
    operating_hours: tuple[str, ...] = ("operating hours", "hours")


COLUMN_HINTS = ColumnHints()

RATING_ORDER = [
    "Significant Improvement Required",
    "Working Towards National Quality Standard",
    "Meeting National Quality Standard",
    "Exceeding National Quality Standard",
    "Excellent",
]
RATING_SCORE = {rating: idx for idx, rating in enumerate(RATING_ORDER, start=1)}

sns.set_theme(style="whitegrid")


def get_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str:
    """Return the column whose name contains one of the candidate substrings."""

    lowered = {col.lower(): col for col in df.columns}
    for pattern in candidates:
        for lookup, original in lowered.items():
            if pattern in lookup:
                return original
    if required:
        raise KeyError(
            "Could not locate a column matching any of the patterns: "
            f"{', '.join(candidates)}. Please verify the dataset schema."
        )
    return ""


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the ACECQA dataset from disk, normalising column names for ease of use."""

    if path is None:
        path = DATA_DIR / DATA_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Download 'Education-services-with-station-access_loc.csv' "
            "and place it inside the repository's data/ directory."
        )

    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that simplify downstream analysis."""

    df = df.copy()

    approval_col = get_column(df, COLUMN_HINTS.approval_date, required=False)
    if approval_col:
        df["approval_date"] = pd.to_datetime(df[approval_col], errors="coerce")
        df["approval_year"] = df["approval_date"].dt.year
    else:
        df["approval_date"] = pd.NaT
        df["approval_year"] = np.nan

    rating_col = get_column(df, COLUMN_HINTS.overall_rating, required=False)
    if rating_col:
        df["overall_rating"] = df[rating_col].fillna("Unknown").astype(str)
        df["overall_rating"] = pd.Categorical(
            df["overall_rating"], categories=RATING_ORDER + ["Unknown"], ordered=True
        )
        df["overall_rating_score"] = df["overall_rating"].map(RATING_SCORE).fillna(0)
    else:
        df["overall_rating"] = "Unknown"
        df["overall_rating_score"] = 0

    capacity_col = get_column(df, COLUMN_HINTS.capacity, required=False)
    if capacity_col:
        df["approved_capacity"] = pd.to_numeric(df[capacity_col], errors="coerce")
    else:
        df["approved_capacity"] = np.nan

    for feature_name, candidates in (
        ("state", COLUMN_HINTS.state),
        ("service_type", COLUMN_HINTS.service_type),
        ("suburb", COLUMN_HINTS.suburb),
        ("postcode", COLUMN_HINTS.postcode),
        ("train_distance_km", COLUMN_HINTS.train_distance_km),
        ("bus_distance_km", COLUMN_HINTS.bus_distance_km),
        ("operating_hours", COLUMN_HINTS.operating_hours),
        ("latitude", COLUMN_HINTS.latitude),
        ("longitude", COLUMN_HINTS.longitude),
    ):
        col = get_column(df, candidates, required=False)
        if col:
            df[feature_name] = df[col]
        else:
            df[feature_name] = np.nan

    return df


def summarise_quality_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return normalised percentages of quality ratings per group (state or service)."""

    grouped = (
        df.groupby([group_col, "overall_rating"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = grouped.groupby(group_col)["count"].transform("sum")
    grouped["share"] = grouped["count"] / totals
    return grouped


def rating_by_transport(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics between transport access and quality rating."""

    metrics: list[MutableMapping[str, float | str]] = []
    for label, distance_col in {
        "Nearest train station": "train_distance_km",
        "Nearest bus stop": "bus_distance_km",
    }.items():
        if distance_col not in df or df[distance_col].isna().all():
            continue
        valid = df[[distance_col, "overall_rating_score"]].dropna()
        if valid.empty:
            continue
        metrics.append(
            {
                "transport_type": label,
                "median_distance_km": valid[distance_col].median(),
                "rating_correlation": valid[distance_col].corr(valid["overall_rating_score"]),
                "services_analyzed": len(valid),
            }
        )
    return pd.DataFrame(metrics)


def plot_quality_by_state(df: pd.DataFrame) -> Path:
    summary = summarise_quality_by_group(df, "state")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=summary,
        x="state",
        y="share",
        hue="overall_rating",
        ax=ax,
    )
    ax.set(
        title="Quality rating mix by state",
        xlabel="State or Territory",
        ylabel="Share of services",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(title="Overall rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "quality_by_state.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_quality_by_service_type(df: pd.DataFrame) -> Path:
    summary = summarise_quality_by_group(df, "service_type")
    top_types = summary.groupby("service_type")["count"].sum().nlargest(8).index
    filtered = summary[summary["service_type"].isin(top_types)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=filtered,
        x="share",
        y="service_type",
        hue="overall_rating",
        ax=ax,
        orient="h",
    )
    ax.set(
        title="Top service types by quality distribution",
        xlabel="Share of services",
        ylabel="Service type",
        xlim=(0, 1),
    )
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(title="Overall rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "quality_by_service_type.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_transport_vs_rating(df: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    for label, distance_col in {
        "Train": "train_distance_km",
        "Bus": "bus_distance_km",
    }.items():
        if distance_col not in df or df[distance_col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(
            data=df,
            x=distance_col,
            y="overall_rating_score",
            ax=ax,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "#d62728"},
        )
        ax.set(
            title=f"Overall rating score vs. distance to nearest {label.lower()} stop",
            xlabel=f"Distance to nearest {label.lower()} (km)",
            ylabel="Overall rating score (ordinal)",
        )
        output_path = OUTPUT_DIR / f"rating_vs_{label.lower()}_distance.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def plot_capacity_vs_transport(df: pd.DataFrame) -> Path | None:
    if df["approved_capacity"].isna().all():
        return None
    distance_col = "train_distance_km" if "train_distance_km" in df else "bus_distance_km"
    if distance_col not in df or df[distance_col].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=distance_col,
        y="approved_capacity",
        hue="overall_rating",
        ax=ax,
        alpha=0.6,
    )
    ax.set(
        title="Capacity vs. proximity to public transport",
        xlabel=f"Distance to nearest {distance_col.split('_')[0]} stop (km)",
        ylabel="Approved capacity (places)",
    )
    ax.legend(title="Overall rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "capacity_vs_transport.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_spatial_distribution(df: pd.DataFrame) -> Optional[Path]:
    if gpd is None or Point is None:
        print("GeoPandas is not available – skipping spatial visualisation.")
        return None

    if df[["longitude", "latitude"]].isna().any(axis=None):
        filtered = df.dropna(subset=["longitude", "latitude"])
    else:
        filtered = df

    if filtered.empty:
        print("No coordinates found – skipping spatial visualisation.")
        return None

    gdf = gpd.GeoDataFrame(
        filtered,
        geometry=[Point(xy) for xy in zip(filtered["longitude"], filtered["latitude"])],
        crs="EPSG:4326",
    )

    australia = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    australia = australia[australia["name"] == "Australia"]

    fig, ax = plt.subplots(figsize=(8, 8))
    australia.to_crs(epsg=3577).plot(ax=ax, color="#f0f0f0", edgecolor="#999999")
    gdf.to_crs(epsg=3577).plot(
        ax=ax,
        column="overall_rating",
        markersize=15,
        legend=True,
        legend_kwds={"title": "Overall rating", "loc": "lower left"},
        alpha=0.6,
    )
    ax.set_axis_off()
    ax.set_title("Spatial distribution of services and overall ratings", pad=12)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "spatial_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_operating_hours_distribution(df: pd.DataFrame) -> Optional[Path]:
    if "operating_hours" not in df or df["operating_hours"].isna().all():
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="overall_rating",
        y="operating_hours",
        ax=ax,
    )
    ax.set(
        title="Operating hour spread by quality rating",
        xlabel="Overall rating",
        ylabel="Reported weekly operating hours",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "operating_hours_by_rating.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def describe_trends(df: pd.DataFrame) -> str:
    """Produce a text summary to help structure presentation talking points."""

    lines: list[str] = []

    state_summary = summarise_quality_by_group(df, "state")
    best_states = (
        state_summary[state_summary["overall_rating"] == "Exceeding National Quality Standard"]
        .sort_values("share", ascending=False)
        .head(3)
    )
    worst_states = (
        state_summary[state_summary["overall_rating"] == "Working Towards National Quality Standard"]
        .sort_values("share", ascending=False)
        .head(3)
    )

    def _format_rows(frame: pd.DataFrame) -> str:
        return ", ".join(
            f"{row['state']}: {row['share']:.0%}" for _, row in frame.iterrows() if row["share"] > 0
        )

    if not best_states.empty:
        lines.append("Top performing states (share of Exceeding services): " + _format_rows(best_states))
    if not worst_states.empty:
        lines.append(
            "States needing support (share Working Towards): " + _format_rows(worst_states)
        )

    transport_summary = rating_by_transport(df)
    for _, row in transport_summary.iterrows():
        corr = row["rating_correlation"]
        if pd.notna(corr):
            direction = "positive" if corr > 0 else "negative"
            lines.append(
                f"Correlation between rating and {row['transport_type'].lower()} distance is "
                f"{corr:.2f} ({direction})."
            )

    capacity_mean = df["approved_capacity"].mean()
    if not np.isnan(capacity_mean):
        lines.append(f"Average approved capacity: {capacity_mean:,.0f} places.")

    approval_counts = df["approval_year"].value_counts().sort_index()
    if not approval_counts.empty:
        latest_year = approval_counts.index.max()
        lines.append(
            f"Most recent approvals concentrated in {int(latest_year)} with "
            f"{int(approval_counts.loc[latest_year])} new services."
        )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    df = enrich_data(df)

    exported: dict[str, Path | list[Path] | None] = {
        "quality_by_state": plot_quality_by_state(df),
        "quality_by_service_type": plot_quality_by_service_type(df),
        "transport_vs_rating": plot_transport_vs_rating(df),
        "capacity_vs_transport": plot_capacity_vs_transport(df),
        "spatial_distribution": plot_spatial_distribution(df),
        "operating_hours": plot_operating_hours_distribution(df),
    }

    summary = describe_trends(df)
    if summary:
        print("Key talking points:\n" + summary)
    else:
        print("Analysis complete. Refer to the generated figures for insights.")

    generated = {
        key: value
        for key, value in exported.items()
        if value is not None and (not isinstance(value, list) or value)
    }
    print("\nExported artefacts:")
    for key, path in generated.items():
        if isinstance(path, list):
            for idx, sub_path in enumerate(path, start=1):
                print(f"  - {key}[{idx}]: {sub_path.relative_to(REPO_ROOT)}")
        else:
            print(f"  - {key}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
