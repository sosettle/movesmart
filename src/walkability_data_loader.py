"""
CBSA-level walkability features from EPA Smart Location Database.

Use load_cbsa_walkability_dataset() in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CBSA_COL = "CBSA"
POP_COL = "TotPop"

MEAN_FEATURES = {
    "NatWalkInd": "nat_walk_index_mean",
    "D1B": "population_density_mean",
    "D1D": "activity_density_mean",
    "D3B": "intersection_density_mean",
}


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Return a NaN-safe weighted mean."""
    mask = series.notna() & weights.notna() & (weights > 0)
    valid_series = series[mask]
    valid_weights = weights[mask]
    return float((valid_series * valid_weights).sum() / valid_weights.sum())


def load_raw_walkability(path: str | Path) -> pd.DataFrame:
    """Load the EPA Smart Location Database CSV at block-group level."""
    return pd.read_csv(Path(path), encoding="utf-8-sig")


def aggregate_cbsa_walkability(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate block-group EPA metrics to CBSA level using the current schema."""
    df = df[[CBSA_COL, POP_COL, *MEAN_FEATURES]].copy()
    df[CBSA_COL] = df[CBSA_COL].astype("Int64").astype(str).str.strip()
    df = df[df[CBSA_COL] != "<NA>"].copy()

    for col in [POP_COL, *MEAN_FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mean_df = (
        df.groupby(CBSA_COL, as_index=False)[list(MEAN_FEATURES)]
        .mean()
        .rename(columns=MEAN_FEATURES)
    )

    walk_weighted = (
        df.groupby(CBSA_COL)
        .apply(lambda group: weighted_mean(group["NatWalkInd"], group[POP_COL]))
        .rename("nat_walk_index_pop_weighted")
        .reset_index()
    )

    agg_df = mean_df.merge(walk_weighted, on=CBSA_COL, how="left")
    agg_df = agg_df.rename(columns={CBSA_COL: "cbsa_code"})

    id_cols: Iterable[str] = ("cbsa_code",)
    other_cols = [col for col in agg_df.columns if col not in id_cols]
    return agg_df[list(id_cols) + other_cols]


def load_cbsa_walkability_dataset(
    path: str = "data/raw/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv",
) -> pd.DataFrame:
    """
    Convenience one‑liner: load EPA Smart Location CSV and aggregate to CBSA.

    Example
    -------
        from walkability_data_loader import load_cbsa_walkability_dataset
        walk_df = load_cbsa_walkability_dataset()
    """
    df = load_raw_walkability(path)
    return aggregate_cbsa_walkability(df)


if __name__ == "__main__":
    """
    Script entrypoint: build the CBSA-level walkability dataset and write it to CSV.

    Usage (from project root):
        python -m src.walkability_data_loader
    """
    output_path = Path("data/processed/Walkability_Data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    walk_df = load_cbsa_walkability_dataset()
    walk_df.to_csv(output_path, index=False)
    print(f"Wrote {len(walk_df)} rows to {output_path}")

