from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .recommender import DIMENSION_SCORE_COLS, build_dimension_scores

DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / "Final_Merged_Imputed_Dataset.csv"
OUTPUT_CSV = DATA_DIR / "cbsa_feature_scores_0_1.csv"

# Signed series (can be negative); arcsinh compresses tails without requiring positivity.
ARCSINH_COLS = (
    "job_growth",
    "population_growth",
)

# Nonnegative, strongly right-skewed on the current CBSA panel.
LOG_TRANSFORM_COLS = (
    "MedianGrossRent_B25064",
    "MedianHomeValue_B25077",
    "activity_density_mean",
    "age_22_34_share",
    "age_65_plus_share",
    "annual_snowfall",
    "blue_collar_share",
    "industry_ag_forestry_mining_share",
    "industry_arts_rec_accommodation_food_share",
    "industry_finance_real_estate_share",
    "industry_information_share",
    "industry_prof_sci_mgmt_admin_share",
    "industry_public_admin_share",
    "intersection_density_mean",
    "population_density_per_sq_mile",
    "poverty_rate",
    "property_crime_per_population",
    "sales_share",
    "unemployment_rate",
    "violent_crime_per_population",
)

# Lower raw value = better final score, so invert after scaling.
INVERSE_SCORE_COLS = {
    "MedianHomeValue_B25077",
    "MedianGrossRent_B25064",
    "unemployment_rate",
    "violent_crime_per_population",
    "property_crime_per_population",
    "obesity_share",
    "physical_inactivity_share",
    "depression_share",
    "current_asthma_share",
    "diabetes_share",
    "stroke_share",
    "coronary_heart_disease_share",
    "arthritis_share",
    "any_disability_share",
    "current_smoking_share",
    "_temp_distance",
    "temp_seasonality",
    "annual_snowfall",
    "annual_precipitation",
}

# Columns used to build the 0-1 feature scores.
SCORE_FEATURE_COLS = [
    "MedianHomeValue_B25077",
    "MedianGrossRent_B25064",
    "MedianHouseholdIncome_B19013",
    "job_growth",
    "population_growth",
    "unemployment_rate",
    "violent_crime_per_population",
    "property_crime_per_population",
    "bachelors_or_higher_share",
    "obesity_share",
    "physical_inactivity_share",
    "depression_share",
    "current_asthma_share",
    "diabetes_share",
    "stroke_share",
    "coronary_heart_disease_share",
    "arthritis_share",
    "any_disability_share",
    "current_smoking_share",
    "nat_walk_index_pop_weighted",
    "diversity_index",
    "population_density_per_sq_mile",
    "intersection_density_mean",
    "activity_density_mean",
    "avg_annual_temp",
    "temp_seasonality",
    "annual_snowfall",
    "annual_precipitation",
]

ID_COLS = [
    "cbsa_code",
    "cbsa_name",
    "city",
    "state",
    "cbsa_type",
]


def apply_feature_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ARCSINH_COLS:
        if col in out.columns:
            out[col] = np.arcsinh(out[col])

    for col in LOG_TRANSFORM_COLS:
        if col in out.columns:
            out[col] = np.log1p(out[col])

    return out



def winsorize_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            continue
        lo = out[col].quantile(lower)
        hi = out[col].quantile(upper)
        out[col] = out[col].clip(lo, hi)

    return out



def minmax_score(series: pd.Series, invert: bool = False) -> pd.Series:
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        scaled = pd.Series(0.5, index=s.index, dtype=float)
    else:
        scaled = (s - min_val) / (max_val - min_val)

    if invert:
        scaled = 1.0 - scaled

    return scaled.clip(0.0, 1.0)



def build_feature_scores(
    raw_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Create 0-1 feature scores for each CBSA.

    Returns a dataframe containing the ID columns plus <feature>_feature_score columns.
    Higher is always better.
    """
    feature_cols = feature_cols or SCORE_FEATURE_COLS

    missing = [col for col in feature_cols if col not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    transformed_df = raw_df.copy()
    transformed_df["_temp_distance"] = (transformed_df["avg_annual_temp"] - 65).abs()

    transformed_df = apply_feature_transforms(transformed_df)

    scoring_cols = feature_cols + ["_temp_distance"]
    score_input_df = transformed_df[scoring_cols].copy()

    if winsorize:
        score_input_df = winsorize_columns(score_input_df, columns=scoring_cols)

    out = raw_df[[col for col in ID_COLS if col in raw_df.columns]].copy()

    for col in scoring_cols:
        out[f"{col}_feature_score"] = minmax_score(
            score_input_df[col],
            invert=col in INVERSE_SCORE_COLS,
        )

    return out


def build_composite_scores(feature_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add composite 0-1 dimension score columns onto a feature-score dataframe."""
    return build_dimension_scores(feature_scores_df)


def build_feature_and_composite_scores(
    raw_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    winsorize: bool = True,
) -> pd.DataFrame:
    """
    Convenience: build <feature>_feature_score columns + composite dimension scores.

    Returns ID columns + feature scores + composite scores.
    """
    feat = build_feature_scores(raw_df, feature_cols=feature_cols, winsorize=winsorize)
    out = build_composite_scores(feat)

    missing = [c for c in DIMENSION_SCORE_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"Composite score columns were not created: {missing}")

    return out



def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    feature_scores_df = build_feature_and_composite_scores(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feature_scores_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved standardized 0-1 feature scores to: {OUTPUT_CSV}")
    print(feature_scores_df.head())


if __name__ == "__main__":
    main()
