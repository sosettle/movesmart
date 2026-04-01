from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


# Enriched CSV keeps 0–1 scores; all recommendation outputs use 0–5 (same range as sliders).
SCORE_MAX = 5.0
DISPLAY_DECIMALS = 2

DIMENSION_SCORE_COLS = [
    "affordability_score",
    "job_growth_score",
    "safety_score",
    "education_score",
    "health_score",
    "walkability_score",
    "diversity_score",
    "urban_score",
    "weather_warmth_score",
    "weather_mildness_score",
]



def build_dimension_scores(feature_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Build composite 0-1 dimension scores from 0-1 feature scores."""
    df = feature_scores_df.copy()

    df["affordability_score"] = (
        df["MedianHomeValue_B25077_feature_score"] * (1 / 3)
        + df["MedianGrossRent_B25064_feature_score"] * (1 / 3)
        + df["MedianHouseholdIncome_B19013_feature_score"] * (1 / 3)
    )

    df["job_growth_score"] = (
        df["job_growth_feature_score"] * 0.50
        + df["population_growth_feature_score"] * 0.20
        + df["unemployment_rate_feature_score"] * 0.30
    )

    df["safety_score"] = (
        df["violent_crime_per_population_feature_score"] * 0.50
        + df["property_crime_per_population_feature_score"] * 0.50
    )

    df["education_score"] = df["bachelors_or_higher_share_feature_score"]

    df["health_score"] = (
        df["obesity_share_feature_score"] * 0.10
        + df["physical_inactivity_share_feature_score"] * 0.10
        + df["depression_share_feature_score"] * 0.10
        + df["current_asthma_share_feature_score"] * 0.08
        + df["diabetes_share_feature_score"] * 0.12
        + df["stroke_share_feature_score"] * 0.10
        + df["coronary_heart_disease_share_feature_score"] * 0.12
        + df["arthritis_share_feature_score"] * 0.08
        + df["any_disability_share_feature_score"] * 0.10
        + df["current_smoking_share_feature_score"] * 0.10
    )

    df["walkability_score"] = df["nat_walk_index_pop_weighted_feature_score"]
    df["diversity_score"] = df["diversity_index_feature_score"]

    df["urban_score"] = (
        df["population_density_per_sq_mile_feature_score"] * 0.50
        + df["intersection_density_mean_feature_score"] * 0.25
        + df["activity_density_mean_feature_score"] * 0.25
    )

    df["weather_warmth_score"] = df["avg_annual_temp_feature_score"]

    df["weather_mildness_score"] = (
        df["_temp_distance_feature_score"] * 0.40
        + df["temp_seasonality_feature_score"] * 0.30
        + df["annual_snowfall_feature_score"] * 0.20
        + df["annual_precipitation_feature_score"] * 0.10
    )

    df[DIMENSION_SCORE_COLS] = df[DIMENSION_SCORE_COLS].clip(0.0, 1.0)
    return df


def _score_column_names_for_0_5(df: pd.DataFrame) -> list[str]:
    """Dimension, recommendation, and per-feature score columns (stored 0–1 in CSV)."""
    names: list[str] = []
    for col in df.columns:
        if col in DIMENSION_SCORE_COLS or col == "recommendation_score":
            names.append(col)
        elif col.endswith("_feature_score"):
            names.append(col)
    return names


def scale_stored_scores_to_0_5(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stored 0–1 scores to 0–5 (linear; ranking unchanged)."""
    out = df.copy()
    hi = SCORE_MAX
    for col in _score_column_names_for_0_5(out):
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if s.notna().any():
            out[col] = (s * hi).clip(0.0, hi).round(DISPLAY_DECIMALS)
    return out


def get_preference_weights(user_inputs: Mapping[str, float]) -> dict[str, float]:
    """Convert user preference ratings into normalized squared weights."""
    valid_inputs = {
        col: float(rating)
        for col, rating in user_inputs.items()
        if rating is not None and float(rating) >= 0
    }

    if not valid_inputs:
        raise ValueError("No valid user inputs were provided.")

    squared_total = sum(rating ** 2 for rating in valid_inputs.values())
    if squared_total == 0:
        raise ValueError("At least one user rating must be greater than 0.")

    return {
        col: (rating ** 2) / squared_total
        for col, rating in valid_inputs.items()
    }



def apply_affordability_filter(
    df: pd.DataFrame,
    user_income: float | None = None,
    housing_mode: str = "either",
) -> pd.DataFrame:
    """Filter cities based on a simple rent / home value affordability rule."""
    valid_housing_modes = {"rent", "buy", "either"}
    if housing_mode not in valid_housing_modes:
        raise ValueError(f"housing_mode must be one of {valid_housing_modes}")

    ranked = df.copy()

    if user_income is None:
        return ranked

    max_affordable_rent = (user_income / 12.0) * 0.30
    max_affordable_home_value = user_income * 3.5

    if housing_mode == "rent":
        ranked = ranked[ranked["MedianGrossRent_B25064"] <= max_affordable_rent]
    elif housing_mode == "buy":
        ranked = ranked[ranked["MedianHomeValue_B25077"] <= max_affordable_home_value]
    else:  # either
        ranked = ranked[
            (ranked["MedianGrossRent_B25064"] <= max_affordable_rent)
            | (ranked["MedianHomeValue_B25077"] <= max_affordable_home_value)
        ]

    return ranked



def score_cities(
    df: pd.DataFrame,
    user_inputs: Mapping[str, float],
    score_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply weighted preference scoring to the supplied city dataframe."""
    ranked = df.copy()
    score_cols = score_cols or DIMENSION_SCORE_COLS

    valid_inputs = {
        col: rating
        for col, rating in user_inputs.items()
        if col in ranked.columns and col in score_cols and rating is not None
    }

    if not valid_inputs:
        raise ValueError("No valid scoring columns from user_inputs were found in the dataframe.")

    weights = get_preference_weights(valid_inputs)

    ranked["recommendation_score"] = 0.0
    for col, weight in weights.items():
        ranked["recommendation_score"] += ranked[col] * weight

    ranked["recommendation_score"] = ranked["recommendation_score"].clip(0.0, 1.0)
    return ranked


def add_text_to_cbsa(
    df: pd.DataFrame,
    wiki_path: str | Path = "data/processed/cbsa_wiki_wikivoyage_summaries_df.csv",
) -> pd.DataFrame:
    """Attach wiki/tagline/summary text by cbsa_code (optional file)."""
    out = df.copy()
    if "cbsa_code" not in out.columns:
        return out
    out["cbsa_code"] = out["cbsa_code"].astype(str).str.strip()
    path = Path(wiki_path)
    text_cols = ("city_wiki_wikivoyage_text", "tagline", "summary")
    if not path.is_file():
        for col in text_cols:
            if col not in out.columns:
                out[col] = ""
        return out
    wiki = pd.read_csv(path, dtype={"cbsa_code": str})
    wiki["cbsa_code"] = wiki["cbsa_code"].astype(str).str.strip()
    keep = ["cbsa_code"] + [c for c in text_cols if c in wiki.columns]
    wiki = wiki[keep].drop_duplicates(subset=["cbsa_code"], keep="last")
    out = out.drop(columns=[c for c in text_cols if c in out.columns], errors="ignore")
    out = out.merge(wiki, on="cbsa_code", how="left")
    for col in text_cols:
        if col not in out.columns:
            out[col] = ""
        else:
            out[col] = out[col].fillna("").astype(str)
    return out


def recommend_cities(
    df: pd.DataFrame,
    user_inputs: Mapping[str, float],
    user_income: float | None = None,
    housing_mode: str = "either",
    top_n: int = 30,
    score_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return the top recommended cities; all score columns are on a 0–5 scale."""
    ranked = apply_affordability_filter(
        df=df,
        user_income=user_income,
        housing_mode=housing_mode,
    )

    if ranked.empty:
        return ranked.copy()

    ranked = score_cities(
        df=ranked,
        user_inputs=user_inputs,
        score_cols=score_cols,
    )

    ranked = ranked.sort_values("recommendation_score", ascending=False)
    out = ranked.head(top_n).copy()
    return scale_stored_scores_to_0_5(out)
