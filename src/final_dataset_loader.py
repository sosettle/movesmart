"""Build the final CBSA-level modeling dataset from fixed processed source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from models.cluster_model import build_cluster_mapping
from .standardize_scores import build_feature_and_composite_scores


DEFAULT_OUTPUT_PATH = Path("data/final/Final_Base_Dataset.csv")
DEFAULT_ENRICHED_OUTPUT_PATH = Path("data/final/Final_Enriched_Dataset.csv")
WALK_PATH = Path("data/processed/Walkability_Data.csv")
CENSUS_PATH = Path("data/processed/Census_Data.csv")
PLACES_PATH = Path("data/processed/Places_Data.csv")
CRIME_PATH = Path("data/processed/Crime_Data.csv")
WEATHER_PATH = Path("data/processed/Weather_Data.csv")
ID_COLUMNS = ["cbsa_code", "cbsa_name", "city", "state", "cbsa_type", "contains_imputed"]
WEATHER_DUPLICATE_COLUMNS = {"cbsa_name", "cbsa_type", "centroid_lat", "centroid_lon"}
PLACES_DUPLICATE_COLUMNS = {"cbsa_title"}
NON_IMPUTED_COLUMNS = {
    "cbsa_code",
    "cbsa_name",
    "city",
    "state",
    "cbsa_type",
    "contains_imputed",
    "station_id",
    "station_name",
}
ZERO_FILL_COLUMNS = {"annual_snowfall"}


def drop_puerto_rico_cbsas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Puerto Rico metro/micro areas (state == PR). Used before score scaling only."""
    if "state" not in df.columns:
        return df
    st = df["state"].astype(str).str.strip().str.upper()
    return df.loc[~st.eq("PR")].copy()


def load_processed_csv(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV and normalize the CBSA key."""
    df = pd.read_csv(Path(path), dtype={"cbsa_code": str})
    df["cbsa_code"] = df["cbsa_code"].astype(str).str.strip()
    return df


def validate_one_row_per_cbsa(df: pd.DataFrame, *, cbsa_col: str = "cbsa_code") -> None:
    """Raise if the dataframe is not one row per CBSA."""
    if cbsa_col not in df.columns:
        raise KeyError(f"Missing required column: {cbsa_col}")
    dupes = df[cbsa_col].duplicated(keep=False)
    if dupes.any():
        examples = df.loc[dupes, cbsa_col].astype(str).head(10).tolist()
        raise ValueError(f"Expected one row per {cbsa_col}; found duplicates (examples): {examples}")


def build_group_medians(df: pd.DataFrame, cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    """Return median values aligned to df rows for the requested grouping."""
    return df.groupby(group_cols, dropna=False)[cols].transform("median")


def hierarchical_median_impute(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fill numeric NaNs using progressively broader median groups.

    Returns the imputed dataframe and a row-level flag indicating whether any
    targeted column was filled during imputation.
    """
    out = df.copy()
    original_missing = out[cols].isna()

    for group_cols in (["cbsa_type", "state"], ["state"], ["cbsa_type"]):
        medians = build_group_medians(out, cols, group_cols)
        out[cols] = out[cols].fillna(medians)

    global_medians = out[cols].median(numeric_only=True)
    out[cols] = out[cols].fillna(global_medians)

    imputed_mask = original_missing & out[cols].notna()
    contains_imputed = imputed_mask.any(axis=1).astype(int)
    return out, contains_imputed


def reorder_columns(
    df: pd.DataFrame,
    census_cols: list[str],
    walk_cols: list[str],
    places_cols: list[str],
    crime_cols: list[str],
    weather_cols: list[str],
) -> pd.DataFrame:
    """Put identifiers first, then preserve source-specific feature order."""
    ordered = []
    for col in ID_COLUMNS:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    for source_cols in (census_cols, walk_cols, places_cols, crime_cols, weather_cols):
        for col in source_cols:
            if col in df.columns and col not in ordered:
                ordered.append(col)

    tail = [col for col in df.columns if col not in ordered]
    return df[ordered + tail]


def build_final_dataset(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Merge processed CBSA datasets (all U.S. CBSAs including PR), impute, and write the base dataset."""
    walk = load_processed_csv(WALK_PATH)
    census = load_processed_csv(CENSUS_PATH)
    places = load_processed_csv(PLACES_PATH)
    crime = load_processed_csv(CRIME_PATH)
    weather = load_processed_csv(WEATHER_PATH)

    census = census[[col for col in census.columns if not col.endswith("_prior")]].copy()
    places = places.drop(columns=[col for col in PLACES_DUPLICATE_COLUMNS if col in places.columns], errors="ignore")
    weather = weather.drop(columns=[col for col in WEATHER_DUPLICATE_COLUMNS if col in weather.columns], errors="ignore")

    census_cols = [col for col in census.columns if col != "cbsa_code"]
    walk_cols = [col for col in walk.columns if col != "cbsa_code"]
    places_cols = [col for col in places.columns if col != "cbsa_code"]
    crime_cols = [col for col in crime.columns if col != "cbsa_code"]
    weather_cols = [col for col in weather.columns if col != "cbsa_code"]

    final_df = census.merge(walk, on="cbsa_code", how="left", validate="one_to_one")
    final_df = final_df.merge(places, on="cbsa_code", how="left", validate="one_to_one")
    final_df = final_df.merge(crime, on="cbsa_code", how="left", validate="one_to_one")
    final_df = final_df.merge(weather, on="cbsa_code", how="left", validate="one_to_one")

    for col in ZERO_FILL_COLUMNS:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    numeric_cols = [
        col
        for col in final_df.columns
        if col not in NON_IMPUTED_COLUMNS and is_numeric_dtype(final_df[col]) and not is_bool_dtype(final_df[col])
    ]
    final_df, contains_imputed = hierarchical_median_impute(final_df, numeric_cols)
    final_df = final_df.copy()
    final_df["contains_imputed"] = contains_imputed
    final_df["violent_crime_per_population"] = final_df["violent_crime_count"] / final_df["TotalPopulation_B01003"]
    final_df["property_crime_per_population"] = final_df["property_crime_count"] / final_df["TotalPopulation_B01003"]
    crime_cols.extend(["violent_crime_per_population", "property_crime_per_population"])
    final_df = reorder_columns(final_df, census_cols, walk_cols, places_cols, crime_cols, weather_cols)
    validate_one_row_per_cbsa(final_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    return final_df


def build_final_enriched_dataset(
    *,
    base_output_path: str | Path = DEFAULT_OUTPUT_PATH,
    enriched_output_path: str | Path = DEFAULT_ENRICHED_OUTPUT_PATH,
    save_cluster_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Build the CBSA-level dataset and enrich it with feature scores, composite scores, and cluster labels.

    When ``save_cluster_artifacts`` is True (default), writes under ``data/clustering_output/``:
    ``Dataset_with_Cluster_Labels.csv``, ``CBSACODE_Cluster_Labels_Only.csv``,
    ``Clustering_Evaluation_Metrics.csv``, and ``Cluster_Label_Names.csv``.

    Output stays at one row per cbsa_code (mainland only in the enriched CSV; PR excluded after merge).
    """
    base = build_final_dataset(output_path=base_output_path)
    validate_one_row_per_cbsa(base)

    # 1) Cluster on full base (includes Puerto Rico CBSAs).
    cluster_mapping = build_cluster_mapping(base, save=save_cluster_artifacts)
    validate_one_row_per_cbsa(cluster_mapping)

    # 2) Feature scores + composites: exclude PR (scaling / dimension scores are mainland-only).
    base_for_scores = drop_puerto_rico_cbsas(base)
    scores = build_feature_and_composite_scores(base_for_scores)
    validate_one_row_per_cbsa(scores)

    # Avoid duplicate identifier columns on merge (keep base as the source of IDs).
    score_payload = scores.drop(columns=[c for c in scores.columns if c in ID_COLUMNS and c != "cbsa_code"])

    out = base.merge(score_payload, on="cbsa_code", how="left", validate="one_to_one")
    out = out.merge(cluster_mapping, on="cbsa_code", how="left", validate="one_to_one")
    validate_one_row_per_cbsa(out)

    out = drop_puerto_rico_cbsas(out)
    validate_one_row_per_cbsa(out)

    enriched_output_path = Path(enriched_output_path)
    enriched_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(enriched_output_path, index=False)

    return out


if __name__ == "__main__":
    df = build_final_enriched_dataset(save_cluster_artifacts=True)
    print(f"Wrote {len(df)} rows and {len(df.columns)} columns to {DEFAULT_ENRICHED_OUTPUT_PATH}")
    print("Cluster artifacts: data/clustering_output/")
    print("Run from project root with: python -m src.final_dataset_loader")
