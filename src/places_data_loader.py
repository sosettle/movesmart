"""
CBSA-level health prevalence features from CDC PLACES tract-level data.

Pipeline:
- Load CDC PLACES tract rows from the GIS-friendly tract CSV
- Parse tract point geometry from geolocation
- Load CBSA polygons from the Census shapefile with GeoPandas
- Spatially join PLACES tract points to CBSA polygons
- Keep a curated set of crude-prevalence PLACES measures
- Aggregate tract prevalence values to the CBSA level using population weights
"""

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


MEASURE_ID_TO_FEATURE = {
    "OBESITY": "obesity_share",
    "LPA": "physical_inactivity_share",
    "DEPRESSION": "depression_share",
    "CASTHMA": "current_asthma_share",
    "DIABETES": "diabetes_share",
    "STROKE": "stroke_share",
    "CHD": "coronary_heart_disease_share",
    "ARTHRITIS": "arthritis_share",
    "DISABILITY": "any_disability_share",
    "CSMOKING": "current_smoking_share",
    "BINGE": "binge_drinking_share",
}

FEATURE_COLUMNS = list(MEASURE_ID_TO_FEATURE.values())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase snake_case."""
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(col).strip().lower())).strip("_")
        for col in df.columns
    ]
    return df


def parse_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Parse longitude and latitude from the PLACES geolocation column."""
    coords = df["geolocation"].astype(str).str.extract(r"POINT \(([-0-9.]+) ([-0-9.]+)\)")
    df = df.copy()
    df["longitude"] = pd.to_numeric(coords[0], errors="coerce")
    df["latitude"] = pd.to_numeric(coords[1], errors="coerce")
    return df


def load_cbsa_shapes(shapefile_path: str | Path) -> gpd.GeoDataFrame:
    """Load CBSA polygons and keep join-ready columns."""
    shapefile_path = Path(shapefile_path)
    cbsa = gpd.read_file(shapefile_path)[["CBSAFP", "NAME", "geometry"]].copy()
    cbsa = cbsa.rename(columns={"CBSAFP": "cbsa_code", "NAME": "cbsa_title"})
    return cbsa.to_crs("EPSG:4326")


def filter_places_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Keep requested tract-level PLACES measures and numeric fields."""
    df = df.copy()

    for col in ["totalpopulation", "totalpop18plus"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    out = df[["tractfips", "totalpopulation", "totalpop18plus", "longitude", "latitude"]].copy()
    for measure_id, feature_name in MEASURE_ID_TO_FEATURE.items():
        source_col = f"{measure_id.lower()}_crudeprev"
        out[feature_name] = pd.to_numeric(
            df[source_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    out["population_weight"] = out["totalpop18plus"].where(
        out["totalpop18plus"].notna(),
        out["totalpopulation"],
    )
    return out


def to_places_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert PLACES tract rows to a GeoDataFrame in EPSG:4326."""
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")


def prepare_tract_level_data(df: gpd.GeoDataFrame) -> pd.DataFrame:
    """Keep one row per tract after spatial join."""
    ordered_cols = ["cbsa_code", "cbsa_title", "tractfips", "population_weight"] + FEATURE_COLUMNS
    return df[ordered_cols].drop_duplicates(subset=["cbsa_code", "cbsa_title", "tractfips"])


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    """Return a NaN-safe weighted average for one feature."""
    values = pd.to_numeric(series, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = (~values.isna()) & (~weights.isna()) & (weights > 0)
    if not mask.any():
        return np.nan
    values = values[mask]
    weights = weights[mask]
    return float((values * weights).sum() / weights.sum())


def aggregate_places_to_cbsa(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Aggregate tract-level PLACES features to one weighted row per CBSA."""
    rows = []
    for (cbsa_code, cbsa_title), group in df.groupby(["cbsa_code", "cbsa_title"], dropna=False):
        out = {
            "cbsa_code": cbsa_code,
            "cbsa_title": cbsa_title,
            "n_tracts": int(group["tractfips"].nunique()),
            "total_population_weight": pd.to_numeric(group["population_weight"], errors="coerce").sum(min_count=1),
        }
        for col in feature_cols:
            out[col] = weighted_average(group[col], group["population_weight"])
        rows.append(out)

    final_cols = ["cbsa_code", "cbsa_title", "n_tracts", "total_population_weight"] + feature_cols
    if not rows:
        return pd.DataFrame(columns=final_cols)
    return pd.DataFrame(rows)[final_cols].sort_values(["cbsa_code", "cbsa_title"]).reset_index(drop=True)


def load_places_cbsa_data(
    places_path: str | Path,
    shapefile_path: str | Path,
) -> pd.DataFrame:
    """Load CDC PLACES tract rows and aggregate curated health measures to the CBSA level."""
    places_path = Path(places_path)
    shapefile_path = Path(shapefile_path)

    places = pd.read_csv(places_path, low_memory=False)
    places = standardize_columns(places)
    print(f"Loaded {len(places)} PLACES rows.")
    places["tractfips"] = places["tractfips"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    places = parse_geolocation(places)
    places = places.dropna(subset=["longitude", "latitude"]).copy()
    places = filter_places_measures(places)

    places_gdf = to_places_geodataframe(places)
    cbsa_gdf = load_cbsa_shapes(shapefile_path)
    matched = gpd.sjoin(places_gdf, cbsa_gdf, how="inner", predicate="within").drop(columns=["index_right"])
    print(f"Matched {len(matched)} PLACES rows to a CBSA.")

    tract_level = prepare_tract_level_data(matched)
    cbsa_df = aggregate_places_to_cbsa(tract_level, FEATURE_COLUMNS)
    print(f"Produced {len(cbsa_df)} unique CBSAs.")
    return cbsa_df


if __name__ == "__main__":
    """
    Script entrypoint: build the CBSA-level PLACES dataset and write it to CSV.

    Usage (from project root):
        python -m src.places_data_loader
    """
    df = load_places_cbsa_data(
        places_path="data/raw/PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260314.csv",
        shapefile_path="data/raw/shapefiles/tl_2023_us_cbsa.shp",
    )
    output_path = Path("data/processed/Places_Data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
