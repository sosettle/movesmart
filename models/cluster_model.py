"""
CBSA clustering: log1p on selected columns → RobustScaler → 5–95% clip → PCA (90% variance)
→ first 3 PCs → KMeans (k=6 and k=20). Writes CSVs under data/clustering_output/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
N_INIT = 10

DATA_DIR = Path("data")
CLUSTER_OUTPUT_DIR = DATA_DIR / "clustering_output"
INPUT_CSV = DATA_DIR / "Final_Merged_Imputed_Dataset.csv"
OUT_FULL = CLUSTER_OUTPUT_DIR / "Dataset_with_Cluster_Labels.csv"
OUT_MAPPING = CLUSTER_OUTPUT_DIR / "CBSACODE_Cluster_Labels_Only.csv"
OUT_METRICS = CLUSTER_OUTPUT_DIR / "Clustering_Evaluation_Metrics.csv"

DROP_COLS = [
    "cbsa_code",
    "cbsa_name",
    "city",
    "state",
    "cbsa_type",
    "contains_imputed",
    "TotalPopulation_B01003",
    "TotalPovertyUniverse_B17001",
    "BelowPoverty_B17001",
    "IndustryTotalEmployed_C24050",
    "IndustryAgForestryMining_C24050",
    "IndustryConstruction_C24050",
    "IndustryManufacturing_C24050",
    "IndustryWholesaleTrade_C24050",
    "IndustryRetailTrade_C24050",
    "IndustryTransportUtilities_C24050",
    "IndustryInformation_C24050",
    "IndustryFinanceRealEstate_C24050",
    "IndustryProfSciMgmtAdminWaste_C24050",
    "IndustryEducationHealthCare_C24050",
    "IndustryArtsRecAccommodationFood_C24050",
    "IndustryOtherServices_C24050",
    "IndustryPublicAdmin_C24050",
    "TotalEmployed_C24010",
    "ManagementBusiness_C24010",
    "ScienceEngineering_C24010",
    "ServiceOccupations_C24010",
    "SalesOffice_C24010",
    "Construction_C24010",
    "ProductionTransportation_C24010",
    "TotalPopulation25Plus_B15003",
    "BachelorsDegree_B15003",
    "MastersDegree_B15003",
    "ProfessionalDegree_B15003",
    "DoctorateDegree_B15003",
    "TotalRace_B02001",
    "WhiteAlone_B02001",
    "BlackAlone_B02001",
    "AmericanIndianAlaskaNative_B02001",
    "AsianAlone_B02001",
    "NativeHawaiianPacificIslander_B02001",
    "SomeOtherRace_B02001",
    "TwoOrMoreRaces_B02001",
    "LaborForce_B23025",
    "Unemployed_B23025",
    "age_22_34_total",
    "age_65_plus_total",
    "land_area_m2",
    "water_area_m2",
    "land_area_sqmi",
    "water_area_sqmi",
    "centroid_lat",
    "centroid_lon",
    "nat_walk_index_mean",
    "population_density_mean",
    "n_tracts",
    "total_population_weight",
    "violent_crime_count",
    "property_crime_count",
    "total_crime_count",
    "station_id",
    "station_name",
    "station_distance_km",
    "snow_binary",
    "n_months_available",
    "has_complete_year",
]

LOG_TRANSFORM_COLS = [
    "activity_density_mean",
    "intersection_density_mean",
    "population_density_per_sq_mile",
    "annual_snowfall",
    "violent_crime_per_population",
    "property_crime_per_population",
]


def _wcss(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        centroid = pts.mean(axis=0)
        total += float(((pts - centroid) ** 2).sum())
    return total


def assign_clusters(df: pd.DataFrame, *, save: bool = True):
    """Returns (full_df_with_labels, cbsa_mapping_df, metrics_df). Optionally writes CSVs."""
    feat = df.drop(columns=DROP_COLS).copy()
    feat[LOG_TRANSFORM_COLS] = np.log1p(feat[LOG_TRANSFORM_COLS])

    X = RobustScaler().fit_transform(feat)
    X = pd.DataFrame(X, columns=feat.columns, index=feat.index)
    for col in X.columns:
        lo, hi = X[col].quantile(0.05), X[col].quantile(0.95)
        X[col] = X[col].clip(lo, hi)

    X_pca = PCA(n_components=0.9, random_state=RANDOM_STATE).fit_transform(X)
    X_reduced = X_pca[:, :3]

    y6 = KMeans(6, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced)
    y20 = KMeans(20, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced)

    full = df.copy()
    full["cluster_k6"] = y6
    full["cluster_k20"] = y20
    mapping = full[["cbsa_code", "cluster_k6", "cluster_k20"]].copy()

    metrics = pd.DataFrame(
        [
            {
                "model": "cluster_k6",
                "n_clusters": 6,
                "cluster_min_size": int(pd.Series(y6).value_counts().min()),
                "cluster_max_size": int(pd.Series(y6).value_counts().max()),
                "silhouette_score": float(silhouette_score(X_reduced, y6)),
                "wcss": _wcss(X_reduced, y6),
            },
            {
                "model": "cluster_k20",
                "n_clusters": 20,
                "cluster_min_size": int(pd.Series(y20).value_counts().min()),
                "cluster_max_size": int(pd.Series(y20).value_counts().max()),
                "silhouette_score": float(silhouette_score(X_reduced, y20)),
                "wcss": _wcss(X_reduced, y20),
            },
        ]
    )

    if save:
        CLUSTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        full.to_csv(OUT_FULL, index=False)
        mapping.to_csv(OUT_MAPPING, index=False)
        metrics.to_csv(OUT_METRICS, index=False)

    return full, mapping, metrics


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    _, _, metrics = assign_clusters(df)
    print(metrics.to_string(index=False))
    print(f"\nWrote:\n  {OUT_FULL.resolve()}\n  {OUT_MAPPING.resolve()}\n  {OUT_METRICS.resolve()}")


if __name__ == "__main__":
    main()
