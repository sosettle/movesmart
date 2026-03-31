"""
CBSA clustering: variance-stabilizing transforms → RobustScaler → selective 5–95% winsorize
→ PCA (90% variance) → first 3 PCs → KMeans (k=5 and k=22). Writes CSVs under data/clustering_output/.

Transforms (from inspecting skew on Final_Merged_Imputed_Dataset):
- ``log1p`` on strongly right-skewed nonnegative features (raw skew ≳ 1.25), plus the original
  density/crime/snow fields.
- ``arcsinh`` on signed growth rates (negative values possible, heavy tails).
- Winsorize *after* scaling only for columns that still show long tails: |skew| > 0.5,
  excess kurtosis > 0.5, or >2% of values beyond |z|=4 in robust-scaled space.
  Well-behaved columns (e.g. many bounded shares after scaling) are left unclipped.
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

# After RobustScaler: clip 5–95% only if any of these hold (tuned on current CBSA panel).
CLIP_SKEW_ABS = 0.5
CLIP_EXCESS_KURTOSIS = 0.5
CLIP_PCT_BEYOND_4 = 2.0  # percent of rows with |x| > 4

DATA_DIR = Path("data")
CLUSTER_OUTPUT_DIR = DATA_DIR / "clustering_output"
INPUT_CSV = DATA_DIR / "Final_Merged_Imputed_Dataset.csv"
OUT_FULL = CLUSTER_OUTPUT_DIR / "Dataset_with_Cluster_Labels.csv"
OUT_MAPPING = CLUSTER_OUTPUT_DIR / "CBSACODE_Cluster_Labels_Only.csv"
OUT_METRICS = CLUSTER_OUTPUT_DIR / "Clustering_Evaluation_Metrics.csv"
OUT_LABELS = CLUSTER_OUTPUT_DIR / "Cluster_Label_Names.csv"

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

# Signed series (can be negative); arcsinh compresses tails without requiring positivity.
ARCSINH_COLS = (
    "job_growth",
    "population_growth",
)

# Nonnegative, strongly right-skewed on the training panel (raw skew ≳ 1.25) + original density/crime/snow.
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

NAME_FEATURES = (
    "MedianHouseholdIncome_B19013",
    "MedianHomeValue_B25077",
    "MedianGrossRent_B25064",
    "bachelors_or_higher_share",
    "unemployment_rate",
    "poverty_rate",
    "violent_crime_per_population",
    "property_crime_per_population",
    "nat_walk_index_pop_weighted",
    "population_density_per_sq_mile",
    "intersection_density_mean",
    "activity_density_mean",
    "diversity_index",
    "avg_annual_temp",
    "temp_seasonality",
    "annual_precipitation",
    "annual_snowfall",
    "job_growth",
    "population_growth",
    "median_home_value_growth",
    "median_rent_growth",
)


def apply_variance_stabilizing_transforms(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    for c in ARCSINH_COLS:
        out[c] = np.arcsinh(out[c])
    for c in LOG_TRANSFORM_COLS:
        out[c] = np.log1p(out[c])
    return out


def selective_winsorize_robust(X: pd.DataFrame) -> pd.DataFrame:
    """5–95% clip only on scaled columns that still look long-tailed."""
    X = X.copy()
    for col in X.columns:
        s = X[col].skew()
        k_ex = X[col].kurtosis()
        tail_pct = float((X[col].abs() > 4).mean() * 100)
        if (
            abs(s) > CLIP_SKEW_ABS
            or k_ex > CLIP_EXCESS_KURTOSIS
            or tail_pct > CLIP_PCT_BEYOND_4
        ):
            lo, hi = X[col].quantile(0.05), X[col].quantile(0.95)
            X[col] = X[col].clip(lo, hi)
    return X


def wcss(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        centroid = pts.mean(axis=0)
        total += float(((pts - centroid) ** 2).sum())
    return total


def cluster_names(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Create conversational names per cluster label based on a small set of interpretable features.
    Names are deterministic given the input dataset.
    """
    cols = [c for c in NAME_FEATURES if c in df.columns]
    if not cols:
        out = pd.DataFrame({cluster_col: sorted(df[cluster_col].unique())})
        out[f"{cluster_col}_name"] = out[cluster_col].astype(str)
        return out

    # Use medians to avoid outlier sensitivity.
    cluster_meds = df.groupby(cluster_col, dropna=False)[cols].median(numeric_only=True)
    global_med = df[cols].median(numeric_only=True)
    global_iqr = (df[cols].quantile(0.75) - df[cols].quantile(0.25)).replace(0, np.nan)
    z = (cluster_meds - global_med) / global_iqr
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_to_phrase = {
        "MedianHouseholdIncome_B19013": ("Higher-income", "Lower-income"),
        "MedianHomeValue_B25077": ("Expensive housing", "Affordable housing"),
        "MedianGrossRent_B25064": ("High-rent", "Low-rent"),
        "bachelors_or_higher_share": ("More educated", "Less educated"),
        "nat_walk_index_pop_weighted": ("More walkable", "Car-dependent"),
        "population_density_per_sq_mile": ("Denser", "More spread out"),
        "diversity_index": ("More diverse", "Less diverse"),
        "job_growth": ("Fast job growth", "Weak job growth"),
        "population_growth": ("Fast population growth", "Weak population growth"),
        "avg_annual_temp": ("Warmer", "Cooler"),
        "annual_snowfall": ("Snowier", "Less snowy"),
        "unemployment_rate": ("Lower unemployment", "Higher unemployment"),
        "poverty_rate": ("Lower poverty", "Higher poverty"),
        "violent_crime_per_population": ("Lower violent crime", "Higher violent crime"),
        "property_crime_per_population": ("Lower property crime", "Higher property crime"),
    }

    def pick(label: int) -> str:
        row = z.loc[label].copy()

        # Build a ranked list of (abs_z, phrase) across interpretable features.
        ranked: list[tuple[float, str]] = []
        for f, (pos, neg) in feature_to_phrase.items():
            if f not in row.index:
                continue
            val = float(row[f])
            if abs(val) < 0.35:
                continue
            ranked.append((abs(val), pos if val > 0 else neg))

        ranked.sort(key=lambda t: t[0], reverse=True)
        parts = []
        for _, phr in ranked:
            if phr not in parts:
                parts.append(phr)
            if len(parts) == 2:
                break

        # Deterministic fallback: pick the single strongest feature even if mild.
        if not parts:
            strongest = row.abs().sort_values(ascending=False)
            for f in strongest.index.tolist():
                if f in feature_to_phrase:
                    val = float(row[f])
                    pos, neg = feature_to_phrase[f]
                    parts = [pos if val >= 0 else neg]
                    break
        if not parts:
            parts = ["Balanced"]

        return " / ".join(parts)

    out = pd.DataFrame({cluster_col: cluster_meds.index.astype(int)})
    out[f"{cluster_col}_name"] = [pick(int(i)) for i in out[cluster_col].tolist()]
    return out.sort_values(cluster_col).reset_index(drop=True)


def build_cluster_label_report(
    df: pd.DataFrame,
    label_col: str,
    *,
    top_traits: int = 5,
) -> pd.DataFrame:
    """
    Build a human-facing labeling report per label (traits, candidate names, final name, description, trade-offs).
    """
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.drop(columns=[label_col], errors="ignore")

    overall_mean = num.mean(numeric_only=True)
    overall_med = num.median(numeric_only=True)
    overall_iqr = (num.quantile(0.75) - num.quantile(0.25)).replace(0, np.nan)

    feature_to_phrase = {
        "MedianHouseholdIncome_B19013": ("Higher-income", "Lower-income"),
        "MedianHomeValue_B25077": ("Expensive housing", "Affordable housing"),
        "MedianGrossRent_B25064": ("High-rent", "Low-rent"),
        "bachelors_or_higher_share": ("More educated", "Less educated"),
        "nat_walk_index_pop_weighted": ("More walkable", "Car-dependent"),
        "population_density_per_sq_mile": ("Denser", "More spread out"),
        "diversity_index": ("More diverse", "Less diverse"),
        "job_growth": ("Fast job growth", "Weak job growth"),
        "population_growth": ("Fast population growth", "Weak population growth"),
        "avg_annual_temp": ("Warmer", "Cooler"),
        "annual_snowfall": ("Snowier", "Less snowy"),
        "unemployment_rate": ("Lower unemployment", "Higher unemployment"),
        "poverty_rate": ("Lower poverty", "Higher poverty"),
        "violent_crime_per_population": ("Lower violent crime", "Higher violent crime"),
        "property_crime_per_population": ("Lower property crime", "Higher property crime"),
    }

    def candidate_names(traits: list[str]) -> list[str]:
        base = [t for t in traits if t]
        if not base:
            return ["Balanced Everyday Metros", "All-Around Cities", "Generalist Metros"]
        a = base[0]
        b = base[1] if len(base) > 1 else ""

        n1 = f"{a} Hubs"
        n2 = f"{a} Lifestyle Metros" if not b else f"{a} + {b} Metros"
        n3 = f"{a} Sweet-Spot Cities" if not b else f"{a} / {b} Cities"
        return [n1, n2, n3]

    # Hardcoded, user-facing labels (from prior naming pass).
    # If a label is not found here, we fall back to the deterministic auto-name.
    HARDCODED = {
        "cluster": {
            0: {
                "final_name": "Affluent Retirement Growth Metros",
                "candidates": [
                    "Affluent Retirement Growth Metros",
                    "High-Cost Lifestyle Hubs",
                    "Upscale Sun-Seeker Enclaves",
                ],
                "description": "Older-leaning metros with above-average home values and rents, often with a lifestyle/leisure tilt and solid recent growth.",
                "tradeoffs": "High cost of living; less starter-city friendly.",
            },
            1: {
                "final_name": "Snowbelt Manufacturing Hubs",
                "candidates": [
                    "Snowbelt Factory Towns",
                    "Cold-Weather Manufacturing Metros",
                    "Seasonal Rust-Belt Hubs",
                ],
                "description": "Cold, highly seasonal metros with heavier manufacturing presence—classic snowbelt profiles.",
                "tradeoffs": "Tough winters; amenities can be more limited than warmer/coastal metros.",
            },
            2: {
                "final_name": "Pricey Walkable City Centers",
                "candidates": [
                    "Pricey Walkable City Centers",
                    "Urban Amenity Hotspots",
                    "High-Rent Walkable Hubs",
                ],
                "description": "Dense, high-amenity, walkable metros with elevated rents and home values (often in snowier regions).",
                "tradeoffs": "Higher costs; less space per dollar.",
            },
            3: {
                "final_name": "Hardship & Health Challenge Metros",
                "candidates": [
                    "Struggling Health-Burdened Metros",
                    "Hardship & Health Challenge Cities",
                    "Underserved Recovery Regions",
                ],
                "description": "Metros that stand out for higher poverty and multiple adverse health indicators—places where affordability may exist but outcomes lag.",
                "tradeoffs": "Lower cost potential, but weaker public-health profile and often fewer high-opportunity signals.",
            },
            4: {
                "final_name": "Young Frontier Metros",
                "candidates": [
                    "Young Frontier Growth Metros",
                    "Boomtown Starter Cities",
                    "Younger Work-First Metros",
                ],
                "description": "Younger-leaning metros with a more work-first economic footprint (including resource-linked areas) and somewhat higher safety volatility.",
                "tradeoffs": "Energy and youth skew, but less stability/amenity polish and potentially higher crime risk.",
            },
        },
        "sub_cluster": {
            0: {
                "final_name": "Snowbelt Prosperity Hubs",
                "candidates": ["Snowy High-Earner Cities", "Cold-Weather Power Metros", "Snowbelt Prosperity Hubs"],
                "description": "Dense, very snowy metros with strong incomes and a work-heavy mix.",
                "tradeoffs": "Winter intensity; higher day-to-day friction (weather/commute).",
            },
            1: {
                "final_name": "Stagnating Industrial Towns",
                "candidates": ["Stagnating Industrial Towns", "Slow-Growth Factory Markets", "Post-Boom Manufacturing Metros"],
                "description": "Lower-growth metros with an industrial footprint and weaker momentum.",
                "tradeoffs": "Often more affordable, but fewer fast-growth opportunities.",
            },
            2: {
                "final_name": "Recovery & Resilience Regions",
                "candidates": ["High-Need Health Burden Metros", "Tough-Break Cities", "Recovery & Resilience Regions"],
                "description": "Metros that stand out for compounding health and economic challenges.",
                "tradeoffs": "Potential affordability; harder quality-of-life baseline.",
            },
            3: {
                "final_name": "Premium Snowbelt City Centers",
                "candidates": ["Premium Snowbelt City Centers", "High-Rent Winter Metros", "Dense & Expensive Snow Cities"],
                "description": "Ultra-urban snowbelt metros with premium housing costs and strong urban intensity.",
                "tradeoffs": "High cost and winter; pays back in access/amenities.",
            },
            4: {
                "final_name": "Starter-Career City Hubs",
                "candidates": ["Young Urban Rent Hotspots", "Starter-Career City Hubs", "Youthful Walkable Metros"],
                "description": "Young-skewing metros that feel career-starter friendly—walkable, active, but pricier.",
                "tradeoffs": "High rent; can feel competitive.",
            },
            5: {
                "final_name": "Premium Price Metros",
                "candidates": ["Luxury Housing Markets", "High-Cost Homeowner Havens", "Premium Price Metros"],
                "description": "Metros defined primarily by high housing costs.",
                "tradeoffs": "Budget pressure; often requires higher income.",
            },
            6: {
                "final_name": "Rent-First Urban Hubs",
                "candidates": ["High-Rent Dense Metros", "Apartment-Heavy City Markets", "Rent-First Urban Hubs"],
                "description": "Dense metros where rent runs high relative to the national baseline.",
                "tradeoffs": "Less space per dollar; cost.",
            },
            7: {
                "final_name": "Sunbelt Stability Cities",
                "candidates": ["Warm Job-Secure Metros", "Sunbelt Stability Cities", "Low-Unemployment Warm Hubs"],
                "description": "Warmer metros with notably better unemployment profiles.",
                "tradeoffs": "May trade walkability for sprawl depending on metro.",
            },
            8: {
                "final_name": "Winter Work Squeeze Metros",
                "candidates": ["Snowy Job-Tight Markets", "Cold-Weather Strugglers", "Winter Work Squeeze Metros"],
                "description": "Snow-heavy metros with weaker job conditions.",
                "tradeoffs": "Job market risk; winter burden.",
            },
            9: {
                "final_name": "Budget Traditional Hubs",
                "candidates": ["Lower-Income Homogeneous Metros", "Small-Market Traditions", "Budget Traditional Hubs"],
                "description": "Metros that skew less diverse and lower income vs average.",
                "tradeoffs": "Lower costs may exist; fewer big-city network effects.",
            },
            10: {
                "final_name": "Cold-Climate Knowledge Hubs",
                "candidates": ["Educated Snowbelt Cities", "Winter College-Town Networks", "Cold-Climate Knowledge Hubs"],
                "description": "Snowbelt metros with stronger education signals.",
                "tradeoffs": "Winter; housing varies widely.",
            },
            11: {
                "final_name": "Safe but Slow Metros",
                "candidates": ["Safe but Slow Metros", "Quiet Stability Markets", "Low-Crime Slow-Growth Cities"],
                "description": "Lower-crime metros that aren’t growing quickly.",
                "tradeoffs": "Stability over dynamism; fewer breakout opportunities.",
            },
            12: {
                "final_name": "Compact Costly Cities",
                "candidates": ["Dense High-Rent Cores", "Compact Costly Cities", "Transit-Lean Rent Markets"],
                "description": "Dense metros where rent is a defining feature.",
                "tradeoffs": "Cost vs access.",
            },
            13: {
                "final_name": "Sunbelt Sweet-Spot Cities",
                "candidates": ["Warm Comfort Metros", "Sunbelt Sweet-Spot Cities", "Warm & Doing-Well Hubs"],
                "description": "Warmer metros with better poverty outcomes.",
                "tradeoffs": "Some may be car-dependent; housing can be trending up.",
            },
            14: {
                "final_name": "Snowbelt Blue-Collar Strongholds",
                "candidates": ["Hard-Winter Factory Towns", "Old-School Snowbelt Industry", "Snowbelt Blue-Collar Strongholds"],
                "description": "Cold, manufacturing-heavy snowbelt metros with lower diversity and fewer leisure-economy signals.",
                "tradeoffs": "Winter + fewer lifestyle amenities; often more affordable.",
            },
            15: {
                "final_name": "Upscale Healthy Growth Metros",
                "candidates": ["Upscale Healthy Growth Metros", "High-Cost Growth & Wellness", "Booming Affluent Enclaves"],
                "description": "High-cost, fast-growing metros with comparatively better health signals.",
                "tradeoffs": "Expensive; can feel exclusive.",
            },
            16: {
                "final_name": "Top-Tier Career Cities",
                "candidates": ["Elite Urban Job Centers", "High-Income High-Rent Capitals", "Top-Tier Career Cities"],
                "description": "Dense, expensive, high-income metros with strong professional job mix.",
                "tradeoffs": "Cost and competition; less space.",
            },
            17: {
                "final_name": "Snowy Mature Markets",
                "candidates": ["Older Snowbelt Settlers", "Cold-Climate Retirement Towns", "Snowy Mature Markets"],
                "description": "Snowbelt metros with an older age profile and somewhat better obesity outcomes.",
                "tradeoffs": "Winter; slower youth/career churn.",
            },
            18: {
                "final_name": "Booming but Aging Cities",
                "candidates": ["Growing Older-Skew Metros", "Booming but Aging Cities", "Expansion-Phase Heartland"],
                "description": "Growing metros that still skew older and show some health-burden signals.",
                "tradeoffs": "Growth upside; health profile not as strong.",
            },
            19: {
                "final_name": "Frontier Risk-Reward Cities",
                "candidates": ["Rough-Edge Boomtowns", "Young Resource Metros", "Frontier Risk-Reward Cities"],
                "description": "Younger resource-linked metros with higher crime and hardship signals—high-variance places.",
                "tradeoffs": "Opportunity can exist; stability/safety can lag.",
            },
            20: {
                "final_name": "Cold-Season Worktowns",
                "candidates": ["Cold-Season Worktowns", "Seasonal Recreation & Work Metros", "Snow-Season Blue-Collar Mix"],
                "description": "Seasonal metros with colder winters and a work-oriented mix.",
                "tradeoffs": "Winter + lifestyle-health risk signals.",
            },
            21: {
                "final_name": "Active Risk-Tradeoff Cities",
                "candidates": ["Young Government & Nightlife Hubs", "Active Risk-Tradeoff Cities", "Walkable Young Metros"],
                "description": "Younger metros with more activity and walkability, but elevated violent-crime risk.",
                "tradeoffs": "Lifestyle/access vs safety.",
            },
        },
    }

    name_map = cluster_names(df, label_col).set_index(label_col)[f"{label_col}_name"].to_dict()

    rows = []
    for label in sorted(df[label_col].unique()):
        mask = df[label_col] == label
        means = num.loc[mask].mean(numeric_only=True)
        z = ((means - overall_med) / overall_iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top = z.abs().sort_values(ascending=False).head(top_traits)

        trait_lines = []
        trait_phrases = []
        for f in top.index.tolist():
            delta = float(means[f] - overall_mean.get(f, 0.0))
            direction = "higher" if delta >= 0 else "lower"
            trait_lines.append(
                f"{f}: {direction} vs avg ({means[f]:.4g} vs {overall_mean.get(f, np.nan):.4g})"
            )
            if f in feature_to_phrase:
                pos, neg = feature_to_phrase[f]
                trait_phrases.append(pos if float(z[f]) >= 0 else neg)

        hard = HARDCODED.get(label_col, {}).get(int(label))
        final_name = (hard or {}).get("final_name") or name_map.get(int(label), "Balanced")
        cands = (hard or {}).get("candidates") or candidate_names(trait_phrases[:5])

        # 1–2 sentence description
        if hard and hard.get("description"):
            desc = hard["description"]
        elif trait_phrases:
            desc = f"Metros that skew {trait_phrases[0].lower()}" + (
                f", often also {trait_phrases[1].lower()}." if len(trait_phrases) > 1 else "."
            )
        else:
            desc = "A balanced mix of metros without a single dominating trait."

        # Trade-offs (simple, user-facing)
        trade = []
        if "High-rent" in trait_phrases or "Expensive housing" in trait_phrases:
            trade.append("higher housing costs")
        if "Car-dependent" in trait_phrases or "More spread out" in trait_phrases:
            trade.append("less walkability")
        if "Higher violent crime" in trait_phrases or "Higher property crime" in trait_phrases:
            trade.append("more safety risk")
        if "Snowier" in trait_phrases:
            trade.append("harsher winters")
        if "Higher poverty" in trait_phrases or "Higher unemployment" in trait_phrases:
            trade.append("weaker economic conditions")
        if hard and hard.get("tradeoffs"):
            tradeoffs = hard["tradeoffs"]
        else:
            tradeoffs = "high variation by metro" if not trade else ", ".join(trade)

        rows.append(
            {
                "key_traits": " | ".join(trait_lines[:top_traits]),
                "candidate_name_1": cands[0],
                "candidate_name_2": cands[1],
                "candidate_name_3": cands[2],
                "final_name": final_name,
                "description": desc,
                "tradeoffs": tradeoffs,
                label_col: int(label),
            }
        )

    return pd.DataFrame(rows).sort_values([label_col]).reset_index(drop=True)


def assign_clusters(df: pd.DataFrame, *, save: bool = True):
    """Returns (full_df_with_labels, cbsa_mapping_df, metrics_df, label_names_df). Optionally writes CSVs."""
    feat = df.drop(columns=DROP_COLS).copy()
    feat = apply_variance_stabilizing_transforms(feat)

    X = RobustScaler().fit_transform(feat)
    X = pd.DataFrame(X, columns=feat.columns, index=feat.index)
    X = selective_winsorize_robust(X)

    X_pca = PCA(n_components=0.9, random_state=RANDOM_STATE).fit_transform(X)
    X_reduced = X_pca[:, :3]

    # Finalized models from notebooks/02_clustering (2).ipynb
    cluster = KMeans(5, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced)
    sub_cluster = KMeans(22, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced)

    full = df.copy()
    full["cluster"] = cluster
    full["sub_cluster"] = sub_cluster

    cluster_short = cluster_names(full, "cluster").rename(columns={"cluster_name": "cluster_text"})
    sub_short = cluster_names(full, "sub_cluster").rename(columns={"sub_cluster_name": "sub_cluster_text"})
    full = full.merge(cluster_short, on="cluster", how="left")
    full = full.merge(sub_short, on="sub_cluster", how="left")

    # Rich label report (traits, candidates, description, trade-offs)
    report_cluster = build_cluster_label_report(full, "cluster").rename(
        columns={
            "final_name": "cluster_final_name",
            "description": "cluster_description",
            "tradeoffs": "cluster_tradeoffs",
            "candidate_name_1": "cluster_candidate_name_1",
            "candidate_name_2": "cluster_candidate_name_2",
            "candidate_name_3": "cluster_candidate_name_3",
            "key_traits": "cluster_key_traits",
        }
    )
    report_sub = build_cluster_label_report(full, "sub_cluster").rename(
        columns={
            "final_name": "sub_cluster_final_name",
            "description": "sub_cluster_description",
            "tradeoffs": "sub_cluster_tradeoffs",
            "candidate_name_1": "sub_cluster_candidate_name_1",
            "candidate_name_2": "sub_cluster_candidate_name_2",
            "candidate_name_3": "sub_cluster_candidate_name_3",
            "key_traits": "sub_cluster_key_traits",
        }
    )
    full = full.merge(report_cluster, on="cluster", how="left")
    full = full.merge(report_sub, on="sub_cluster", how="left")

    mapping = full[
        [
            "cbsa_code",
            "cluster",
            "cluster_text",
            "cluster_key_traits",
            "cluster_candidate_name_1",
            "cluster_candidate_name_2",
            "cluster_candidate_name_3",
            "cluster_final_name",
            "cluster_description",
            "cluster_tradeoffs",
            "sub_cluster",
            "sub_cluster_text",
            "sub_cluster_key_traits",
            "sub_cluster_candidate_name_1",
            "sub_cluster_candidate_name_2",
            "sub_cluster_candidate_name_3",
            "sub_cluster_final_name",
            "sub_cluster_description",
            "sub_cluster_tradeoffs",
        ]
    ].copy()

    metrics = pd.DataFrame(
        [
            {
                "model": "pcareducedk5",
                "n_clusters": 5,
                "cluster_min_size": int(pd.Series(cluster).value_counts().min()),
                "cluster_max_size": int(pd.Series(cluster).value_counts().max()),
                "silhouette_score": float(silhouette_score(X_reduced, cluster)),
                "wcss": wcss(X_reduced, cluster),
            },
            {
                "model": "pcareducedk22",
                "n_clusters": 22,
                "cluster_min_size": int(pd.Series(sub_cluster).value_counts().min()),
                "cluster_max_size": int(pd.Series(sub_cluster).value_counts().max()),
                "silhouette_score": float(silhouette_score(X_reduced, sub_cluster)),
                "wcss": wcss(X_reduced, sub_cluster),
            },
        ]
    )

    if save:
        CLUSTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        full.to_csv(OUT_FULL, index=False)
        mapping.to_csv(OUT_MAPPING, index=False)
        metrics.to_csv(OUT_METRICS, index=False)
        labels_export = report_cluster.copy()
        labels_export.insert(0, "level", "cluster")
        labels_export = labels_export.rename(
            columns={
                "cluster": "label",
                "cluster_key_traits": "key_traits",
                "cluster_candidate_name_1": "candidate_name_1",
                "cluster_candidate_name_2": "candidate_name_2",
                "cluster_candidate_name_3": "candidate_name_3",
                "cluster_final_name": "final_name",
                "cluster_description": "description",
                "cluster_tradeoffs": "tradeoffs",
            }
        )

        report_sub_export = report_sub.copy()
        report_sub_export.insert(0, "level", "sub_cluster")
        report_sub_export = report_sub_export.rename(
            columns={
                "sub_cluster": "label",
                "sub_cluster_key_traits": "key_traits",
                "sub_cluster_candidate_name_1": "candidate_name_1",
                "sub_cluster_candidate_name_2": "candidate_name_2",
                "sub_cluster_candidate_name_3": "candidate_name_3",
                "sub_cluster_final_name": "final_name",
                "sub_cluster_description": "description",
                "sub_cluster_tradeoffs": "tradeoffs",
            }
        )

        pd.concat([labels_export, report_sub_export], ignore_index=True).to_csv(OUT_LABELS, index=False)

    return full, mapping, metrics, report_cluster, report_sub


def build_cluster_mapping(df: pd.DataFrame, *, save: bool = False) -> pd.DataFrame:
    """
    Convenience wrapper to generate only the CBSA-level cluster label columns.

    Returns one row per cbsa_code with all cluster label columns produced by the model.
    """
    _, mapping, _, _, _ = assign_clusters(df, save=save)
    return mapping.copy()


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    _, _, metrics, _, _ = assign_clusters(df)
    print(metrics.to_string(index=False))
    print(
        f"\nWrote:\n  {OUT_FULL.resolve()}\n  {OUT_MAPPING.resolve()}\n  {OUT_METRICS.resolve()}\n  {OUT_LABELS.resolve()}"
    )


if __name__ == "__main__":
    main()
