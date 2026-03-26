"""
CBSA-level ACS 5-year (2023) data loader. Fetches from api.census.gov and builds
lifestyle features. Use load_cbsa_lifestyle_dataset() in a notebook for a full run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Raised when the Census API returns an error or unexpected response.
class CensusApiError(RuntimeError):
    pass


CBSA_GEO_FIELD = "metropolitan statistical area/micropolitan statistical area"
ACS_YEAR = 2023
# Prior year for growth rates (population, jobs, rent, home value). ACS 5-year available back to ~2009.
PRIOR_YEAR_GROWTH = 2020

# Variables needed only for growth-rate calculation (fetched for PRIOR_YEAR_GROWTH).
GROWTH_VARIABLES: List[str] = [
    "B01003_001E",   # Total population
    "C24050_001E",   # Total employed (all industries)
    "B25064_001E",   # Median gross rent
    "B25077_001E",   # Median home value
]
GROWTH_VAR_LABELS: dict[str, str] = {
    "B01003_001E": "TotalPopulation_B01003",
    "C24050_001E": "IndustryTotalEmployed_C24050",
    "B25064_001E": "MedianGrossRent_B25064",
    "B25077_001E": "MedianHomeValue_B25077",
}


# All ACS variables we request (estimates only): core sociodemographics + industry totals + median age.
ACS_VARIABLES: List[str] = [
    # Income and population
    "B19013_001E",  # Median household income
    "B01003_001E",  # Total population
    "B01002_001E",  # Median age (total population)
    # Age structure (B01001 - sex by age buckets we need for 22–34 and 65+ shares)
    "B01001_009E",  # Male 22 to 24 years
    "B01001_010E",  # Male 25 to 29 years
    "B01001_011E",  # Male 30 to 34 years
    "B01001_019E",  # Male 65 to 66 years
    "B01001_020E",  # Male 67 to 69 years
    "B01001_021E",  # Male 70 to 74 years
    "B01001_022E",  # Male 75 to 79 years
    "B01001_023E",  # Male 80 to 84 years
    "B01001_024E",  # Male 85 years and over
    "B01001_032E",  # Female 22 to 24 years
    "B01001_033E",  # Female 25 to 29 years
    "B01001_034E",  # Female 30 to 34 years
    "B01001_042E",  # Female 65 to 66 years
    "B01001_043E",  # Female 67 to 69 years
    "B01001_044E",  # Female 70 to 74 years
    "B01001_045E",  # Female 75 to 79 years
    "B01001_046E",  # Female 80 to 84 years
    "B01001_047E",  # Female 85 years and over
    # Poverty (B17001 - poverty status in the past 12 months)
    "B17001_001E",  # Total population for whom poverty status is determined
    "B17001_002E",  # Income in the past 12 months below poverty level
    # Industry totals (C24050 - total employed by industry)
    "C24050_001E",  # Total: all industries
    "C24050_002E",  # Agriculture, forestry, fishing and hunting, and mining
    "C24050_003E",  # Construction
    "C24050_004E",  # Manufacturing
    "C24050_005E",  # Wholesale trade
    "C24050_006E",  # Retail trade
    "C24050_007E",  # Transportation and warehousing, and utilities
    "C24050_008E",  # Information
    "C24050_009E",  # Finance and insurance, and real estate and rental and leasing
    "C24050_010E",  # Professional, scientific, and management, and administrative and waste management services
    "C24050_011E",  # Educational services, and health care and social assistance
    "C24050_012E",  # Arts, entertainment, and recreation, and accommodation and food services
    "C24050_013E",  # Other services, except public administration
    "C24050_014E",  # Public administration
    # Occupation structure (C24010 - used for white/blue-collar and related shares)
    "C24010_001E", "C24010_003E", "C24010_004E", "C24010_006E", "C24010_007E", "C24010_008E", "C24010_009E",
    # Education (B15003 - attainment 25+)
    "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
    # Race diversity (B02001)
    "B02001_001E", "B02001_002E", "B02001_003E", "B02001_004E", "B02001_005E", "B02001_006E", "B02001_007E", "B02001_008E",
    # Housing costs
    "B25064_001E",  # Median gross rent
    "B25077_001E",  # Median home value
    # Labor force and unemployment (B23025)
    "B23025_003E", "B23025_005E",
]

# CamelCase labels for raw ACS columns: descriptive name + table id.
VAR_LABELS: dict[str, str] = {
    # Income and population
    "B19013_001E": "MedianHouseholdIncome_B19013",
    "B01003_001E": "TotalPopulation_B01003",
    "B01002_001E": "MedianAge_B01002",
    # Age structure (B01001 - only buckets we use)
    "B01001_009E": "Male_22to24_B01001",
    "B01001_010E": "Male_25to29_B01001",
    "B01001_011E": "Male_30to34_B01001",
    "B01001_019E": "Male_65to66_B01001",
    "B01001_020E": "Male_67to69_B01001",
    "B01001_021E": "Male_70to74_B01001",
    "B01001_022E": "Male_75to79_B01001",
    "B01001_023E": "Male_80to84_B01001",
    "B01001_024E": "Male_85plus_B01001",
    "B01001_032E": "Female_22to24_B01001",
    "B01001_033E": "Female_25to29_B01001",
    "B01001_034E": "Female_30to34_B01001",
    "B01001_042E": "Female_65to66_B01001",
    "B01001_043E": "Female_67to69_B01001",
    "B01001_044E": "Female_70to74_B01001",
    "B01001_045E": "Female_75to79_B01001",
    "B01001_046E": "Female_80to84_B01001",
    "B01001_047E": "Female_85plus_B01001",
    # Poverty (B17001)
    "B17001_001E": "TotalPovertyUniverse_B17001",
    "B17001_002E": "BelowPoverty_B17001",
    # Industry totals (C24050)
    "C24050_001E": "IndustryTotalEmployed_C24050",
    "C24050_002E": "IndustryAgForestryMining_C24050",
    "C24050_003E": "IndustryConstruction_C24050",
    "C24050_004E": "IndustryManufacturing_C24050",
    "C24050_005E": "IndustryWholesaleTrade_C24050",
    "C24050_006E": "IndustryRetailTrade_C24050",
    "C24050_007E": "IndustryTransportUtilities_C24050",
    "C24050_008E": "IndustryInformation_C24050",
    "C24050_009E": "IndustryFinanceRealEstate_C24050",
    "C24050_010E": "IndustryProfSciMgmtAdminWaste_C24050",
    "C24050_011E": "IndustryEducationHealthCare_C24050",
    "C24050_012E": "IndustryArtsRecAccommodationFood_C24050",
    "C24050_013E": "IndustryOtherServices_C24050",
    "C24050_014E": "IndustryPublicAdmin_C24050",
    # Occupation (C24010)
    "C24010_001E": "TotalEmployed_C24010",
    "C24010_003E": "ManagementBusiness_C24010",
    "C24010_004E": "ScienceEngineering_C24010",
    "C24010_006E": "ServiceOccupations_C24010",
    "C24010_007E": "SalesOffice_C24010",
    "C24010_008E": "Construction_C24010",
    "C24010_009E": "ProductionTransportation_C24010",
    # Education attainment 25+ (B15003)
    "B15003_001E": "TotalPopulation25Plus_B15003",
    "B15003_022E": "BachelorsDegree_B15003",
    "B15003_023E": "MastersDegree_B15003",
    "B15003_024E": "ProfessionalDegree_B15003",
    "B15003_025E": "DoctorateDegree_B15003",
    # Race diversity (B02001)
    "B02001_001E": "TotalRace_B02001",
    "B02001_002E": "WhiteAlone_B02001",
    "B02001_003E": "BlackAlone_B02001",
    "B02001_004E": "AmericanIndianAlaskaNative_B02001",
    "B02001_005E": "AsianAlone_B02001",
    "B02001_006E": "NativeHawaiianPacificIslander_B02001",
    "B02001_007E": "SomeOtherRace_B02001",
    "B02001_008E": "TwoOrMoreRaces_B02001",
    # Housing costs
    "B25064_001E": "MedianGrossRent_B25064",
    "B25077_001E": "MedianHomeValue_B25077",
    # Labor force and unemployment (B23025)
    "B23025_003E": "LaborForce_B23025",
    "B23025_005E": "Unemployed_B23025",
}

# Race-alone variables used for Simpson diversity index.
RACE_GROUP_VARS: list[str] = [f"B02001_{i:03d}E" for i in range(2, 9)]
RACE_GROUP_LABELS: list[str] = [VAR_LABELS[v] for v in RACE_GROUP_VARS]
# Rename raw ACS columns to camelCase label + table id (e.g. MedianIncome_B19013).
def apply_column_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw ACS columns to camelCase label + table id. Leaves other columns unchanged."""
    rename = {c: VAR_LABELS[c] for c in df.columns if c in VAR_LABELS}
    return df.rename(columns=rename)


# Divide two Series; return NaN where denominator is 0 or NaN.
def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace({0: np.nan})
    return numer / denom_safe


# Cast non-id columns to numeric (errors='coerce').
def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in ("cbsa_code", "cbsa_name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_session() -> Session:
    """Create a requests Session with retry logic for Census API calls."""
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_acs_cbsa_request(
    session: Session,
    year: int,
    get_vars: List[str],
    api_key: str | None,
    timeout: int,
) -> pd.DataFrame:
    """One GET request for ACS 5-year at CBSA level. Returns df with cbsa_code, cbsa_name, and requested vars."""
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    params: dict[str, str] = {
        "get": ",".join(get_vars),
        "for": "metropolitan statistical area/micropolitan statistical area:*",
    }
    if api_key:
        params["key"] = api_key
    resp = session.get(endpoint, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise CensusApiError(f"HTTP {resp.status_code} from Census API")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise CensusApiError("Invalid JSON from Census API")
    if isinstance(data, dict) and "error" in data:
        raise CensusApiError(f"Census API error: {data['error']}")
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise CensusApiError("Unexpected Census API response shape")
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={CBSA_GEO_FIELD: "cbsa_code", "NAME": "cbsa_name"})
    return df


# Fetch ACS 5-year 2023 for all CBSA; returns raw variables plus cbsa_code, cbsa_name.
def get_acs_cbsa_2023(api_key: str | None = None, timeout: int = 60) -> pd.DataFrame:
    """Fetch ACS 5-year 2023 for all CBSA. api_key optional (env CENSUS_API_KEY)."""
    key = api_key or os.getenv("CENSUS_API_KEY")
    session = make_session()

    dfs: list[pd.DataFrame] = []
    chunk_size = 40
    for i in range(0, len(ACS_VARIABLES), chunk_size):
        vars_chunk = ACS_VARIABLES[i : i + chunk_size]
        get_vars = list(vars_chunk)
        if i == 0 and "NAME" not in get_vars:
            get_vars.insert(0, "NAME")
        df_chunk = fetch_acs_cbsa_request(session, ACS_YEAR, get_vars, key, timeout)
        dfs.append(df_chunk)

    if not dfs:
        raise CensusApiError("No data returned from Census API")
    df = dfs[0]
    for extra in dfs[1:]:
        df = df.merge(extra.drop(columns=["cbsa_name"], errors="ignore"), on="cbsa_code", how="outer")
    df = cast_numeric(df)
    id_cols = ["cbsa_code", "cbsa_name"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]
    return df


def get_acs_cbsa_prior_for_growth(api_key: str | None = None, timeout: int = 60) -> pd.DataFrame:
    """Fetch ACS 5-year for PRIOR_YEAR_GROWTH with only variables needed for growth rates.
    Returns df with cbsa_code and columns named like TotalPopulation_B01003_prior, etc.
    """
    key = api_key or os.getenv("CENSUS_API_KEY")
    session = make_session()

    get_vars = ["NAME"] + GROWTH_VARIABLES
    df = fetch_acs_cbsa_request(session, PRIOR_YEAR_GROWTH, get_vars, key, timeout)
    df = cast_numeric(df)
    rename = {v: f"{GROWTH_VAR_LABELS[v]}_prior" for v in GROWTH_VARIABLES if v in df.columns}
    df = df.rename(columns=rename)
    df = df[["cbsa_code"] + [c for c in df.columns if c != "cbsa_code" and c != "cbsa_name"]]
    return df


def add_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add population_growth, job_growth, median_rent_growth, median_home_value_growth (current vs prior year).
    Expects df to have current columns and _prior columns from merge with get_acs_cbsa_prior_for_growth().
    """
    df = df.copy()
    pop_curr = df["TotalPopulation_B01003"]
    pop_prior = df["TotalPopulation_B01003_prior"]
    df["population_growth"] = safe_divide(pop_curr - pop_prior, pop_prior)

    jobs_curr = df["IndustryTotalEmployed_C24050"]
    jobs_prior = df["IndustryTotalEmployed_C24050_prior"]
    df["job_growth"] = safe_divide(jobs_curr - jobs_prior, jobs_prior)

    rent_curr = df["MedianGrossRent_B25064"]
    rent_prior = df["MedianGrossRent_B25064_prior"]
    df["median_rent_growth"] = safe_divide(rent_curr - rent_prior, rent_prior)

    home_curr = df["MedianHomeValue_B25077"]
    home_prior = df["MedianHomeValue_B25077_prior"]
    df["median_home_value_growth"] = safe_divide(home_curr - home_prior, home_prior)
    return df


# Parse cbsa_name ("City-Region, ST Metro Area" or "... Micro Area") into city, state, cbsa_type.
def add_cbsa_name_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add city, state, and cbsa_type (Metro/Micro) by splitting cbsa_name. Expects cbsa_name column."""
    df = df.copy()
    parts = df["cbsa_name"].fillna("").astype(str).str.split(", ", n=1, expand=True)
    df["city"] = parts[0]
    state_type = parts[1].fillna("")
    df["cbsa_type"] = np.where(state_type.str.endswith(" Metro Area"), "Metro", np.where(state_type.str.endswith(" Micro Area"), "Micro", None))
    df["state"] = state_type.str.replace(r"\s+Metro Area$", "", regex=True).str.replace(r"\s+Micro Area$", "", regex=True).str.strip()
    return df


# Add derived industry/occupation/education/affordability/diversity columns.
# Expects df with mapped column names (labels).
def build_lifestyle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived share columns and diversity index. Input must have labeled columns (after apply_column_labels)."""
    df = df.copy()

    # Industry shares (use C24050 total employed by industry / total employed all industries).
    tot_ind = df["IndustryTotalEmployed_C24050"]
    df["industry_ag_forestry_mining_share"] = safe_divide(df["IndustryAgForestryMining_C24050"], tot_ind) * 100
    df["industry_construction_share"] = safe_divide(df["IndustryConstruction_C24050"], tot_ind) * 100
    df["industry_manufacturing_share"] = safe_divide(df["IndustryManufacturing_C24050"], tot_ind) * 100
    df["industry_wholesale_share"] = safe_divide(df["IndustryWholesaleTrade_C24050"], tot_ind) * 100
    df["industry_retail_share"] = safe_divide(df["IndustryRetailTrade_C24050"], tot_ind) * 100
    df["industry_transport_utilities_share"] = safe_divide(df["IndustryTransportUtilities_C24050"], tot_ind) * 100
    df["industry_information_share"] = safe_divide(df["IndustryInformation_C24050"], tot_ind) * 100
    df["industry_finance_real_estate_share"] = safe_divide(df["IndustryFinanceRealEstate_C24050"], tot_ind) * 100
    df["industry_prof_sci_mgmt_admin_share"] = safe_divide(df["IndustryProfSciMgmtAdminWaste_C24050"], tot_ind) * 100
    df["industry_education_health_share"] = safe_divide(df["IndustryEducationHealthCare_C24050"], tot_ind) * 100
    df["industry_arts_rec_accommodation_food_share"] = (
        safe_divide(df["IndustryArtsRecAccommodationFood_C24050"], tot_ind) * 100
    )
    df["industry_other_services_share"] = safe_divide(df["IndustryOtherServices_C24050"], tot_ind) * 100
    df["industry_public_admin_share"] = safe_divide(df["IndustryPublicAdmin_C24050"], tot_ind) * 100

    # Occupation shares (C24010).
    tot_occ = df["TotalEmployed_C24010"]
    df["white_collar_share"] = (
        safe_divide(df["ManagementBusiness_C24010"] + df["ScienceEngineering_C24010"], tot_occ) * 100
    )
    df["blue_collar_share"] = (
        safe_divide(df["Construction_C24010"] + df["ProductionTransportation_C24010"], tot_occ) * 100
    )
    df["service_job_share"] = safe_divide(df["ServiceOccupations_C24010"], tot_occ) * 100
    df["sales_share"] = safe_divide(df["SalesOffice_C24010"], tot_occ) * 100

    # Education: share of population 25+ with bachelor's degree or higher (B15003).
    bachelors_plus = (
        df["BachelorsDegree_B15003"]
        + df["MastersDegree_B15003"]
        + df["ProfessionalDegree_B15003"]
        + df["DoctorateDegree_B15003"]
    )
    df["bachelors_or_higher_share"] = safe_divide(bachelors_plus, df["TotalPopulation25Plus_B15003"]) * 100

    # Age structure: total counts and shares of age 22–34 and 65+ using B01001 buckets and total population.
    age_22_34 = (
        df["Male_22to24_B01001"]
        + df["Male_25to29_B01001"]
        + df["Male_30to34_B01001"]
        + df["Female_22to24_B01001"]
        + df["Female_25to29_B01001"]
        + df["Female_30to34_B01001"]
    )
    age_65_plus = (
        df["Male_65to66_B01001"]
        + df["Male_67to69_B01001"]
        + df["Male_70to74_B01001"]
        + df["Male_75to79_B01001"]
        + df["Male_80to84_B01001"]
        + df["Male_85plus_B01001"]
        + df["Female_65to66_B01001"]
        + df["Female_67to69_B01001"]
        + df["Female_70to74_B01001"]
        + df["Female_75to79_B01001"]
        + df["Female_80to84_B01001"]
        + df["Female_85plus_B01001"]
    )
    total_pop = df["TotalPopulation_B01003"]
    df["age_22_34_total"] = age_22_34
    df["age_65_plus_total"] = age_65_plus
    df["age_22_34_share"] = safe_divide(age_22_34, total_pop) * 100
    df["age_65_plus_share"] = safe_divide(age_65_plus, total_pop) * 100

    # Drop raw B01001 age bucket columns now that we have totals and shares.
    age_bucket_cols = [
        "Male_22to24_B01001",
        "Male_25to29_B01001",
        "Male_30to34_B01001",
        "Male_65to66_B01001",
        "Male_67to69_B01001",
        "Male_70to74_B01001",
        "Male_75to79_B01001",
        "Male_80to84_B01001",
        "Male_85plus_B01001",
        "Female_22to24_B01001",
        "Female_25to29_B01001",
        "Female_30to34_B01001",
        "Female_65to66_B01001",
        "Female_67to69_B01001",
        "Female_70to74_B01001",
        "Female_75to79_B01001",
        "Female_80to84_B01001",
        "Female_85plus_B01001",
    ]
    df = df.drop(columns=[c for c in age_bucket_cols if c in df.columns], errors="ignore")

    # Unemployment rate based on ACS labor force counts (B23025).
    df["unemployment_rate"] = safe_divide(df["Unemployed_B23025"], df["LaborForce_B23025"]) * 100

    # Poverty rate: share of population below poverty level (B17001).
    df["poverty_rate"] = safe_divide(df["BelowPoverty_B17001"], df["TotalPovertyUniverse_B17001"]) * 100

    # Race diversity (Simpson index) using race-alone categories.
    race_tot = df["TotalRace_B02001"]
    race_props = df[RACE_GROUP_LABELS].fillna(0).div(race_tot.replace(0, np.nan), axis=0)
    df["diversity_index"] = 1 - race_props.pow(2).sum(axis=1)

    return df



# Load CBSA Gazetteer file (2023_Gaz_cbsa_national.txt). Returns DataFrame with columns renamed; no key coercion.
def add_gazetteer_data(df: pd.DataFrame, land_area_path: str = "data/raw/2023_Gaz_cbsa_national.txt") -> pd.DataFrame:
    """Read Gazetteer txt (tab-separated) and rename columns. Join key/coercion is up to you."""
    path = Path(land_area_path)
    df2 = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df2.columns = df2.columns.str.strip()
    df2.rename(
        columns={
            "CSAFP": "csa_code",
            "GEOID": "cbsa_code",
            "NAME": "cbsa_name",
            "CBSA_TYPE": "cbsa_type",
            "ALAND": "land_area_m2",
            "AWATER": "water_area_m2",
            "ALAND_SQMI": "land_area_sqmi",
            "AWATER_SQMI": "water_area_sqmi",
            "INTPTLAT": "centroid_lat",
            "INTPTLONG": "centroid_lon",
        },
        inplace=True,
    )
    df2["cbsa_code"] = df2["cbsa_code"].astype(str).str.strip()
    df["cbsa_code"] = df["cbsa_code"].astype(str).str.strip()
    df2 = df2.drop(columns=["csa_code", "cbsa_name", "cbsa_type"], errors="ignore")
    df = df.merge(df2, on="cbsa_code", how="left")
    df = df.drop(columns=["csa_code", "cbsa_name_y", "cbsa_type_y"], errors="ignore")
    df["population_density_per_sq_mile"] = safe_divide(df["TotalPopulation_B01003"], df["land_area_sqmi"])
    return df


# One-call pipeline: fetch ACS, apply labels, add growth rates, build lifestyle features, add density.
def load_cbsa_lifestyle_dataset(
    api_key: str | None = None,
    land_area_path: str = "data/raw/2023_Gaz_cbsa_national.txt",
) -> pd.DataFrame:
    """Fetch ACS CBSA 2023, build lifestyle features, add growth rates and density."""
    df = get_acs_cbsa_2023(api_key=api_key)
    df = add_cbsa_name_components(df)
    df = apply_column_labels(df)

    # Prior-year data for growth rates (population, jobs, rent, home value).
    prior_df = get_acs_cbsa_prior_for_growth(api_key=api_key)
    df["cbsa_code"] = df["cbsa_code"].astype(str).str.strip()
    prior_df["cbsa_code"] = prior_df["cbsa_code"].astype(str).str.strip()
    df = df.merge(prior_df, on="cbsa_code", how="left")
    df = add_growth_rates(df)

    df = build_lifestyle_features(df)
    df = add_gazetteer_data(df, land_area_path=land_area_path)
    return df


if __name__ == "__main__":
    """
    Script entrypoint: build the full CBSA lifestyle dataset and write it to CSV.

    Usage (from project root, with CENSUS_API_KEY set if needed):
        python -m src.census_data_loader
    """
    output_path = Path("data/processed/Census_Data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    census_df = load_cbsa_lifestyle_dataset()
    census_df.to_csv(output_path, index=False)
    print(f"Wrote {len(census_df)} rows to {output_path}")
