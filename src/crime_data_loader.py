"""
CBSA-level crime counts from FBI city-level crime data.

Pipeline:
- Load FBI city crime from data/raw/FBI_Crime_Data_By_City_with_Counties.csv
- Normalize city + state
- Map (city, state) -> CBSA code using data/raw/ZIP_CBSA_122023.csv
  (uses most common CBSA per city/state in the ZIP file)
- Aggregate city crime to CBSA level

Example
-------
from crime_data_loader import load_cbsa_crime_dataset

crime_df = load_cbsa_crime_dataset(
    city_crime_path="data/raw/FBI_Crime_Data_By_City_with_Counties.csv",
    zip_cbsa_path="data/raw/ZIP_CBSA_122023.csv",
)
crime_df.head()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATE_TO_ABBR: dict[str, str] = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC",
    "PUERTO RICO": "PR",
}


def normalize_city(s: pd.Series) -> pd.Series:
    """Uppercase and collapse spaces for city names."""
    return s.astype(str).str.strip().str.replace("\n", " ").str.replace(r"\s+", " ", regex=True).str.upper()


def state_full_to_abbr(s: pd.Series) -> pd.Series:
    """Map full state names to 2-letter postal abbreviations."""
    return s.astype(str).str.strip().str.upper().map(STATE_TO_ABBR)


def load_fbi_city_crime(path: str | Path) -> pd.DataFrame:
    """
    Load FBI city-level crime and compute total_crime_count.

    Expected columns (FBI_Crime_Data_By_City_with_Counties.csv):
    State, City, Population, Violent crime, Property crime, State_Clean, City_Clean.
    """
    df = pd.read_csv(Path(path), encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\r\n", "\n", regex=False)
    df = df.rename(columns={"Violent\ncrime": "violent_crime_count", "Property\ncrime": "property_crime_count"})

    df["violent_crime_count"] = pd.to_numeric(
        df["violent_crime_count"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df["property_crime_count"] = pd.to_numeric(
        df["property_crime_count"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df["total_crime_count"] = df["violent_crime_count"].fillna(0) + df["property_crime_count"].fillna(0)

    df["city_clean"] = df["City_Clean"].astype(str).str.strip()
    df["state_abbr"] = state_full_to_abbr(df["State_Clean"])
    df["city_norm"] = normalize_city(df["city_clean"])

    return df[
        [
            "city_clean",
            "city_norm",
            "state_abbr",
            "violent_crime_count",
            "property_crime_count",
            "total_crime_count",
        ]
    ]


def load_zip_cbsa_mapping(path: str | Path) -> pd.DataFrame:
    """
    Load ZIP-to-CBSA file and build a city/state -> CBSA mapping
    using the most common CBSA per city/state in the ZIP file.
    """
    z = pd.read_csv(Path(path))
    z["city_norm"] = normalize_city(z["USPS_ZIP_PREF_CITY"])
    z["state_abbr"] = z["USPS_ZIP_PREF_STATE"].astype(str).str.strip().str.upper()
    z["cbsa_code"] = z["CBSA"].astype(str).str.strip()
    return (
        z.groupby(["city_norm", "state_abbr"])["cbsa_code"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )


def merge_city_crime_to_cbsa(
    city_df: pd.DataFrame,
    city_to_cbsa_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cbsa_code to FBI city crime and aggregate to CBSA."""
    merged = city_df.merge(city_to_cbsa_df, on=["city_norm", "state_abbr"], how="left")
    mapped = merged.dropna(subset=["cbsa_code"]).copy()
    agg_cols = ["violent_crime_count", "property_crime_count", "total_crime_count"]
    mapped["cbsa_code"] = mapped["cbsa_code"].astype(str).str.strip()
    cbsa = mapped.groupby("cbsa_code", as_index=False)[agg_cols].sum(min_count=1)
    return cbsa[["cbsa_code"] + agg_cols]


def load_cbsa_crime_dataset(
    city_crime_path: str = "data/raw/FBI_Crime_Data_By_City_with_Counties.csv",
    zip_cbsa_path: str = "data/raw/ZIP_CBSA_122023.csv",
) -> pd.DataFrame:
    """
    One-call loader: FBI city data -> CBSA-level crime counts.

    Returns a DataFrame with columns:
    - cbsa_code
    - violent_crime_count
    - property_crime_count
    - total_crime_count
    """
    city = load_fbi_city_crime(city_crime_path)
    city_to_cbsa = load_zip_cbsa_mapping(zip_cbsa_path)
    return merge_city_crime_to_cbsa(city, city_to_cbsa)


if __name__ == "__main__":
    """
    Script entrypoint: build the CBSA-level crime dataset and write it to CSV.

    Usage (from project root):
        python -m src.crime_data_loader
    """
    output_path = Path("data/processed/Crime_Data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crime_df = load_cbsa_crime_dataset()
    crime_df.to_csv(output_path, index=False)
    print(f"Wrote {len(crime_df)} rows to {output_path}")

