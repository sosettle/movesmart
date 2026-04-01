"""
CBSA-level weather and climate features built from NOAA 2006–2020 monthly normals.

This module:
    1. Loads CBSA centroids from the Census Gazetteer file.
    2. Downloads NOAA monthly normals station CSVs from the public directory.
    3. Extracts monthly temperature / precipitation / snowfall normals.
    4. Summarizes them into annual station-level climate features.
    5. Maps each CBSA to its nearest station using great-circle distance.
    6. Returns a tidy CBSA-level dataframe of climate features.

Example
-------
    from src.weather_data_loader import load_cbsa_weather_dataset

    weather_df = load_cbsa_weather_dataset()
    print(weather_df.head())
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


NOAA_DEFAULT_BASE_URL = "https://www.ncei.noaa.gov/data/normals-monthly/2006-2020/access/"
DEFAULT_CACHE_DIR = Path("data/raw/weather/noaa_monthly_normals")
STATION_SUMMARY_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020.parquet")
STATION_METADATA_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020_metadata.json")
DEFAULT_CBSA_GAZETTEER = Path("data/raw/2023_Gaz_cbsa_national.txt")

# One row per station after summarizing monthly NOAA files (also Parquet schema).
STATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "station_id",
    "station_name",
    "station_lat",
    "station_lon",
    "avg_annual_temp",
    "jan_avg_temp",
    "jul_avg_temp",
    "annual_precipitation",
    "annual_snowfall",
    "n_months_available",
    "has_complete_year",
)

# Climate fields copied from the nearest station onto each CBSA row.
STATION_CLIMATE_COLUMNS: tuple[str, ...] = (
    "avg_annual_temp",
    "jan_avg_temp",
    "jul_avg_temp",
    "annual_precipitation",
    "annual_snowfall",
    "n_months_available",
    "has_complete_year",
)

# Final column order for CBSA-level output.
CBSA_WEATHER_OUTPUT_COLUMNS: tuple[str, ...] = (
    "cbsa_code",
    "cbsa_name",
    "cbsa_type",
    "centroid_lat",
    "centroid_lon",
    "station_id",
    "station_name",
    "station_distance_km",
    *STATION_CLIMATE_COLUMNS,
    "temp_seasonality",
    "snow_binary",
)

MONTH_DAY_COUNTS: dict[int, int] = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def create_session() -> Session:
    """Create a requests Session with retry logic for robustness."""
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


def standardize_column_name(name: str) -> str:
    """Convert NOAA column names to lowercase snake_case."""
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-z]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def standardize_noaa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NOAA station CSV columns to lowercase snake_case."""
    df = df.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]
    return df


def list_noaa_station_csvs(base_url: str = NOAA_DEFAULT_BASE_URL) -> list[str]:
    """Retrieve the NOAA directory HTML and parse all station CSV URLs."""
    session = create_session()
    resp: Response = session.get(base_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch NOAA directory listing: HTTP {resp.status_code}")

    csv_files = re.findall(r'href=["\']([^"\']+?\.csv)["\']', resp.text, flags=re.IGNORECASE)
    urls = [urljoin(base_url, f) for f in csv_files]
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    logger.info("Found %d NOAA station CSVs at %s", len(out), base_url)
    return out


def download_station_csv(
    url: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Download one station CSV to a local cache directory (skip if already present)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / Path(url).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    session = create_session()
    logger.debug("Downloading NOAA station CSV: %s", url)
    resp: Response = session.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {resp.status_code}")

    dest.write_bytes(resp.content)
    return dest


def parse_station_monthly_normals(path: str | Path) -> pd.DataFrame:
    """Load one station CSV and return typed monthly rows for downstream summaries."""
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read station CSV %s: %s", path, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df = standardize_noaa_columns(df)

    required_id_cols = ["station", "latitude", "longitude", "name", "month"]
    for col in required_id_cols:
        if col not in df.columns:
            logger.warning("Station CSV %s missing required column '%s'; skipping", path, col)
            return pd.DataFrame()

    keep_cols = [
        "station",
        "latitude",
        "longitude",
        "name",
        "month",
        "mly_tavg_normal",
        "mly_prcp_normal",
        "mly_snow_normal",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    for col in ("mly_tavg_normal", "mly_prcp_normal", "mly_snow_normal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["month"].between(1, 12, inclusive="both")]
    return df


def summarize_station_normals(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize one station's monthly normals into a single-row DataFrame."""
    if df.empty or "station" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    temp = df.get("mly_tavg_normal")
    prcp = df.get("mly_prcp_normal")

    if temp is None or prcp is None:
        return pd.DataFrame()

    temp = pd.to_numeric(temp, errors="coerce")
    prcp = pd.to_numeric(prcp, errors="coerce")

    if not temp.notna().any() or not prcp.notna().any():
        return pd.DataFrame()

    df["month"] = df["month"].astype("Int64")
    months_present = df["month"].dropna().unique()
    n_months_available = int(len(months_present))
    has_complete_year = n_months_available == 12

    month_days = df["month"].map(MONTH_DAY_COUNTS)
    mask_temp = temp.notna() & month_days.notna()
    if mask_temp.any():
        avg_annual_temp = float((temp[mask_temp] * month_days[mask_temp]).sum() / month_days[mask_temp].sum())
    else:
        avg_annual_temp = math.nan

    jan_vals = temp[df["month"] == 1].dropna()
    jul_vals = temp[df["month"] == 7].dropna()
    jan_avg_temp = float(jan_vals.iloc[0]) if not jan_vals.empty else math.nan
    jul_avg_temp = float(jul_vals.iloc[0]) if not jul_vals.empty else math.nan

    annual_precipitation = float(prcp.dropna().sum()) if prcp.notna().any() else math.nan

    snow = df.get("mly_snow_normal")
    if snow is None:
        annual_snowfall = math.nan
    else:
        snow = pd.to_numeric(snow, errors="coerce")
        annual_snowfall = float(snow.dropna().sum()) if snow.notna().any() else math.nan

    row = {
        "station_id": str(df["station"].iloc[0]),
        "station_name": str(df.get("name", pd.Series([""])).iloc[0]),
        "station_lat": float(df.get("latitude", pd.Series([math.nan])).iloc[0]),
        "station_lon": float(df.get("longitude", pd.Series([math.nan])).iloc[0]),
        "avg_annual_temp": avg_annual_temp,
        "jan_avg_temp": jan_avg_temp,
        "jul_avg_temp": jul_avg_temp,
        "annual_precipitation": annual_precipitation,
        "annual_snowfall": annual_snowfall,
        "n_months_available": n_months_available,
        "has_complete_year": has_complete_year,
    }
    return pd.DataFrame([row], columns=list(STATION_SUMMARY_COLUMNS))


def build_station_climate_normals_from_noaa(
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Download/parse NOAA station CSVs and write Parquet + metadata JSON."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    STATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    urls = list_noaa_station_csvs(base_url=base_url)
    if max_files is not None:
        urls = urls[:max_files]

    summaries: list[pd.DataFrame] = []
    n_files_total = len(urls)
    n_files_download_failed = 0
    n_files_parse_failed = 0
    n_stations_summarized = 0

    for idx, url in enumerate(urls, start=1):
        try:
            local_path = download_station_csv(url, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to download error: %s", url, exc)
            n_files_download_failed += 1
            continue

        try:
            monthly_df = parse_station_monthly_normals(local_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to parse error: %s", local_path, exc)
            n_files_parse_failed += 1
            continue

        if monthly_df.empty:
            n_files_parse_failed += 1
            continue

        summary_df = summarize_station_normals(monthly_df)
        if summary_df.empty:
            continue

        summaries.append(summary_df)
        n_stations_summarized += 1

        if idx % 100 == 0 or idx == n_files_total:
            logger.info(
                "Processed %d/%d NOAA station files (summarized %d stations, "
                "download failures=%d, parse/empty failures=%d)",
                idx,
                n_files_total,
                n_stations_summarized,
                n_files_download_failed,
                n_files_parse_failed,
            )

    if summaries:
        stations_df = pd.concat(summaries, ignore_index=True)
    else:
        stations_df = pd.DataFrame(columns=list(STATION_SUMMARY_COLUMNS))

    stations_df.to_parquet(STATION_SUMMARY_PATH, index=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "cache_dir": str(Path(cache_dir).as_posix()),
        "n_files_total": n_files_total,
        "n_files_download_failed": n_files_download_failed,
        "n_files_parse_failed": n_files_parse_failed,
        "n_stations_summarized": n_stations_summarized,
        "station_summary_path": str(STATION_SUMMARY_PATH.as_posix()),
    }
    STATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    return stations_df


def load_station_climate_normals(
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load cached station Parquet or rebuild from NOAA."""
    if STATION_SUMMARY_PATH.exists() and not force_rebuild:
        return pd.read_parquet(STATION_SUMMARY_PATH)
    return build_station_climate_normals_from_noaa(base_url=base_url, cache_dir=cache_dir, max_files=max_files)


def load_cbsa_centroids(cbsa_path: str | Path = DEFAULT_CBSA_GAZETTEER) -> pd.DataFrame:
    """Load CBSA centroids from the Census Gazetteer (tab-separated)."""
    path = Path(cbsa_path)
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "CSAFP": "csa_code",
            "GEOID": "cbsa_code",
            "NAME": "cbsa_name",
            "CBSA_TYPE": "cbsa_type",
            "INTPTLAT": "centroid_lat",
            "INTPTLONG": "centroid_lon",
        }
    )

    df["cbsa_code"] = df["cbsa_code"].astype(str).str.strip()
    df["cbsa_name"] = df["cbsa_name"].astype(str).str.strip()
    df["cbsa_type"] = df["cbsa_type"].astype(str).str.strip()
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")

    return df[["cbsa_code", "cbsa_name", "cbsa_type", "centroid_lat", "centroid_lon"]].copy()


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km; inputs in degrees; arrays broadcast."""
    r_earth_km = 6371.0088

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r_earth_km * c


def attach_nearest_station_to_cbsa(
    cbsa_df: pd.DataFrame,
    station_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map each CBSA centroid to its nearest station and attach climate features."""
    cbsa_df = cbsa_df.copy()
    station_df = station_df.copy()

    station_df = station_df.dropna(subset=["station_lat", "station_lon"])
    if station_df.empty:
        raise ValueError("No station climate records with valid coordinates are available.")

    cbsa_coords = cbsa_df[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float)
    station_coords = station_df[["station_lat", "station_lon"]].to_numpy(dtype=float)

    cbsa_lat = cbsa_coords[:, 0][:, np.newaxis]
    cbsa_lon = cbsa_coords[:, 1][:, np.newaxis]
    station_lat = station_coords[:, 0][np.newaxis, :]
    station_lon = station_coords[:, 1][np.newaxis, :]

    distances = haversine_distance_km(cbsa_lat, cbsa_lon, station_lat, station_lon)
    nearest_idx = distances.argmin(axis=1)
    nearest_dist_km = distances[np.arange(distances.shape[0]), nearest_idx]

    nearest_stations = station_df.iloc[nearest_idx].reset_index(drop=True)
    cbsa_df = cbsa_df.reset_index(drop=True)
    cbsa_df["station_id"] = nearest_stations["station_id"].values
    cbsa_df["station_name"] = nearest_stations["station_name"].values
    cbsa_df["station_distance_km"] = nearest_dist_km

    for col in STATION_CLIMATE_COLUMNS:
        cbsa_df[col] = nearest_stations[col].values

    cbsa_df["temp_seasonality"] = cbsa_df["jul_avg_temp"] - cbsa_df["jan_avg_temp"]
    cbsa_df["snow_binary"] = np.where(cbsa_df["annual_snowfall"] > 0, 1, 0)

    return cbsa_df[list(CBSA_WEATHER_OUTPUT_COLUMNS)]


def load_cbsa_weather_dataset(
    cbsa_path: str | Path = DEFAULT_CBSA_GAZETTEER,
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
    force_rebuild_stations: bool = False,
) -> pd.DataFrame:
    """
    Build CBSA-level weather features: gazetteer → station normals (cache or NOAA) → nearest station.

    Parameters
    ----------
    cbsa_path:
        Census Gazetteer CBSA file (tab-separated).
    base_url:
        NOAA 2006–2020 monthly normals directory.
    cache_dir:
        Local cache for per-station CSV downloads.
    max_files:
        Optional cap on NOAA files (testing).
    force_rebuild_stations:
        If True, rebuild the station Parquet even when it exists.
    """
    cbsa_df = load_cbsa_centroids(cbsa_path)
    station_df = load_station_climate_normals(
        base_url=base_url,
        cache_dir=cache_dir,
        max_files=max_files,
        force_rebuild=force_rebuild_stations,
    )
    return attach_nearest_station_to_cbsa(cbsa_df, station_df)


build_cbsa_weather_features = load_cbsa_weather_dataset


if __name__ == "__main__":
    # Writes data/processed/Weather_Data.csv. First run downloads many NOAA files (hours).
    out = Path("data/processed/Weather_Data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    weather_df = load_cbsa_weather_dataset()
    weather_df.to_csv(out, index=False)
    logger.info(
        "Wrote %d rows to %s (%d columns).",
        len(weather_df),
        out,
        len(weather_df.columns),
    )
    print(weather_df.head())
