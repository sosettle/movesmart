"""
CBSA-level weather features from NOAA 2006-2020 monthly normals station files.

Example
-------
from weather_data_loader import load_cbsa_weather_dataset

weather_df = load_cbsa_weather_dataset()
weather_df.head()
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


NOAA_DEFAULT_BASE_URL = "https://www.ncei.noaa.gov/data/normals-monthly/2006-2020/access/"
DEFAULT_CACHE_DIR = Path("data/raw/weather/noaa_monthly_normals")
STATION_SUMMARY_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020.parquet")
STATION_METADATA_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020_metadata.json")

MONTH_DAY_COUNTS = pd.Series(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
)

STATION_SUMMARY_COLUMNS = [
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
]


def create_session() -> requests.Session:
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
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", name.strip().lower())).strip("_")


def standardize_noaa_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [standardize_column_name(col) for col in df.columns]
    return df


def list_noaa_station_csvs(base_url: str = NOAA_DEFAULT_BASE_URL) -> list[str]:
    session = create_session()
    response = session.get(base_url, timeout=60)
    response.raise_for_status()
    station_files = re.findall(r'href=["\']([^"\']+\.csv)["\']', response.text, flags=re.IGNORECASE)
    return list(dict.fromkeys(urljoin(base_url, filename) for filename in station_files))


def download_station_csv(
    url: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / Path(url).name
    if path.exists():
        return path

    session = create_session()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def parse_station_monthly_normals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_noaa_columns(df)

    if "mly_snow_normal" not in df.columns:
        df["mly_snow_normal"] = np.nan

    df = df[
        [
            "station",
            "latitude",
            "longitude",
            "name",
            "month",
            "mly_tavg_normal",
            "mly_prcp_normal",
            "mly_snow_normal",
        ]
    ].copy()

    for column in [
        "latitude",
        "longitude",
        "month",
        "mly_tavg_normal",
        "mly_prcp_normal",
        "mly_snow_normal",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[df["month"].isin(MONTH_DAY_COUNTS.index)].copy()


def summarize_station_normals(df: pd.DataFrame) -> pd.DataFrame:
    if df["mly_tavg_normal"].notna().sum() == 0 or df["mly_prcp_normal"].notna().sum() == 0:
        return pd.DataFrame(columns=STATION_SUMMARY_COLUMNS)

    month_days = df["month"].map(MONTH_DAY_COUNTS)
    weighted_temp = (df["mly_tavg_normal"] * month_days).sum(skipna=True)
    temp_day_total = month_days[df["mly_tavg_normal"].notna()].sum()
    january = df.loc[df["month"] == 1, "mly_tavg_normal"].dropna()
    july = df.loc[df["month"] == 7, "mly_tavg_normal"].dropna()

    return pd.DataFrame(
        [
            {
                "station_id": df["station"].iloc[0],
                "station_name": df["name"].iloc[0],
                "station_lat": df["latitude"].iloc[0],
                "station_lon": df["longitude"].iloc[0],
                "avg_annual_temp": weighted_temp / temp_day_total,
                "jan_avg_temp": january.iloc[0] if not january.empty else np.nan,
                "jul_avg_temp": july.iloc[0] if not july.empty else np.nan,
                "annual_precipitation": df["mly_prcp_normal"].sum(skipna=True),
                "annual_snowfall": df["mly_snow_normal"].sum(skipna=True)
                if df["mly_snow_normal"].notna().any()
                else np.nan,
                "n_months_available": df["month"].nunique(),
                "has_complete_year": df["month"].nunique() == 12,
            }
        ]
    )


def build_station_climate_normals_from_noaa(
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    STATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    urls = list_noaa_station_csvs(base_url)
    if max_files is not None:
        urls = urls[:max_files]

    summaries = []
    skipped_files = 0

    for index, url in enumerate(urls, start=1):
        try:
            station_path = download_station_csv(url, cache_dir)
            station_df = parse_station_monthly_normals(station_path)
            summary_df = summarize_station_normals(station_df)
            if not summary_df.empty:
                summaries.append(summary_df)
        except Exception as exc:  # noqa: BLE001
            skipped_files += 1
            logger.warning("Skipping %s: %s", url, exc)

        if index % 100 == 0 or index == len(urls):
            logger.info("Processed %d/%d NOAA station files", index, len(urls))

    station_climate_df = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=STATION_SUMMARY_COLUMNS)
    )
    station_climate_df.to_parquet(STATION_SUMMARY_PATH, index=False)

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "base_url": base_url,
        "cache_dir": str(cache_dir),
        "file_count": len(urls),
        "skipped_file_count": skipped_files,
        "output_path": str(STATION_SUMMARY_PATH),
    }
    STATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return station_climate_df


def load_station_climate_normals(
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if STATION_SUMMARY_PATH.exists() and not force_rebuild:
        return pd.read_parquet(STATION_SUMMARY_PATH)
    return build_station_climate_normals_from_noaa(base_url=base_url, cache_dir=cache_dir, max_files=max_files)


def load_cbsa_centroids(cbsa_path: str | Path = "data/raw/2023_Gaz_cbsa_national.txt") -> pd.DataFrame:
    df = pd.read_csv(cbsa_path, sep="\t", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
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
    earth_radius_km = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return earth_radius_km * 2 * np.arcsin(np.sqrt(a))


def build_cbsa_weather_features(
    cbsa_path: str = "data/raw/2023_Gaz_cbsa_national.txt",
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
    force_rebuild_stations: bool = False,
) -> pd.DataFrame:
    cbsa_df = load_cbsa_centroids(cbsa_path)
    station_df = load_station_climate_normals(
        base_url=base_url,
        cache_dir=cache_dir,
        max_files=max_files,
        force_rebuild=force_rebuild_stations,
    ).dropna(subset=["station_lat", "station_lon"])

    cbsa_lat = cbsa_df["centroid_lat"].to_numpy()[:, np.newaxis]
    cbsa_lon = cbsa_df["centroid_lon"].to_numpy()[:, np.newaxis]
    station_lat = station_df["station_lat"].to_numpy()[np.newaxis, :]
    station_lon = station_df["station_lon"].to_numpy()[np.newaxis, :]

    distances = haversine_distance_km(cbsa_lat, cbsa_lon, station_lat, station_lon)
    nearest_station_index = distances.argmin(axis=1)
    nearest_station_distance = distances[np.arange(len(cbsa_df)), nearest_station_index]
    nearest_station_df = station_df.iloc[nearest_station_index].reset_index(drop=True)

    weather_df = cbsa_df.reset_index(drop=True).copy()
    weather_df["station_id"] = nearest_station_df["station_id"].to_numpy()
    weather_df["station_name"] = nearest_station_df["station_name"].to_numpy()
    weather_df["station_distance_km"] = nearest_station_distance
    weather_df["avg_annual_temp"] = nearest_station_df["avg_annual_temp"].to_numpy()
    weather_df["jan_avg_temp"] = nearest_station_df["jan_avg_temp"].to_numpy()
    weather_df["jul_avg_temp"] = nearest_station_df["jul_avg_temp"].to_numpy()
    weather_df["annual_precipitation"] = nearest_station_df["annual_precipitation"].to_numpy()
    weather_df["annual_snowfall"] = nearest_station_df["annual_snowfall"].to_numpy()
    weather_df["n_months_available"] = nearest_station_df["n_months_available"].to_numpy()
    weather_df["has_complete_year"] = nearest_station_df["has_complete_year"].to_numpy()
    weather_df["temp_seasonality"] = weather_df["jul_avg_temp"] - weather_df["jan_avg_temp"]
    weather_df["snow_binary"] = (weather_df["annual_snowfall"] > 0).astype(int)

    return weather_df[
        [
            "cbsa_code",
            "cbsa_name",
            "cbsa_type",
            "centroid_lat",
            "centroid_lon",
            "station_id",
            "station_name",
            "station_distance_km",
            "avg_annual_temp",
            "jan_avg_temp",
            "jul_avg_temp",
            "annual_precipitation",
            "annual_snowfall",
            "n_months_available",
            "has_complete_year",
            "temp_seasonality",
            "snow_binary",
        ]
    ]


def load_cbsa_weather_dataset(
    cbsa_path: str = "data/raw/2023_Gaz_cbsa_national.txt",
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    return build_cbsa_weather_features(
        cbsa_path=cbsa_path,
        base_url=base_url,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    weather_df = load_cbsa_weather_dataset()
    logger.info("Loaded CBSA weather dataset with %d rows and %d columns.", len(weather_df), len(weather_df.columns))
    print(weather_df.head())
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
    from weather_data_loader import load_cbsa_weather_dataset

    weather_df = load_cbsa_weather_dataset()
    print(weather_df.head())
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Simple default configuration; caller can override.
    logging.basicConfig(level=logging.INFO)


NOAA_DEFAULT_BASE_URL = "https://www.ncei.noaa.gov/data/normals-monthly/2006-2020/access/"
DEFAULT_CACHE_DIR = Path("data/raw/weather/noaa_monthly_normals")
STATION_SUMMARY_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020.parquet")
STATION_METADATA_PATH = Path("data/processed/noaa_station_climate_normals_2006_2020_metadata.json")


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
    name = name.strip()
    name = name.lower()
    # Replace non-alphanumeric characters with underscore.
    name = re.sub(r"[^0-9a-z]+", "_", name)
    # Collapse multiple underscores and strip leading/trailing.
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def standardize_noaa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize NOAA station CSV columns to lowercase snake_case.

    Examples of mappings:
        STATION         -> station
        LATITUDE        -> latitude
        LONGITUDE       -> longitude
        NAME            -> name
        month           -> month
        MLY-TAVG-NORMAL -> mly_tavg_normal
        MLY-PRCP-NORMAL -> mly_prcp_normal
        MLY-SNOW-NORMAL -> mly_snow_normal
    """
    df = df.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]
    return df


def list_noaa_station_csvs(base_url: str = NOAA_DEFAULT_BASE_URL) -> List[str]:
    """
    Retrieve the NOAA directory HTML and parse all station CSV URLs.

    Parameters
    ----------
    base_url:
        Base NOAA directory URL for 2006–2020 normals.

    Returns
    -------
    List of full URLs for per-station CSV files.
    """
    session = create_session()
    resp: Response = session.get(base_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch NOAA directory listing: HTTP {resp.status_code}")

    html = resp.text
    # Simple anchor tag parsing; NOAA directory is a static index page.
    csv_files = re.findall(r'href=["\']([^"\']+?\.csv)["\']', html, flags=re.IGNORECASE)
    urls = [urljoin(base_url, f) for f in csv_files]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: List[str] = []
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
    """
    Download one station CSV to a local cache directory.

    If the file already exists locally, reuse it instead of downloading again.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(url).name
    dest = cache_dir / filename
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
    """
    Load one station CSV and return monthly rows for that station.

    The returned DataFrame has standardized, typed columns and keeps only
    the subset needed for downstream climate summaries.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read station CSV %s: %s", path, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df = standardize_noaa_columns(df)

    # Ensure presence of core identifier columns.
    required_id_cols = ["station", "latitude", "longitude", "name", "month"]
    for col in required_id_cols:
        if col not in df.columns:
            logger.warning("Station CSV %s missing required column '%s'; skipping", path, col)
            return pd.DataFrame()

    # Keep only the needed columns; others are dropped to keep things tidy.
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

    # Cast dtypes.
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    for col in ("mly_tavg_normal", "mly_prcp_normal", "mly_snow_normal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid month values.
    df = df[df["month"].between(1, 12, inclusive="both")]
    if df.empty:
        return df

    return df


def summarize_station_normals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize one-station monthly normals into a single-row station summary.

    Output columns:
        - station_id
        - station_name
        - station_lat
        - station_lon
        - avg_annual_temp
        - jan_avg_temp
        - jul_avg_temp
        - annual_precipitation
        - annual_snowfall
        - n_months_available
        - has_complete_year

    Rules:
        - Require at least some non-null MLY-TAVG-NORMAL and MLY-PRCP-NORMAL
          values or return an empty DataFrame.
        - If fewer than 12 distinct months exist, still return a row but
          has_complete_year is False.
        - If snowfall column is missing or entirely NaN, annual_snowfall is NaN.
    """
    if df.empty:
        return pd.DataFrame()

    # Defensive copy and ensure we are looking at a single station.
    df = df.copy()
    if "station" not in df.columns:
        return pd.DataFrame()

    temp = df.get("mly_tavg_normal")
    prcp = df.get("mly_prcp_normal")

    if temp is None or prcp is None:
        # Missing core climate variables – do not create a summary row.
        return pd.DataFrame()

    temp = pd.to_numeric(temp, errors="coerce")
    prcp = pd.to_numeric(prcp, errors="coerce")

    if not temp.notna().any() or not prcp.notna().any():
        return pd.DataFrame()

    df["month"] = df["month"].astype("Int64")
    months_present = df["month"].dropna().unique()
    n_months_available = int(len(months_present))
    has_complete_year = n_months_available == 12

    # Day-weighted annual average temperature.
    month_days = df["month"].map(MONTH_DAY_COUNTS)
    mask_temp = temp.notna() & month_days.notna()
    if mask_temp.any():
        avg_annual_temp = float((temp[mask_temp] * month_days[mask_temp]).sum() / month_days[mask_temp].sum())
    else:
        avg_annual_temp = math.nan

    # January and July average temperatures.
    jan_mask = df["month"] == 1
    jul_mask = df["month"] == 7
    jan_vals = temp[jan_mask].dropna()
    jul_vals = temp[jul_mask].dropna()
    jan_avg_temp = float(jan_vals.iloc[0]) if not jan_vals.empty else math.nan
    jul_avg_temp = float(jul_vals.iloc[0]) if not jul_vals.empty else math.nan

    # Annual precipitation and snowfall.
    annual_precipitation = float(prcp.dropna().sum()) if prcp.notna().any() else math.nan

    snow = df.get("mly_snow_normal")
    if snow is None:
        annual_snowfall = math.nan
    else:
        snow = pd.to_numeric(snow, errors="coerce")
        annual_snowfall = float(snow.dropna().sum()) if snow.notna().any() else math.nan

    # Basic identifiers: station id, name, coordinates.
    station_id = str(df["station"].iloc[0])
    station_name = str(df.get("name", pd.Series([""])).iloc[0])
    station_lat = float(df.get("latitude", pd.Series([math.nan])).iloc[0])
    station_lon = float(df.get("longitude", pd.Series([math.nan])).iloc[0])

    out = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "station_name": station_name,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "avg_annual_temp": avg_annual_temp,
                "jan_avg_temp": jan_avg_temp,
                "jul_avg_temp": jul_avg_temp,
                "annual_precipitation": annual_precipitation,
                "annual_snowfall": annual_snowfall,
                "n_months_available": n_months_available,
                "has_complete_year": has_complete_year,
            }
        ]
    )
    return out


def build_station_climate_normals_from_noaa(
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Build station-level climate normals from NOAA monthly normals.

    This function:
        - Lists all NOAA station CSVs.
        - Downloads and parses them with retry and local caching.
        - Summarizes each station to one row.
        - Concatenates all station summaries into one DataFrame.
        - Saves the combined summary to a Parquet file plus a small metadata JSON.
    """
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
            # Either incomplete required variables or otherwise unusable.
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
        stations_df = pd.DataFrame(
            columns=[
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
            ]
        )

    # Persist results for reproducibility.
    stations_df.to_parquet(STATION_SUMMARY_PATH, index=False)

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
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
    """
    Load or build the combined station climate normals dataset.

    If the Parquet summary already exists and force_rebuild=False, this simply
    reads the file. Otherwise, it calls build_station_climate_normals_from_noaa().
    """
    if STATION_SUMMARY_PATH.exists() and not force_rebuild:
        return pd.read_parquet(STATION_SUMMARY_PATH)
    return build_station_climate_normals_from_noaa(base_url=base_url, cache_dir=cache_dir, max_files=max_files)


def load_cbsa_centroids(cbsa_path: str | Path = "data/raw/census/2023_Gaz_cbsa_national.txt") -> pd.DataFrame:
    """
    Load CBSA centroids from the Census Gazetteer file.

    Returns DataFrame with:
        - cbsa_code
        - cbsa_name
        - cbsa_type
        - centroid_lat
        - centroid_lon
    """
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

    # Normalize identifier and coordinate types.
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
    """
    Compute great-circle distance between two sets of points using the haversine formula.

    Inputs are in degrees; output is in kilometers. Arrays are broadcast as needed.
    """
    # Earth radius in kilometers.
    R = 6371.0088

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def attach_nearest_station_to_cbsa(
    cbsa_df: pd.DataFrame,
    station_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each CBSA centroid to its nearest station and attach climate features.

    Returns a tidy CBSA-level DataFrame with:
        - cbsa_code
        - cbsa_name
        - cbsa_type
        - centroid_lat
        - centroid_lon
        - station_id
        - station_name
        - station_distance_km
        - avg_annual_temp
        - jan_avg_temp
        - jul_avg_temp
        - annual_precipitation
        - annual_snowfall
        - n_months_available
        - has_complete_year
        - temp_seasonality
        - snow_binary
    """
    cbsa_df = cbsa_df.copy()
    station_df = station_df.copy()

    # Drop stations with missing coordinates.
    station_df = station_df.dropna(subset=["station_lat", "station_lon"])
    if station_df.empty:
        raise ValueError("No station climate records with valid coordinates are available.")

    cbsa_coords = cbsa_df[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float)
    station_coords = station_df[["station_lat", "station_lon"]].to_numpy(dtype=float)

    # Compute pairwise distances via broadcasting: (n_cbsa, n_station)
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

    # Attach climate variables and completeness flags.
    for col in [
        "avg_annual_temp",
        "jan_avg_temp",
        "jul_avg_temp",
        "annual_precipitation",
        "annual_snowfall",
        "n_months_available",
        "has_complete_year",
    ]:
        cbsa_df[col] = nearest_stations[col].values

    # Optional derived fields.
    cbsa_df["temp_seasonality"] = cbsa_df["jul_avg_temp"] - cbsa_df["jan_avg_temp"]
    cbsa_df["snow_binary"] = np.where(cbsa_df["annual_snowfall"] > 0, 1, 0)

    # Reorder columns to match specification.
    cols_order = [
        "cbsa_code",
        "cbsa_name",
        "cbsa_type",
        "centroid_lat",
        "centroid_lon",
        "station_id",
        "station_name",
        "station_distance_km",
        "avg_annual_temp",
        "jan_avg_temp",
        "jul_avg_temp",
        "annual_precipitation",
        "annual_snowfall",
        "n_months_available",
        "has_complete_year",
        "temp_seasonality",
        "snow_binary",
    ]
    cbsa_df = cbsa_df[cols_order]
    return cbsa_df


def build_cbsa_weather_features(
    cbsa_path: str = "data/raw/2023_Gaz_cbsa_national.txt",
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_files: int | None = None,
    force_rebuild_stations: bool = False,
) -> pd.DataFrame:
    """
    High-level wrapper to build CBSA-level weather features.

    Steps:
        1. Load CBSA centroids from Gazetteer file.
        2. Build or load station climate summaries from NOAA monthly normals.
        3. Map each CBSA to its nearest station.
        4. Join station climate features to CBSA.
    """
    cbsa_df = load_cbsa_centroids(cbsa_path)
    station_df = load_station_climate_normals(
        base_url=base_url,
        cache_dir=cache_dir,
        max_files=max_files,
        force_rebuild=force_rebuild_stations,
    )
    cbsa_weather_df = attach_nearest_station_to_cbsa(cbsa_df, station_df)
    return cbsa_weather_df


def load_cbsa_weather_dataset(
    cbsa_path: str = "data/raw/2023_Gaz_cbsa_national.txt",
    base_url: str = NOAA_DEFAULT_BASE_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Notebook-friendly convenience function: build CBSA-level weather dataset.

    Parameters
    ----------
    cbsa_path:
        Path to Census Gazetteer CBSA file (tab-separated).
    base_url:
        Base NOAA directory for 2006–2020 monthly normals.
    cache_dir:
        Local directory for caching downloaded NOAA station CSVs.

    Returns
    -------
    DataFrame with one row per CBSA and the weather/climate columns described
    in attach_nearest_station_to_cbsa().

    Example
    -------
        from weather_data_loader import load_cbsa_weather_dataset

        weather_df = load_cbsa_weather_dataset()
        weather_df.head()
    """
    return build_cbsa_weather_features(
        cbsa_path=cbsa_path,
        base_url=base_url,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    weather_df = load_cbsa_weather_dataset()
    logger.info(
        "Loaded CBSA weather dataset with %d rows and %d columns.",
        len(weather_df),
        len(weather_df.columns),
    )
    print(weather_df.head())

