# MoveSmart

Streamlit app for exploring U.S. CBSA (metro) recommendations using a merged census, health, crime, walkability, and weather panel. The **modeling dataset** is built by Python loaders under `src/`, then merged in `src/final_dataset_loader.py`.

## Main application

| Entry point | Role |
|-------------|------|
| **`app.py`** | Streamlit UI (`streamlit run app.py`). Reads **`data/final/Final_Enriched_Dataset.csv`**. |

Other Python modules (`src/recommender.py`, `src/visualizations.py`) are imported by the app. Clustering used in the final dataset lives in **`models/cluster_model.py`**.

---

## Repository layout

```
movesmart/
├── app.py                      # Streamlit app (main UI)
├── data/
│   ├── raw/                    # Source files (not in git); obtain locally (see Step 0)
│   ├── processed/              # Per-source CBSA tables (loader outputs)
│   └── final/                  # Final_Base_Dataset.csv, Final_Enriched_Dataset.csv
├── models/
│   └── cluster_model.py        # KMeans / PCA; used by final_dataset_loader
├── scripts/
│   ├── run_pipeline.ps1        # Windows: orchestrates loaders (+ optional weather)
│   └── run_pipeline.sh         # POSIX: same
├── src/
│   ├── census_data_loader.py
│   ├── crime_data_loader.py
│   ├── places_data_loader.py
│   ├── walkability_data_loader.py
│   ├── weather_data_loader.py  # slow; normally skipped (use Weather_Data.csv)
│   ├── final_dataset_loader.py # merges processed → final + scores + clusters
│   ├── standardize_scores.py   # score columns (imported by final_dataset_loader)
│   ├── recommender.py
│   ├── visualizations.py
│   └── wiki_text_loader.py     # optional; AWS Bedrock; not in core pipeline
├── requirements.txt
└── Makefile
```

---

## Setup

**Python 3.10+** recommended (uses `list[str]` / modern typing in several modules).

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux / Git Bash:**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional Census API key (better rate limits): set `CENSUS_API_KEY` in your shell (for example `$env:CENSUS_API_KEY='…'` in PowerShell) before running the census loader. The loader also runs without a key.

**GeoPandas:** On some Windows setups, `pip install geopandas` is enough; if install fails, use [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or a Conda environment with `geopandas` from conda-forge.

---

## Data pipeline (reproducible order)

All commands assume the **repository root** as the current working directory.

### Step 0 — Raw inputs (`data/raw/`)

| Loader | Required paths (defaults in code) |
|--------|-------------------------------------|
| **Census** | `data/raw/2023_Gaz_cbsa_national.txt` (Census CBSA gazetteer). ACS tables are fetched from **api.census.gov** (optional `CENSUS_API_KEY`). |
| **Crime** | `data/raw/FBI_Crime_Data_By_City_with_Counties.csv`, `data/raw/ZIP_CBSA_122023.csv` |
| **PLACES** | `data/raw/PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260314.csv` (or your tract file with the same column expectations), **`data/raw/shapefiles/tl_2023_us_cbsa.shp`** plus sidecars (`.dbf`, `.shx`, `.prj`, …). |
| **Walkability** | `data/raw/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv` |
| **Weather** | *Skipped for normal reproduction* — use committed **`data/processed/Weather_Data.csv`**. Full rebuild uses the gazetteer + thousands of NOAA downloads (many hours). |

### Step 1 — Build processed CBSA tables

Run in this order (census first is conventional; crime/places/walkability only depend on raw files, not on each other):

```powershell
# Windows PowerShell (from repo root)
python -m src.census_data_loader
python -m src.crime_data_loader
python -m src.places_data_loader
python -m src.walkability_data_loader
```

**Skip weather** and keep using the repo’s `data/processed/Weather_Data.csv`. Do **not** run `weather_data_loader` unless you intend to wait for a full NOAA pull.

If you must rebuild weather:

```powershell
python -m src.weather_data_loader
```

That writes **`data/processed/Weather_Data.csv`** (and uses `data/raw/weather/noaa_monthly_normals/` as a cache).

### Step 2 — Final dataset (merge + imputation + scores + clusters)

```powershell
python -m src.final_dataset_loader
```

**Outputs:**

| File | Description |
|------|-------------|
| `data/processed/Census_Data.csv` | Census loader |
| `data/processed/Crime_Data.csv` | Crime loader |
| `data/processed/Places_Data.csv` | PLACES loader |
| `data/processed/Walkability_Data.csv` | Walkability loader |
| `data/processed/Weather_Data.csv` | Weather loader (or committed copy) |
| `data/final/Final_Base_Dataset.csv` | Merged + imputed base |
| `data/final/Final_Enriched_Dataset.csv` | Base + feature/composite scores + cluster columns (**app input**) |

---

## One-shot scripts

**Windows:**

```powershell
.\scripts\run_pipeline.ps1
# Full weather rebuild (slow):
.\scripts\run_pipeline.ps1 -IncludeWeather
```

**Git Bash / WSL / macOS / Linux:**

```bash
bash scripts/run_pipeline.sh
INCLUDE_WEATHER=1 bash scripts/run_pipeline.sh   # slow
```

**Make** (if `make` is available):

```bash
make install
make pipeline-no-weather   # recommended
make pipeline              # includes weather
make app                   # streamlit run app.py
```

---

## Run the Streamlit app

```powershell
streamlit run app.py
```

Ensure **`data/final/Final_Enriched_Dataset.csv`** exists (run `final_dataset_loader` after the processed inputs exist).

---

## Dependencies (by concern)

| Area | Packages |
|------|----------|
| App | `streamlit`, `folium`, `streamlit-folium`, `plotly`, `pandas`, `numpy` |
| Census / crime / walkability / weather HTTP | `requests`, `urllib3` |
| PLACES spatial join | `geopandas` (+ GDAL stack via pip or conda) |
| Clustering + scaling in `models/cluster_model.py` | `scikit-learn` |
| Optional wiki + Bedrock | `boto3` (`src/wiki_text_loader.py`) |

---

## License / data provenance

Respect terms of use for Census API, CDC PLACES, FBI crime statistics, EPA Smart Location Database, and NOAA normals when redistributing derived files.
