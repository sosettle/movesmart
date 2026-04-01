#!/usr/bin/env bash
# Run from repository root: bash scripts/run_pipeline.sh
# With weather (very slow): INCLUDE_WEATHER=1 bash scripts/run_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

step() {
  local msg=$1
  shift
  echo "==> $msg"
  "$@"
}

step "Census (API + gazetteer)" python -m src.census_data_loader
step "Crime" python -m src.crime_data_loader
step "PLACES (GeoPandas + shapefile)" python -m src.places_data_loader
step "Walkability" python -m src.walkability_data_loader

if [[ "${INCLUDE_WEATHER:-}" == "1" ]]; then
  step "Weather (NOAA download — very slow)" python -m src.weather_data_loader
else
  echo "==> Skipping weather loader (use committed data/processed/Weather_Data.csv). Set INCLUDE_WEATHER=1 to rebuild."
  if [[ ! -f data/processed/Weather_Data.csv ]]; then
    echo "error: missing data/processed/Weather_Data.csv" >&2
    exit 1
  fi
fi

step "Final merged + enriched dataset" python -m src.final_dataset_loader
echo "Done. App input: data/final/Final_Enriched_Dataset.csv"
