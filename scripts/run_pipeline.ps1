# Run from repository root:  .\scripts\run_pipeline.ps1
# Default skips weather (uses data/processed/Weather_Data.csv). Use -IncludeWeather for a full NOAA rebuild.

param(
    [switch]$IncludeWeather
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Step {
    param([string]$Message, [scriptblock]$Action)
    Write-Host "==> $Message" -ForegroundColor Cyan
    & $Action
}

Invoke-Step "Census (API + gazetteer)" { python -m src.census_data_loader }
Invoke-Step "Crime" { python -m src.crime_data_loader }
Invoke-Step "PLACES (GeoPandas + shapefile)" { python -m src.places_data_loader }
Invoke-Step "Walkability" { python -m src.walkability_data_loader }

if ($IncludeWeather) {
    Invoke-Step "Weather (NOAA download — very slow)" { python -m src.weather_data_loader }
} else {
    Write-Host "==> Skipping weather loader (use committed data/processed/Weather_Data.csv). Pass -IncludeWeather to rebuild." -ForegroundColor Yellow
    if (-not (Test-Path "data/processed/Weather_Data.csv")) {
        Write-Error "Missing data/processed/Weather_Data.csv. Add the file or run with -IncludeWeather."
    }
}

Invoke-Step "Final merged + enriched dataset" { python -m src.final_dataset_loader }

Write-Host "Done. App input: data/final/Final_Enriched_Dataset.csv" -ForegroundColor Green
