# Run from repo root. POSIX shell (Git Bash, WSL, macOS, Linux).

.PHONY: install pipeline pipeline-no-weather app

install:
	python -m pip install -r requirements.txt

# Full rebuild of processed inputs + final enriched dataset (includes weather; very slow).
pipeline:
	python -m src.census_data_loader
	python -m src.crime_data_loader
	python -m src.places_data_loader
	python -m src.walkability_data_loader
	python -m src.weather_data_loader
	python -m src.final_dataset_loader

# Recommended: use committed data/processed/Weather_Data.csv; skips NOAA download.
pipeline-no-weather:
	python -m src.census_data_loader
	python -m src.crime_data_loader
	python -m src.places_data_loader
	python -m src.walkability_data_loader
	python -m src.final_dataset_loader

app:
	streamlit run app.py
