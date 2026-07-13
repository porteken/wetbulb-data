#!/usr/bin/env bash
set -e

echo "Setting up container filesystem..."
mkdir -p output_tiles pet_data_csv wetbulb_data_csv analytics_data_csv

echo "Generating cities and tiles..."
python cities.py

echo "Starting ${WORKER_SCRIPT:-google_era5.py} worker..."
exec python "${WORKER_SCRIPT:-google_era5.py}" "$@"