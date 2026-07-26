#!/usr/bin/env bash
set -e

echo "Setting up container filesystem..."
mkdir -p output_tiles wetbulb_data_csv analytics_data_csv

echo "Starting ${WORKER_SCRIPT:-nldas.py} worker..."
exec python "${WORKER_SCRIPT:-nldas.py}" "$@"
