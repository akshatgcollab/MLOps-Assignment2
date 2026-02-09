#!/usr/bin/env bash
set -euo pipefail
python src/data_prep.py --csv data/athletes.csv --out data/athletes.parquet
python src/feast_apply.py --repo src/feast_repo --parquet data/athletes.parquet
python src/train.py --repo src/feast_repo --parquet data/athletes.parquet --mlruns mlruns
echo "Run: mlflow ui --backend-store-uri mlruns --port 5000"
