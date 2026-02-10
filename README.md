# Athletes Feature Store (Feast) + MLflow Experiments

This repo turns the original notebook into a reproducible project:
- **Data cleaning + feature engineering** -> Parquet offline store
- **Feast feature store** (local registry + file offline store + sqlite online store)
- **Model training** using historical feature retrieval from Feast
- **Experiment tracking** with MLflow + optional CodeCarbon emissions logging

## Repo layout
- `src/data_prep.py` – cleans `data/athletes.csv` and writes `data/athletes.parquet`
- `src/feast_repo/feature_store.yaml` – Feast config (registry + offline/online stores)
- `src/feast_repo/features.py` – entities + feature views (v1 and v2)
- `src/feast_apply.py` – applies feature definitions and materializes to online store
- `src/train.py` – pulls historical features, trains RF models, logs to MLflow
- `notebooks/` – original notebook (kept for reference)

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Put the raw CSV
Place your raw dataset at:
```
data/athletes.csv
```

### 3) Build offline store parquet
```bash
python src/data_prep.py --csv data/athletes.csv --out data/athletes.parquet
```

### 4) Apply + materialize Feast
```bash
python src/feast_apply.py --repo src/feast_repo --parquet data/athletes.parquet
```

### 5) Train and log MLflow runs
```bash
python src/train.py --repo src/feast_repo --parquet data/athletes.parquet --mlruns mlruns
```

