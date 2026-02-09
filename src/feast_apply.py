import argparse
import pandas as pd
from feast import FeatureStore
from feast_repo.features import athlete, fv_athletes_v1, fv_athletes_v2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="src/feast_repo", help="Path to Feast repo (contains feature_store.yaml)")
    ap.add_argument("--materialize-from", default="2020-01-01T00:00:00Z")
    ap.add_argument("--materialize-to", default="now")
    ap.add_argument("--parquet", default="data/athletes.parquet", help="Offline parquet location (for sanity check)")
    args = ap.parse_args()

    # sanity check parquet exists
    _ = pd.read_parquet(args.parquet)

    store = FeatureStore(repo_path=args.repo)
    store.apply([athlete, fv_athletes_v1, fv_athletes_v2])

    start = pd.Timestamp(args.materialize_from)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    end = pd.Timestamp.now(tz="UTC") if args.materialize_to == "now" else pd.Timestamp(args.materialize_to)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    store.materialize(start, end)
    print("✅ Feast applied + materialized successfully.")

if __name__ == "__main__":
    main()
