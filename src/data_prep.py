import argparse
import numpy as np
import pandas as pd

SENTINELS = {8388607, 9999999, 99999, 9999}

import pandas as pd
import numpy as np

SENTINELS = {8388607, 9999999, 99999, 9999}

def clean_athletes_relaxed(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    # 1) Strip strings + normalize missing tokens
    for col in d.columns:
        if d[col].dtype == object:
            d[col] = d[col].astype(str).str.strip()
            d.loc[d[col].isin(["", "nan", "None", "NA", "N/A", "--"]), col] = np.nan

    # 2) Numeric coercion + replace sentinels + drop negatives
    num_cols = ["age","height","weight","howlong","deadlift","candj","snatch","backsq"]
    for c in num_cols:
        if c in d.columns:
            s = pd.to_numeric(d[c], errors="coerce")
            s = s.replace(list(SENTINELS), np.nan)
            s = s.mask(s < 0, np.nan)
            d[c] = s

    # 3) Light sanity bounds (don’t be too aggressive)
    # Assumption: height in inches, weight in lbs (common in this dataset)
    d = d[(d["age"].between(18, 80)) | d["age"].isna()]
    d = d[(d["height"].between(48, 90)) | d["height"].isna()]
    d = d[(d["weight"].between(50, 400)) | d["weight"].isna()]

    # Lift bounds: keep plausible, but don't drop NaNs yet (target will enforce)
    lift_bounds = {"deadlift": 1200, "candj": 600, "snatch": 500, "backsq": 1200}
    for c, hi in lift_bounds.items():
        d = d[(d[c].between(1, hi)) | d[c].isna()]

    # 4) Target: total_lift requires ALL 4 lifts present (min_count=4)
    lift_cols = ["deadlift", "candj", "snatch", "backsq"]
    d["total_lift"] = d[lift_cols].sum(axis=1, min_count=4)

    # 5) Now enforce target exists + core demographics exist
    d = d.dropna(subset=["total_lift", "age", "height", "weight", "gender"])

    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to athletes.csv")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--timestamp", default="now", help="'now' or ISO timestamp; written to event_timestamp")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    df = clean_athletes_relaxed(raw)

    # Entity key + timestamp (required by Feast)
    df = df.copy()
    df["athlete_entity_id"] = np.arange(len(df), dtype=int)
    if args.timestamp == "now":
        ts = pd.Timestamp.now(tz="UTC")
    else:
        ts = pd.Timestamp(args.timestamp).tz_convert("UTC") if pd.Timestamp(args.timestamp).tzinfo else pd.Timestamp(args.timestamp, tz="UTC")
    df["event_timestamp"] = ts

    # Engineered features (v2)
    df["bmi"] = 703.0 * df["weight"] / (df["height"].replace(0, np.nan) ** 2)
    df["weight_height_ratio"] = df["weight"] / df["height"].replace(0, np.nan)
    df["bmi_age"] = df["bmi"] * df["age"]

    df.to_parquet(args.out, index=False)
    print(f"Wrote parquet: {args.out}  shape={df.shape}")

if __name__ == "__main__":
    main()
