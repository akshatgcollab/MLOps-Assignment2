import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feast import FeatureStore

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from codecarbon import EmissionsTracker

FEATURES = {
    "v1": [
        "fv_athletes_v1:age",
        "fv_athletes_v1:weight",
        "fv_athletes_v1:height",
        "fv_athletes_v1:howlong",
        "fv_athletes_v1:gender",
        "fv_athletes_v1:region",
        "fv_athletes_v1:eat",
        "fv_athletes_v1:background",
        "fv_athletes_v1:experience",
        "fv_athletes_v1:schedule",
        "fv_athletes_v1:total_lift",
    ],
    "v2": [
        "fv_athletes_v2:age",
        "fv_athletes_v2:weight",
        "fv_athletes_v2:height",
        "fv_athletes_v2:howlong",
        "fv_athletes_v2:gender",
        "fv_athletes_v2:region",
        "fv_athletes_v2:eat",
        "fv_athletes_v2:background",
        "fv_athletes_v2:experience",
        "fv_athletes_v2:schedule",
        "fv_athletes_v2:bmi",
        "fv_athletes_v2:weight_height_ratio",
        "fv_athletes_v2:bmi_age",
        "fv_athletes_v2:total_lift",
    ],
}

PARAMS_LIST = [
    {"n_estimators": 150, "max_depth": 8,  "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 1},
]

def build_pipeline(params, numeric_cols, categorical_cols):
    numeric_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("preprocess", pre), ("model", model)])

def run_one(store, entity_df, feature_version, feature_refs, params, emissions_outdir):
    train_df = store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()
    train_df = train_df.dropna(subset=["total_lift"]).copy()

    y = train_df["total_lift"].astype(float)
    X = train_df.drop(columns=["total_lift", "event_timestamp"])

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_name = f"{feature_version}_rf_n{params['n_estimators']}_d{params['max_depth']}_leaf{params['min_samples_leaf']}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_params(params)
        mlflow.log_param("n_rows", len(train_df))
        mlflow.log_param("n_features_raw", X.shape[1])
        mlflow.log_param("n_numeric", len(numeric_cols))
        mlflow.log_param("n_categorical", len(categorical_cols))

        tracker = EmissionsTracker(project_name=run_name, output_dir=emissions_outdir)
        tracker.start()

        pipe = build_pipeline(params, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        emissions = tracker.stop()  # kg CO2
        carbon = float(emissions) if emissions else 0.0

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("carbon_kg", carbon)

        # Pred vs actual
        plt.figure()
        plt.scatter(y_test, preds, alpha=0.5)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(lims, lims)
        plt.xlabel("Actual total_lift")
        plt.ylabel("Predicted total_lift")
        plt.title(f"Pred vs Actual – {run_name}")
        pva_path = "pred_vs_actual.png"
        plt.savefig(pva_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(pva_path)

        # Residuals
        residuals = y_test - preds
        plt.figure()
        plt.scatter(preds, residuals, alpha=0.5)
        plt.axhline(0)
        plt.xlabel("Predicted total_lift")
        plt.ylabel("Residual (actual - pred)")
        plt.title(f"Residuals – {run_name}")
        res_path = "residuals.png"
        plt.savefig(res_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(res_path)

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        return {"run_name": run_name, "feature_version": feature_version, "rmse": rmse, "mae": mae, "r2": r2, "carbon_kg": carbon}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="src/feast_repo", help="Feast repo path (contains feature_store.yaml)")
    ap.add_argument("--parquet", default="data/athletes.parquet")
    ap.add_argument("--mlruns", default="mlruns")
    ap.add_argument("--experiment", default="Feast_Athletes_TotalLift_Regression")
    args = ap.parse_args()

    os.makedirs(args.mlruns, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{os.path.abspath(args.mlruns)}")
    mlflow.set_experiment(args.experiment)

    store = FeatureStore(repo_path=args.repo)

    base = pd.read_parquet(args.parquet)
    entity_df = base[["athlete_entity_id", "event_timestamp"]].copy()

    results = []
    for fv, refs in FEATURES.items():
        for params in PARAMS_LIST:
            results.append(run_one(store, entity_df, fv, refs, params, emissions_outdir=args.mlruns))

    results_df = pd.DataFrame(results).sort_values(["r2", "rmse"], ascending=[False, True])
    out_csv = os.path.join(args.mlruns, "experiment_summary.csv")
    results_df.to_csv(out_csv, index=False)
    print("Saved summary:", out_csv)
    print(results_df)

if __name__ == "__main__":
    main()
