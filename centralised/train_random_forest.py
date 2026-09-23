"""
Train a Random Forest classifier on the balanced, feature-engineered
NF-ToN-IoT dataset and check whether a centralised model can separate the
classes that were confused in the federated FedGATSage results
(ddos/injection/password/scanning/xss — see findings.md / paper.md).

Usage:
    python train_random_forest.py --data data/nfton_balanced.csv

Mac note: only run this on a tiny --data file produced with preprocess.py
--nrows for a smoke test. The real run happens on Kaggle (run_kaggle.ipynb).
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from comet_utils import evaluate_and_log, start_experiment

TARGET_COL = "Attack"
SPLIT_COL = "split"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Balanced CSV from preprocess.py")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-comet", action="store_true", help="Skip Comet logging (local smoke tests)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    class_names = le.classes_.tolist()
    X = df.drop(columns=[TARGET_COL, SPLIT_COL]).select_dtypes(include=[np.number])

    # preprocess.py splits before balancing, so the test set keeps the natural
    # class distribution. Re-splitting here would undo that.
    is_train = (df[SPLIT_COL] == "train").to_numpy()
    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]

    exp = None if args.no_comet else start_experiment(
        "random-forest-nfton-centralised", tags=["random-forest", "nfton", "centralised"]
    )

    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        class_weight="balanced_subsample",
    )
    if exp is not None:
        exp.log_parameters(params)
        exp.log_parameters({
            "n_train": len(X_train), "n_test": len(X_test),
            "n_features": X.shape[1], "classes": class_names,
        })

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    os.makedirs(args.out_dir, exist_ok=True)
    evaluate_and_log(exp, "random_forest", y_test, y_pred, class_names, args.out_dir)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances_path = os.path.join(args.out_dir, "random_forest_feature_importances.csv")
    importances.to_csv(importances_path, header=["importance"])
    if exp is not None:
        exp.log_asset(importances_path)

    model_path = os.path.join(args.out_dir, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    if exp is not None:
        exp.log_model("random_forest", model_path)
        exp.end()


if __name__ == "__main__":
    main()
