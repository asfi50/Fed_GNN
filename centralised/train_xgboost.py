"""
Train an XGBoost classifier on the balanced, feature-engineered NF-ToN-IoT
dataset and check whether a centralised model can separate the classes that
were confused in the federated FedGATSage results (ddos/injection/password/
scanning/xss — see findings.md / paper.md confusion matrices).

Usage:
    python train_xgboost.py --data data/nfton_balanced.csv

Mac note: only run this on a tiny --data file produced with preprocess.py
--nrows for a smoke test. The real run happens on Kaggle (run_kaggle.ipynb).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from comet_utils import evaluate_and_log, start_experiment

TARGET_COL = "Attack"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Balanced CSV from preprocess.py")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-comet", action="store_true", help="Skip Comet logging (local smoke tests)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    exp = None if args.no_comet else start_experiment(
        "xgboost-nfton-centralised", tags=["xgboost", "nfton", "centralised"]
    )

    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        random_state=args.seed,
    )
    if exp is not None:
        exp.log_parameters(params)
        exp.log_parameters({
            "n_train": len(X_train), "n_test": len(X_test),
            "n_features": X.shape[1], "classes": class_names,
        })

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    os.makedirs(args.out_dir, exist_ok=True)
    evaluate_and_log(exp, "xgboost", y_test, y_pred, class_names, args.out_dir)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances_path = os.path.join(args.out_dir, "xgboost_feature_importances.csv")
    importances.to_csv(importances_path, header=["importance"])
    if exp is not None:
        exp.log_asset(importances_path)

    model_path = os.path.join(args.out_dir, "xgboost_model.json")
    model.save_model(model_path)
    if exp is not None:
        exp.log_model("xgboost", model_path)
        exp.end()


if __name__ == "__main__":
    main()
