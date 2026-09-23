"""Shared Comet ML logging + confusion-matrix helpers for the centralised
NF-ToN-IoT XGBoost / Random Forest experiments."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Same API key/pattern already used in experiments/fedgatsage_experiment.py.
COMET_API_KEY = os.environ.get("COMET_API_KEY", "emuhTVn5AAwEm9ALtwsL4SkUo")
COMET_PROJECT = "fedgatsage-centralised"


def start_experiment(name: str, tags=None):
    import comet_ml

    comet_ml.login(api_key=COMET_API_KEY)
    exp = comet_ml.start(project_name=COMET_PROJECT)
    exp.set_name(name)
    if tags:
        exp.add_tags(tags)
    return exp


def plot_confusion_matrix(cm: np.ndarray, class_names, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1, cbar_kws={"label": "row-normalised"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_and_log(exp, model_name: str, y_true, y_pred, class_names, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # The split is by flow vector, not stratified, so a rare class can be absent
    # from the test set. Pin the label set so that stays a row of zeros, not a crash.
    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"[{model_name}] accuracy={acc:.4f} balanced_accuracy={bal_acc:.4f} macro_f1={macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))

    metrics = {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1}
    for cls, m in report.items():
        if isinstance(m, dict):
            safe_cls = cls.replace(" ", "_")
            metrics[f"class_{safe_cls}_precision"] = m["precision"]
            metrics[f"class_{safe_cls}_recall"] = m["recall"]
            metrics[f"class_{safe_cls}_f1"] = m["f1-score"]

    if exp is not None:
        exp.log_metrics(metrics)
        exp.log_confusion_matrix(matrix=cm.tolist(), labels=class_names, title=f"{model_name} confusion matrix")

    cm_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, f"{model_name} — NF-ToN-IoT (centralised)", cm_path)
    if exp is not None:
        exp.log_image(cm_path, name=f"{model_name}_confusion_matrix")

    cm_csv = os.path.join(out_dir, f"{model_name}_confusion_matrix.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv)
    if exp is not None:
        exp.log_asset(cm_csv)

    report_path = os.path.join(out_dir, f"{model_name}_classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if exp is not None:
        exp.log_asset(report_path)

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "confusion_matrix": cm, "report": report}
