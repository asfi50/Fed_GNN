"""
Centralised NF-ToN-IoT preprocessing: engineer flow-based features on the FULL
dataset, then balance classes to a fixed number of rows each.

Group-based features (flows_per_src_ip, dst_ip_diversity, session_regularity, ...)
are only meaningful when computed over the full flow population, so feature
engineering runs BEFORE balancing/sampling.

These features target the exact classes that were confused in the federated
FedGATSage model (ddos/injection/password/scanning/xss all look similar in raw
NetFlow fields, see findings.md and paper.md). The question this centralised
experiment answers: with full access to the data (no federated partitioning),
can a simple tree model separate them?

Usage:
    python preprocess.py --input ../datasets/nftoniot/NF-ToN-IoT.csv \
        --output data/nfton_balanced.csv --samples-per-class 15000

Mac note: only run with --nrows on a small slice for a smoke test. The real
run (full 1.3M rows) happens on Kaggle via run_kaggle.ipynb.
"""
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Attack"
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- base features: DDoS vs Injection/Password/XSS byte/packet shape ---
    df["out_in_bytes_ratio"] = (df["OUT_BYTES"] / (df["IN_BYTES"] + 1e-6)).clip(0, 100).fillna(0)
    total_bytes = df["IN_BYTES"] + df["OUT_BYTES"]
    total_pkts = df["IN_PKTS"] + df["OUT_PKTS"]
    df["bytes_per_packet"] = (total_bytes / (total_pkts + 1e-6)).clip(0, 10000).fillna(0)
    df["pkt_asymmetry"] = (abs(df["IN_PKTS"] - df["OUT_PKTS"]) / (total_pkts + 1e-6)).clip(0, 1).fillna(0)
    df["nf_flow_rate"] = (total_pkts / (df["FLOW_DURATION_MILLISECONDS"] + 1e-3)).clip(0, 1000).fillna(0)

    # --- temporal/diversity features: DDoS vs Scanning vs Password ---
    flows_per_src = df["IPV4_SRC_ADDR"].map(df["IPV4_SRC_ADDR"].value_counts())
    df["flows_per_src_ip"] = flows_per_src.fillna(1)
    df["flows_per_src_ip_norm"] = (df["flows_per_src_ip"] / (len(df) + 1e-6)).clip(0, 1)

    dst_ip_diversity = df.groupby("IPV4_SRC_ADDR")["IPV4_DST_ADDR"].transform("nunique")
    df["dst_ip_diversity"] = dst_ip_diversity.clip(lower=1)
    df["dst_ip_diversity_log"] = np.log1p(df["dst_ip_diversity"])

    dst_port_diversity = df.groupby("IPV4_SRC_ADDR")["L4_DST_PORT"].transform("nunique")
    df["dst_port_diversity"] = dst_port_diversity.clip(lower=1)
    df["dst_port_diversity_log"] = np.log1p(df["dst_port_diversity"])
    df["port_to_flow_ratio"] = (df["dst_port_diversity"] / (df["flows_per_src_ip"] + 1e-6)).clip(0, 1)

    # --- content features: Injection vs XSS vs Password response shape ---
    web_ports = [80, 443, 8080, 8443]
    db_ports = [1433, 1521, 3306, 5432]
    df["is_web_port"] = df["L4_DST_PORT"].isin(web_ports).astype(int)
    df["is_db_port"] = df["L4_DST_PORT"].isin(db_ports).astype(int)

    ratio = (df["OUT_BYTES"] / (df["IN_BYTES"] + 1e-6)).clip(0, 100)
    df["response_size_category"] = pd.cut(
        ratio, bins=[-1, 1.0, 3.0, float("inf")], labels=[0, 1, 2]
    ).astype(float)

    df["byte_per_pkt_out"] = (df["OUT_BYTES"] / (df["OUT_PKTS"] + 1e-6)).clip(0, 10000).fillna(0)
    df["out_in_pkts_ratio"] = (df["OUT_PKTS"] / (df["IN_PKTS"] + 1e-6)).clip(0, 50).fillna(0)
    df["byte_density"] = (
        (df["IN_BYTES"] + df["OUT_BYTES"]) / (df["FLOW_DURATION_MILLISECONDS"] + 1e-3)
    ).clip(0, 100000).fillna(0)

    # --- behavioral features: Password brute-force vs Backdoor vs Scanning ---
    df["is_ephemeral_src"] = (df["L4_SRC_PORT"] > 1024).astype(int)
    df["targets_system_port"] = (df["L4_DST_PORT"] < 1024).astype(int)
    df["port_spread"] = (df["L4_SRC_PORT"] - df["L4_DST_PORT"]).abs()

    median_duration = df["FLOW_DURATION_MILLISECONDS"].median()
    df["is_short_session"] = (df["FLOW_DURATION_MILLISECONDS"] < median_duration / 10).astype(int)
    df["is_long_session"] = (df["FLOW_DURATION_MILLISECONDS"] > median_duration * 10).astype(int)

    port_cv = df.groupby("L4_DST_PORT")["IN_BYTES"].transform(lambda x: x.std() / (x.mean() + 1e-6))
    df["session_regularity"] = (1.0 / (port_cv.fillna(0) + 1e-6)).clip(0, 100)

    df = df.fillna(0)
    return df


def balance_classes(df: pd.DataFrame, samples_per_class: int, min_raw_count: int, seed: int) -> pd.DataFrame:
    counts = df[TARGET_COL].value_counts()
    keep_classes = counts[counts >= min_raw_count].index.tolist()
    dropped = counts[counts < min_raw_count]
    if len(dropped):
        logger.info("Dropping classes with < %d raw rows (too rare to balance): %s",
                     min_raw_count, dropped.to_dict())

    parts = []
    for cls in keep_classes:
        subset = df[df[TARGET_COL] == cls]
        if len(subset) >= samples_per_class:
            sampled = subset.sample(n=samples_per_class, random_state=seed)
        else:
            logger.info("Oversampling %s: %d -> %d rows", cls, len(subset), samples_per_class)
            sampled = subset.sample(n=samples_per_class, replace=True, random_state=seed)
        parts.append(sampled)

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw NF-ToN-IoT.csv")
    parser.add_argument("--output", required=True, help="Path to write the balanced+featured CSV")
    parser.add_argument("--samples-per-class", type=int, default=15000)
    parser.add_argument("--min-raw-count", type=int, default=15000,
                         help="Classes with fewer raw rows than this are dropped (e.g. mitm, ransomware)")
    parser.add_argument("--nrows", type=int, default=None,
                         help="Only read first N rows. Mac smoke-test only — never use for the real run.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading %s%s", args.input, f" (nrows={args.nrows}, SMOKE TEST)" if args.nrows else "")
    df = pd.read_csv(args.input, nrows=args.nrows)
    logger.info("Loaded %d rows. Raw class counts:\n%s", len(df), df[TARGET_COL].value_counts().to_string())

    logger.info("Engineering features on the full dataset...")
    df = engineer_features(df)

    logger.info("Balancing classes to %d rows each...", args.samples_per_class)
    balanced = balance_classes(df, args.samples_per_class, args.min_raw_count, args.seed)

    balanced = balanced.drop(columns=[c for c in DROP_COLS if c in balanced.columns])
    logger.info("Final balanced dataset: %d rows, %d columns", *balanced.shape)
    logger.info("Balanced class distribution:\n%s", balanced[TARGET_COL].value_counts().to_string())

    balanced.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
