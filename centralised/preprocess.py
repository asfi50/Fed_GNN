"""
Centralised NF-ToN-IoT preprocessing: engineer leak-free flow features, split
by flow vector, then balance the training set.

LEAKAGE POLICY — every feature here is computed from a single flow's own
columns. Nothing is derived by grouping across rows.

An earlier version of this script engineered group-based features
(flows_per_src_ip, dst_ip_diversity, dst_port_diversity, port_to_flow_ratio,
session_regularity) over the whole dataset. Those leak the label: each attack
class in NF-ToN-IoT is generated from only 2-10 source IPs, so
flows_per_src_ip takes just 46 distinct values across 1.38M rows and acts as an
attacker-machine fingerprint rather than a traffic statistic. Knowing only
flows_per_src_ip and dst_port_diversity gives the label 67.5% of the time with
no learning at all, and they were computed with full sight of the test rows.
That is the same loophole as Finding 4 in findings.md, re-entered by a
different door, so they are gone.

Also dropped: L4_SRC_PORT (ephemeral, pure memorisation — it was the Random
Forest's top feature) and anything derived from it.

SPLITTING — 63.7% of raw rows belong to an exact duplicate group, so a plain
random split puts identical flows on both sides and rewards memorisation.
Rows are therefore split by flow-feature vector: every distinct vector goes
wholly to train or wholly to test, keeping its natural mix of labels. Train
is then balanced by undersampling; test is left at its natural distribution.

Usage:
    python preprocess.py --input ../datasets/nftoniot/NF-ToN-IoT.csv \
        --output data/nfton_balanced.csv --samples-per-class 15000

Mac note: only run with --nrows on a small slice for a smoke test. The real
run (full 1.3M rows) happens on Kaggle via run_kaggle.ipynb.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Attack"
SPLIT_COL = "split"

# Raw columns kept as features: intrinsic to one flow. IP addresses and
# L4_SRC_PORT are excluded — they identify the capture session, not the attack.
FLOW_COLS = [
    "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES",
    "IN_PKTS", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
]
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-flow derived features. Every expression below uses only columns
    of the same row, so no information crosses the train/test boundary."""
    df = df.copy()

    total_bytes = df["IN_BYTES"] + df["OUT_BYTES"]
    total_pkts = df["IN_PKTS"] + df["OUT_PKTS"]

    df["out_in_bytes_ratio"] = (df["OUT_BYTES"] / (df["IN_BYTES"] + 1e-6)).clip(0, 100).fillna(0)
    df["bytes_per_packet"] = (total_bytes / (total_pkts + 1e-6)).clip(0, 10000).fillna(0)
    df["pkt_asymmetry"] = (abs(df["IN_PKTS"] - df["OUT_PKTS"]) / (total_pkts + 1e-6)).clip(0, 1).fillna(0)
    df["nf_flow_rate"] = (total_pkts / (df["FLOW_DURATION_MILLISECONDS"] + 1e-3)).clip(0, 1000).fillna(0)

    df["byte_per_pkt_out"] = (df["OUT_BYTES"] / (df["OUT_PKTS"] + 1e-6)).clip(0, 10000).fillna(0)
    df["out_in_pkts_ratio"] = (df["OUT_PKTS"] / (df["IN_PKTS"] + 1e-6)).clip(0, 50).fillna(0)
    df["byte_density"] = (total_bytes / (df["FLOW_DURATION_MILLISECONDS"] + 1e-3)).clip(0, 100000).fillna(0)

    # Fixed thresholds, not data-derived — no statistic is fitted on the data here.
    web_ports = [80, 443, 8080, 8443]
    db_ports = [1433, 1521, 3306, 5432]
    df["is_web_port"] = df["L4_DST_PORT"].isin(web_ports).astype(int)
    df["is_db_port"] = df["L4_DST_PORT"].isin(db_ports).astype(int)
    df["targets_system_port"] = (df["L4_DST_PORT"] < 1024).astype(int)

    ratio = (df["OUT_BYTES"] / (df["IN_BYTES"] + 1e-6)).clip(0, 100)
    df["response_size_category"] = pd.cut(
        ratio, bins=[-1, 1.0, 3.0, float("inf")], labels=[0, 1, 2]
    ).astype(float)

    return df.fillna(0)


def split_by_flow_vector(df: pd.DataFrame, test_size: float, seed: int):
    """Assign each distinct flow-feature vector wholly to train or to test.

    63.7% of raw rows belong to an exact duplicate group, so a plain random
    split would put identical flows on both sides and reward memorisation.
    Deduplicating instead would distort the data the other way: 100% of dos,
    password, scanning and xss rows share their feature vector with a different
    class, and collapsing each (vector, label) pair to one row rewrites those
    real class proportions. Splitting by vector avoids both — no vector spans
    the split, and every vector keeps its natural mix of labels.
    """
    groups = df.groupby(FLOW_COLS, sort=False).ngroup()
    n_groups = groups.nunique()
    rng = np.random.default_rng(seed)
    test_groups = set(rng.choice(n_groups, size=int(n_groups * test_size), replace=False).tolist())

    is_test = groups.isin(test_groups)
    logger.info("Split by flow vector: %d distinct vectors -> %d train / %d test rows",
                n_groups, (~is_test).sum(), is_test.sum())
    return df[~is_test], df[is_test]


def balance_classes(df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    """Undersample only. Oversampling with replacement would put the same row on
    both sides of the train/test split, which is the leak we just removed."""
    parts = []
    for cls in df[TARGET_COL].unique():
        subset = df[df[TARGET_COL] == cls]
        n = min(samples_per_class, len(subset))
        if n < samples_per_class:
            logger.info("Class %s has only %d train rows, keeping all (no oversampling)", cls, n)
        parts.append(subset.sample(n=n, random_state=seed))

    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw NF-ToN-IoT.csv")
    parser.add_argument("--output", required=True, help="Path to write the balanced+featured CSV")
    parser.add_argument("--samples-per-class", type=int, default=15000,
                         help="Upper cap per class. Classes with fewer rows keep all of them.")
    parser.add_argument("--min-raw-count", type=int, default=500,
                         help="Classes with fewer rows than this are dropped")
    parser.add_argument("--nrows", type=int, default=None,
                         help="Only read first N rows. Mac smoke-test only — never use for the real run.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading %s%s", args.input, f" (nrows={args.nrows}, SMOKE TEST)" if args.nrows else "")
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
        if args.nrows:
            df = df.head(args.nrows)
    else:
        df = pd.read_csv(args.input, nrows=args.nrows)
    logger.info("Loaded %d rows. Raw class counts:\n%s", len(df), df[TARGET_COL].value_counts().to_string())

    counts = df[TARGET_COL].value_counts()
    dropped = counts[counts < args.min_raw_count]
    if len(dropped):
        logger.info("Dropping classes with < %d rows: %s", args.min_raw_count, dropped.to_dict())
        df = df[df[TARGET_COL].isin(counts[counts >= args.min_raw_count].index)]

    logger.info("Engineering per-flow features (no cross-row grouping)...")
    df = engineer_features(df)

    # Split BEFORE balancing: undersampling discards rows that share a feature
    # vector with another class, so balancing first would hand the test set an
    # artificially separable view of the data and inflate every score.
    train_df, test_df = split_by_flow_vector(df, args.test_size, args.seed)
    logger.info("Test set keeps the natural class distribution and row multiplicity")

    logger.info("Balancing train only (cap %d rows/class, undersample only)...", args.samples_per_class)
    train_df = balance_classes(train_df, args.samples_per_class, args.seed)

    train_df, test_df = train_df.copy(), test_df.copy()
    train_df[SPLIT_COL], test_df[SPLIT_COL] = "train", "test"
    out = pd.concat([train_df, test_df], ignore_index=True)
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns])

    logger.info("Final dataset: %d rows, %d columns", *out.shape)
    logger.info("Train class distribution:\n%s", train_df[TARGET_COL].value_counts().to_string())
    logger.info("Test class distribution:\n%s", test_df[TARGET_COL].value_counts().to_string())
    logger.info("Features: %s", [c for c in out.columns if c not in (TARGET_COL, SPLIT_COL)])

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
