"""
Federated Client Dataset Splitter (IID / Non-IID)
==================================================

Given a raw network-traffic CSV (e.g. NF-ToN-IoT), this script holds out a
shared validation and test set, then partitions the remaining training pool
among a number of simulated federated clients, writing two variants of that
partition side by side:

    llm/dataset/
      ├── val.csv          <- shared across both variants
      ├── test.csv         <- shared across both variants
      ├── iid/
      │   ├── client_1.csv
      │   ├── ...
      │   └── client_N.csv
      └── non_iid/
          ├── client_1.csv
          ├── ...
          └── client_N.csv

val.csv and test.csv live at the top level on purpose: every model and every
partitioning strategy must be evaluated on the exact same held-out data for
the results to be comparable.

- IID split:      each client gets a stratified random sample, so every
                   client sees roughly the same attack-type distribution.
- Non-IID split:  a Dirichlet(alpha) label-skew split, so clients end up
                   with very different attack-type distributions (lower
                   alpha => more skew, higher alpha => closer to IID).

At the end, two attack-distribution tables (rows = attack type,
columns = client) are logged, one for each variant, and also saved as CSVs
next to the client files.

Usage:
    python llm/split_dataset.py --input_file datasets/nftoniot/NF-ToN-IoT.csv \
        --num_clients 5 --max_rows 100000 --alpha 0.5
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Standardize NF-ToN-IoT style column names to a common format
COLUMN_MAPPING = {
    'IPV4_SRC_ADDR': 'Src IP',
    'IPV4_DST_ADDR': 'Dst IP',
    'L4_SRC_PORT': 'Src Port',
    'L4_DST_PORT': 'Dst Port',
    'PROTOCOL': 'Protocol',
    'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Split a CSV dataset into IID and Non-IID federated client shards')
    parser.add_argument('--input_file', type=str, required=True,
                         help='Path to the raw CSV dataset')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'dataset'),
                         help='Directory to save the iid/ and non_iid/ client shards (default: llm/dataset)')
    parser.add_argument('--num_clients', type=int, default=5,
                         help='Number of federated clients')
    parser.add_argument('--max_rows', type=int, default=None,
                         help='Maximum number of rows to sample randomly from the dataset')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                         help='Fraction of the dataset held out as the shared test set')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                         help='Fraction of the dataset held out as the shared validation set')
    parser.add_argument('--label_col', type=str, default='Attack',
                         help='Column holding the attack type used for the distribution tables and non-IID split')
    parser.add_argument('--alpha', type=float, default=0.5,
                         help='Dirichlet concentration parameter for the Non-IID split. '
                              'Lower = more skewed/heterogeneous, higher = closer to IID')
    parser.add_argument('--seed', type=int, default=42,
                         help='Random seed for reproducibility')
    return parser.parse_args()


def load_dataset(input_file, max_rows, seed):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}. Please check the path and try again.")

    df = pd.read_csv(input_file)
    df = df.rename(columns=COLUMN_MAPPING)
    total_rows = len(df)
    logger.info(f"Loaded dataset with {total_rows} records from {input_file}")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        logger.info(f"Randomly sampled down to {len(df)} records (max_rows={max_rows})")

    return df, total_rows


def stratified_holdout(df, label_col, ratio, seed):
    """Carve `ratio` of the rows off as a holdout set, preserving the attack-type mix."""
    # train_test_split cannot stratify on a class that has a single row, which
    # happens for the rarest attacks once --max_rows shrinks the dataset.
    stratify = df[label_col] if df[label_col].value_counts().min() >= 2 else None
    if stratify is None:
        logger.warning("Some attack types have only one row; falling back to an unstratified holdout split.")

    return train_test_split(df, test_size=ratio, random_state=seed, stratify=stratify)


def split_iid(df, label_col, num_clients, seed):
    """Stratified split: every client gets roughly the same attack-type distribution."""
    rng = np.random.RandomState(seed)
    client_indices = [[] for _ in range(num_clients)]

    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        chunks = np.array_split(idx, num_clients)
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    return [df.loc[idx].sample(frac=1, random_state=seed).reset_index(drop=True) for idx in client_indices]


def split_non_iid(df, label_col, num_clients, alpha, seed):
    """Dirichlet(alpha) label-skew split: clients end up with very different attack mixes."""
    rng = np.random.RandomState(seed)
    client_indices = [[] for _ in range(num_clients)]

    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # Convert proportions into cumulative split points over this label's rows
        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, split_points)

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    return [df.loc[idx].sample(frac=1, random_state=seed).reset_index(drop=True) for idx in client_indices]


def save_client_shards(client_dfs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, client_df in enumerate(client_dfs):
        client_id = i + 1
        client_path = os.path.join(output_dir, f'client_{client_id}.csv')
        client_df.to_csv(client_path, index=False)
        logger.info(f"Saved client {client_id} to {client_path} ({len(client_df)} records)")


def build_distribution_table(named_dfs, label_col):
    """Rows = attack type, one column per named frame, cells = row counts. Adds a Total column/row."""
    counts = {name: frame[label_col].value_counts() for name, frame in named_dfs.items()}

    table = pd.DataFrame(counts).fillna(0).astype(int)
    table = table.sort_index()
    table['Total'] = table.sum(axis=1)
    table.loc['Total'] = table.sum(axis=0)
    table.index.name = label_col
    return table


def log_table(title, table):
    logger.info(f"{title}")
    try:
        from tabulate import tabulate
        rendered = tabulate(table, headers='keys', tablefmt='grid')
    except ImportError:
        rendered = table.to_string()
    for line in rendered.splitlines():
        logger.info(line)


def main():
    args = parse_args()

    df, total_rows = load_dataset(args.input_file, args.max_rows, args.seed)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in dataset columns: {list(df.columns)}")

    iid_dir = os.path.join(args.output_dir, 'iid')
    non_iid_dir = os.path.join(args.output_dir, 'non_iid')
    os.makedirs(args.output_dir, exist_ok=True)

    # Global label mapper, written before splitting so every client, model and
    # split shares one label ordering
    label_mapper = {label: i for i, label in enumerate(sorted(df[args.label_col].unique()))}
    mapper_path = os.path.join(args.output_dir, 'label_mapper.json')
    with open(mapper_path, 'w') as f:
        json.dump(label_mapper, f, indent=2)
    logger.info(f"Saved label mapper with {len(label_mapper)} classes to {mapper_path}")

    # Hold out the shared val/test sets first, so both variants are evaluated identically
    train_pool, test_df = stratified_holdout(df, args.label_col, args.test_ratio, args.seed)
    train_pool, val_df = stratified_holdout(
        train_pool, args.label_col, args.val_ratio / (1 - args.test_ratio), args.seed
    )

    val_path = os.path.join(args.output_dir, 'val.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved shared validation set to {val_path} ({len(val_df)} records)")
    logger.info(f"Saved shared test set to {test_path} ({len(test_df)} records)")
    logger.info(f"Training pool for client partitioning: {len(train_pool)} records")

    logger.info(f"Creating IID split across {args.num_clients} clients...")
    iid_clients = split_iid(train_pool, args.label_col, args.num_clients, args.seed)
    save_client_shards(iid_clients, iid_dir)

    logger.info(f"Creating Non-IID split (Dirichlet alpha={args.alpha}) across {args.num_clients} clients...")
    non_iid_clients = split_non_iid(train_pool, args.label_col, args.num_clients, args.alpha, args.seed)
    save_client_shards(non_iid_clients, non_iid_dir)

    global_table = build_distribution_table(
        {'train_pool': train_pool, 'val': val_df, 'test': test_df}, args.label_col
    )
    iid_table = build_distribution_table(
        {f'client_{i + 1}': d for i, d in enumerate(iid_clients)}, args.label_col
    )
    non_iid_table = build_distribution_table(
        {f'client_{i + 1}': d for i, d in enumerate(non_iid_clients)}, args.label_col
    )

    log_table("Global split distribution (rows=attack type, columns=split):", global_table)
    log_table("IID attack distribution (rows=attack type, columns=client):", iid_table)
    log_table("Non-IID attack distribution (rows=attack type, columns=client):", non_iid_table)

    global_table.to_csv(os.path.join(args.output_dir, 'split_distribution.csv'))
    iid_table.to_csv(os.path.join(iid_dir, 'attack_distribution.csv'))
    non_iid_table.to_csv(os.path.join(non_iid_dir, 'attack_distribution.csv'))

    # Provenance for the experiment runs to log to Comet, so any result can be
    # traced back to the exact partition it was trained on
    split_config = {
        'dataset': os.path.basename(args.input_file),
        'dataset_path': os.path.abspath(args.input_file),
        'total_rows_available': total_rows,
        'total_rows_used': len(df),
        'max_rows_requested': args.max_rows,
        'num_clients': args.num_clients,
        'dirichlet_alpha': args.alpha,
        'test_ratio': args.test_ratio,
        'val_ratio': args.val_ratio,
        'train_pool_rows': len(train_pool),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'num_classes': len(label_mapper),
        'attack_classes': list(label_mapper.keys()),
        'iid_client_sizes': [len(d) for d in iid_clients],
        'non_iid_client_sizes': [len(d) for d in non_iid_clients],
        'label_col': args.label_col,
        'seed': args.seed,
        'script': 'split_dataset.py',
    }
    config_path = os.path.join(args.output_dir, 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(split_config, f, indent=2)
    logger.info(f"Saved split config to {config_path}")

    logger.info("All done! IID and Non-IID client shards written to "
                f"{iid_dir} and {non_iid_dir}")


if __name__ == "__main__":
    main()
