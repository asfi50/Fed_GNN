"""Turning netflow CSV rows into tokenised text, and handing clients their per-round slice."""

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Shorter field names cost fewer tokens per row, which directly buys training speed.
SHORT_NAMES = {
    'Src Port': 'sport',
    'Dst Port': 'dport',
    'Protocol': 'proto',
    'L7_PROTO': 'l7',
    'IN_BYTES': 'inb',
    'OUT_BYTES': 'outb',
    'IN_PKTS': 'inp',
    'OUT_PKTS': 'outp',
    'TCP_FLAGS': 'flags',
    'Flow Duration': 'dur',
    'Src IP': 'src',
    'Dst IP': 'dst',
}


def feature_columns(df: pd.DataFrame, drop_columns: List[str], include_ips: bool) -> List[str]:
    drop = set(drop_columns)
    if include_ips:
        drop -= {'Src IP', 'Dst IP'}
    return [c for c in df.columns if c not in drop]


def serialize_frame(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Render each row as "proto=6 sport=443 inb=1234 ...". Vectorised - iterrows is far too slow here."""
    text = None
    for col in columns:
        field = SHORT_NAMES.get(col, col.lower().replace(' ', '_'))
        part = field + '=' + df[col].astype(str)
        text = part if text is None else text + ' ' + part
    return text.tolist()


def load_label_mapper(data_dir: str) -> Dict[str, int]:
    path = os.path.join(data_dir, 'label_mapper.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run llm/split_dataset.py first - it writes the global label mapper."
        )
    with open(path) as f:
        return json.load(f)


def compute_class_weights(data_dir: str, label_mapper: Dict[str, int], max_weight: float = 50.0) -> torch.Tensor:
    """Inverse-frequency weights from the GLOBAL training pool.

    Global rather than per-client on purpose: a client whose shard happens to
    contain none of a class would otherwise generate an infinite weight for it.

    Weights are clipped because NF-ToN-IoT is extremely long-tailed - ransomware
    is ~0.0002% of the data, which unclipped gives it a weight in the thousands
    and lets a single row dominate the gradient.
    """
    stats = pd.read_csv(os.path.join(data_dir, 'split_distribution.csv'), index_col=0)
    counts = stats['train_pool'].drop(index='Total', errors='ignore')

    total = counts.sum()
    num_classes = len(label_mapper)
    weights = torch.ones(num_classes, dtype=torch.float)
    for label, idx in label_mapper.items():
        count = counts.get(label, 0)
        if count > 0:
            weights[idx] = min(total / (num_classes * count), max_weight)

    clipped = [n for n, i in label_mapper.items() if weights[i] >= max_weight]
    if clipped:
        logger.info(f"Class weights clipped at {max_weight} for: {clipped}")
    return weights


class FlowDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        encoded['labels'] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def frame_to_dataset(df: pd.DataFrame, cfg, label_mapper: Dict[str, int]) -> FlowDataset:
    columns = feature_columns(df, cfg.data.drop_columns, cfg.data.include_ips)
    texts = serialize_frame(df, columns)
    labels = df[cfg.data.label_col].map(label_mapper).tolist()
    return FlowDataset(texts, labels)


def make_loader(dataset: FlowDataset, tokenizer, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer, cfg.model.max_length),
    )


class ClientShard:
    """One client's private slice of the training pool.

    The shard is shuffled once with a fixed seed and then walked through in
    consecutive slices, one per round. That way a client never trains on the
    same row twice, and the whole schedule is reproducible.
    """

    def __init__(self, csv_path: str, cfg, label_mapper: Dict[str, int], client_id: int):
        self.client_id = client_id
        self.df = pd.read_csv(csv_path)
        # Per-client seed offset so clients do not walk their shards in lockstep
        self.df = self.df.sample(frac=1, random_state=cfg.seed + client_id).reset_index(drop=True)
        self.cfg = cfg
        self.label_mapper = label_mapper

    def __len__(self):
        return len(self.df)

    def round_frame(self, round_idx: int) -> pd.DataFrame:
        n = self.cfg.federated.rows_per_client_per_round
        if n is None or n >= len(self.df):
            return self.df

        start = (round_idx * n) % len(self.df)
        end = start + n
        if end <= len(self.df):
            return self.df.iloc[start:end]
        # Wrapped past the end of the shard - take the tail plus rows from the front
        return pd.concat([self.df.iloc[start:], self.df.iloc[: end - len(self.df)]])

    def round_dataset(self, round_idx: int) -> FlowDataset:
        return frame_to_dataset(self.round_frame(round_idx), self.cfg, self.label_mapper)

    def label_distribution(self) -> pd.Series:
        return self.df[self.cfg.data.label_col].value_counts()


def load_client_shards(cfg, split: str, label_mapper: Dict[str, int]) -> List[ClientShard]:
    shards = []
    for client_id in range(1, cfg.federated.num_clients + 1):
        path = os.path.join(cfg.data.data_dir, split, f'client_{client_id}.csv')
        shard = ClientShard(path, cfg, label_mapper, client_id)
        logger.info(f"Client {client_id}: {len(shard)} rows in shard, "
                    f"{cfg.federated.rows_per_client_per_round} sampled per round")
        shards.append(shard)
    return shards


def load_eval_frame(cfg, name: str, subset: Optional[int], label_mapper: Dict[str, int]) -> pd.DataFrame:
    """Load val.csv / test.csv, optionally down to a stratified subset."""
    df = pd.read_csv(os.path.join(cfg.data.data_dir, f'{name}.csv'))
    if subset is not None and subset < len(df):
        frac = subset / len(df)
        # At least one row per class, so the rarest attacks do not vanish from
        # the evaluation set entirely
        parts = [
            group.sample(max(1, int(round(len(group) * frac))), random_state=cfg.seed)
            for _, group in df.groupby(cfg.data.label_col)
        ]
        df = pd.concat(parts).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    logger.info(f"Loaded {name} set: {len(df)} rows")
    return df
