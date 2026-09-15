"""Config loading: base.yaml merged with a model-specific override."""

import os
from typing import Any, Dict

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


class Config(dict):
    """Dict with attribute access, so cfg.federated.num_rounds reads cleanly.

    Nested dicts are converted once here rather than on each attribute access, so
    that cfg.federated is the same object every time. Re-wrapping per access
    would hand back a throwaway copy and make any assignment to a nested field
    silently do nothing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value


def load_config(model_name: str, overrides: Dict[str, Any] = None) -> Config:
    """Load configs/base.yaml plus configs/models/<model_name>.yaml."""
    with open(os.path.join(CONFIG_DIR, 'base.yaml')) as f:
        cfg = yaml.safe_load(f)

    model_path = os.path.join(CONFIG_DIR, 'models', f'{model_name}.yaml')
    if not os.path.exists(model_path):
        available = sorted(
            f[:-5] for f in os.listdir(os.path.join(CONFIG_DIR, 'models')) if f.endswith('.yaml')
        )
        raise FileNotFoundError(f"No config for model '{model_name}'. Available: {available}")

    with open(model_path) as f:
        cfg['model'] = yaml.safe_load(f)

    for key, value in (overrides or {}).items():
        if value is not None:
            _set_nested(cfg, key, value)

    return Config(cfg)


def _set_nested(cfg: dict, dotted_key: str, value: Any):
    """_set_nested(cfg, 'federated.num_rounds', 3) -> cfg['federated']['num_rounds'] = 3"""
    keys = dotted_key.split('.')
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
