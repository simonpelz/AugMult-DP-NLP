"""YAML config loading. The full resolved config is hashed and stored in the manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from views.utils import stable_hash

DEFAULTS: Dict[str, Any] = {
    "name": None,
    "seed": 0,
    "k": 8,  # candidate views per example per generator (excluding identity)
    "device": "auto",
    "dataset": {},
    "generators": [],
    "scorers": [],
    "pair_combine": "alternate",  # how per-field candidates become pair views: alternate | zip
}


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    out = {**DEFAULTS, **cfg}
    out["name"] = out["name"] or path.stem
    out["_config_path"] = str(path)
    return out


def config_hash(cfg: Dict[str, Any]) -> str:
    return stable_hash({k: v for k, v in cfg.items() if not k.startswith("_")}, n=8)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
