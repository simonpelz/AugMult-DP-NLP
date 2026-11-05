import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python, numpy and torch (CPU + CUDA). Returns a seeded torch.Generator
    for DataLoaders (Opacus' DPDataLoader reuses it for Poisson sampling), or None if seed is None.

    seed=None keeps the thesis behaviour (unseeded runs)."""
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only affects subprocesses (e.g. dataloader workers started via spawn); hash order in this
    # process is fixed at interpreter start. Code that depends on ordering must sort explicitly.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """DataLoader worker_init_fn: torch already seeds each worker from the loader's base seed,
    propagate it to numpy/random (used by nlpaug augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
