"""Turn K text views per example into ``(N, K, L)`` tensors.

A *view function* maps a list of strings to a list of strings of the same length
(the thesis' nlpaug ``.augment`` callables fit, as do lookups into precomputed
paraphrase/back-translation tables). ``identity`` + model dropout = "dropout views".
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

ViewFn = Callable[[List[str]], List[str]]


def identity(texts: List[str]) -> List[str]:
    return list(texts)


def tokenize_views(
    tokenizer,
    texts_a: Sequence[str],
    view_fns: Sequence[ViewFn],
    texts_b: Optional[Sequence[str]] = None,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """Apply each view fn (to both segments for pair tasks), tokenize, stack to ``(N, K, L)``."""
    per_view = []
    for fn in view_fns:
        a = fn(list(texts_a))
        b = fn(list(texts_b)) if texts_b is not None else None
        enc = tokenizer(a, b, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        per_view.append(enc)
    keys = per_view[0].keys()
    return {k: torch.stack([v[k] for v in per_view], dim=1) for k in keys}
