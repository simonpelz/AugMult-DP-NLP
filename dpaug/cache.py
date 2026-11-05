"""Feature caching for the frozen-encoder ("head-only") regime.

When only the head is trained (the thesis' ``classifier_and_pooler`` setup), the
encoder is a fixed function of each view, so its output can be computed ONCE per
(example, view) and DP-SGD then trains a tiny head on cached features — orders of
magnitude faster, enabling many seeds and large K.

IMPORTANT difference to the thesis runs: features are computed with the encoder in
``eval()`` mode, i.e. WITHOUT encoder dropout. The thesis kept ``model.train()``,
so its encoder dropout acted as an implicit extra augmentation (also for "no
augmentation" runs). To reproduce that effect, add explicit dropout views
(e.g. cache several stochastic passes with ``stochastic_passes > 1``) or put
dropout into the head.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


@torch.no_grad()
def cache_bert_cls(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    batch_size: int = 256,
    stochastic_passes: int = 0,
) -> torch.Tensor:
    """Cache last-layer [CLS] hidden states (pre-pooler) for ``(N, K, L)`` inputs -> ``(N, K', D)``.

    ``stochastic_passes=m>0`` additionally runs the encoder m times in train mode per view
    (dropout views), giving K' = K * (1 + m). Works for any HF model exposing ``.base_model``.
    """
    enc = model.base_model
    n, k = inputs["input_ids"].shape[:2]
    flat = {key: v.reshape(n * k, -1) for key, v in inputs.items()}
    was_training = enc.training
    passes = [False] + [True] * stochastic_passes
    outs = []
    for train_mode in passes:
        enc.train(train_mode)
        chunks = []
        for s in range(0, n * k, batch_size):
            b = {key: v[s : s + batch_size] for key, v in flat.items()}
            chunks.append(enc(**b).last_hidden_state[:, 0])
        outs.append(torch.cat(chunks).reshape(n, k, -1))
    enc.train(was_training)
    return torch.cat(outs, dim=1)


class BertHead(nn.Module):
    """pooler(dense+tanh) -> dropout -> classifier, copied from a BertForSequenceClassification.

    Mirrors the thesis' trainable set (pooler + classifier). Input ``(..., D)`` CLS features.
    """

    def __init__(self, model: nn.Module, copy: bool = True):
        super().__init__()
        import copy as _copy

        c = _copy.deepcopy if copy else (lambda m: m)
        base = model.base_model
        self.pooler_dense = c(base.pooler.dense)
        self.dropout = c(model.dropout)
        self.classifier = c(model.classifier)
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(torch.tanh(self.pooler_dense(features))))
