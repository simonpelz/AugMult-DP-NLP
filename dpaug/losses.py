"""Per-example objectives over a block of K views.

Signature of every loss: ``loss(logits, label, extras) -> scalar`` where
``logits`` has shape ``(K, C)`` (the K views of ONE example), ``label`` is a
0-dim integer tensor and ``extras`` is a dict of per-example tensors (or None).

Privacy note: anything computed here may only depend on this single example
(its views, its label, its own ``extras`` such as public-teacher outputs) and
public models. Mixing examples before clipping breaks the DP analysis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_view_ce(logits: torch.Tensor, label: torch.Tensor, extras=None) -> torch.Tensor:
    """Thesis AugMult objective: cross-entropy averaged over the K views.

    Its gradient is the average of the K per-view gradients (De et al. 2022).
    """
    k = logits.shape[0]
    return F.cross_entropy(logits, label.expand(k))


def ce_of_mean_logits(logits: torch.Tensor, label: torch.Tensor, extras=None) -> torch.Tensor:
    """Cross-entropy of the view-averaged logits (train-time 'TTA' objective)."""
    return F.cross_entropy(logits.mean(0, keepdim=True), label.reshape(1))


def mean_view_ce_with_consistency(lam: float = 1.0):
    """Mean-view CE + lam * mean KL(view || mean prediction) (R-Drop/UDA-style)."""

    def loss(logits, label, extras=None):
        logp = F.log_softmax(logits, dim=-1)
        mean_p = logp.exp().mean(0, keepdim=True)
        kl = (logp.exp() * (logp - mean_p.clamp_min(1e-12).log())).sum(-1).mean()
        return mean_view_ce(logits, label) + lam * kl

    return loss


def mean_view_distillation(alpha: float = 0.5, temperature: float = 1.0, key: str = "teacher_logits"):
    """(1-alpha) * mean-view CE + alpha * T^2 * KL(teacher || student) per view.

    ``extras[key]`` must have shape ``(K, C)`` (per-view teacher logits, e.g. from
    a public LLM that produced/judged the views). Only *public* teachers keep DP intact
    w.r.t. the private data; a teacher fit on private data needs its own accounting.
    """

    def loss(logits, label, extras):
        t = temperature
        teacher = extras[key]
        kd = F.kl_div(
            F.log_softmax(logits / t, -1), F.log_softmax(teacher / t, -1), log_target=True, reduction="batchmean"
        )
        return (1 - alpha) * mean_view_ce(logits, label) + alpha * (t**2) * kd

    return loss
