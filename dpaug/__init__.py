"""dpaug — functional (torch.func) Augmentation Multiplicity for DP-SGD.

Opacus-fork-free re-implementation of the thesis' AugMult idea:
per-example gradients are computed over a block of K views of the same
example (any per-example objective over those K views), then clipped,
summed, noised and accounted exactly as in DP-SGD.

Only Opacus' *accountants* are used; no GradSampleModule / hooks.
"""

from dpaug.augmult import (
    DPAugMult,
    DPConfig,
    PoissonSampler,
    clip_per_example,
    per_example_grads,
    privatize,
)
from dpaug.losses import (
    ce_of_mean_logits,
    mean_view_ce,
    mean_view_ce_with_consistency,
    mean_view_distillation,
)

__all__ = [
    "DPAugMult",
    "DPConfig",
    "PoissonSampler",
    "clip_per_example",
    "per_example_grads",
    "privatize",
    "mean_view_ce",
    "ce_of_mean_logits",
    "mean_view_ce_with_consistency",
    "mean_view_distillation",
]
