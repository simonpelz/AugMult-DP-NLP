"""Functional AugMult DP-SGD core.

Data layout: every input tensor has shape ``(N, K, ...)`` — N examples, K views
each (e.g. ``input_ids`` of shape ``(N, K, L)`` or cached features ``(N, K, D)``).
Labels have shape ``(N,)``. For every example the model is run on its K views as
an inner batch of size K, a per-example objective turns the ``(K, C)`` logits into
one scalar, and ``torch.func.grad`` differentiates that scalar w.r.t. the trainable
parameters. ``vmap`` vectorises this over the N examples (``randomness="different"``
gives every example its own dropout masks; within an example the K views also get
independent masks because they are separate rows of the inner batch).

The per-example gradient is therefore the gradient of the per-example objective —
for :func:`dpaug.losses.mean_view_ce` exactly the thesis' "average of K view
gradients" — and DP-SGD's clip -> sum -> noise -> account applies unchanged
(privacy unit = one original example; views never leave that example).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Mapping, Optional, Union

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from dpaug.losses import mean_view_ce
from dpaug.masks import prepare_inputs

Tensor = torch.Tensor
Inputs = Union[Tensor, Mapping[str, Tensor]]
PerExampleLoss = Callable[[Tensor, Tensor, Optional[Mapping[str, Tensor]]], Tensor]


def _default_logits(out):
    return out.logits if hasattr(out, "logits") else out


def trainable_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def _index(x: Inputs, sl) -> Inputs:
    if isinstance(x, Mapping):
        return {k: v[sl] for k, v in x.items()}
    return x[sl]


def _num_examples(x: Inputs) -> int:
    if isinstance(x, Mapping):
        return next(iter(x.values())).shape[0]
    return x.shape[0]


def per_example_grads(
    model: nn.Module,
    inputs: Inputs,
    labels: Tensor,
    extras: Optional[Mapping[str, Tensor]] = None,
    per_example_loss: PerExampleLoss = mean_view_ce,
    logits_fn: Callable = _default_logits,
    randomness: str = "different",
    params: Optional[Dict[str, Tensor]] = None,
    mask_mode: Optional[str] = "auto",
) -> Dict[str, Tensor]:
    """Per-example gradients ``{name: (N, *param.shape)}`` of ``per_example_loss``.

    ``inputs``: tensor ``(N, K, ...)`` (passed positionally) or dict of such tensors
    (passed as keyword arguments, e.g. HF ``input_ids``/``attention_mask``).
    Only parameters with ``requires_grad=True`` are differentiated; frozen ones are
    taken from the module as constants.
    ``mask_mode``: an ``(N, K, L)`` attention_mask is expanded to a 4D additive mask
    before vmap (see :mod:`dpaug.masks`); "auto" picks causal vs bidirectional from
    ``model.config.model_type``; None leaves inputs untouched.
    """
    if params is None:
        params = {n: p.detach() for n, p in trainable_parameters(model).items()}
    is_dict = isinstance(inputs, Mapping)
    if is_dict:
        inputs = prepare_inputs(inputs, model, mask_mode)

    def loss_one(p, x_one, y_one, e_one):
        if is_dict:
            out = functional_call(model, p, args=(), kwargs=dict(x_one), strict=False)
        else:
            out = functional_call(model, p, args=(x_one,), strict=False)
        return per_example_loss(logits_fn(out), y_one, e_one)

    extras_in_dims = None if extras is None else 0
    fn = vmap(grad(loss_one), in_dims=(None, 0, 0, extras_in_dims), randomness=randomness)
    return fn(params, inputs, labels, extras)


def clip_per_example(grads: Dict[str, Tensor], max_grad_norm: float, eps: float = 1e-6):
    """Flat (all-layer) per-example clipping. Returns (clipped_grads, per_example_norms)."""
    sq = None
    for g in grads.values():
        s = g.reshape(g.shape[0], -1).pow(2).sum(1)
        sq = s if sq is None else sq + s
    norms = sq.sqrt()
    factor = (max_grad_norm / (norms + eps)).clamp(max=1.0)
    clipped = {n: g * factor.view(-1, *([1] * (g.dim() - 1))) for n, g in grads.items()}
    return clipped, norms


@dataclass
class DPConfig:
    """DP-SGD hyper-parameters.

    ``noise_independent_of_C`` reproduces the thesis parameterisation
    (``util/privacy_engine_util.add_noise``): the clipped sum is divided by C and
    noise has std ``sigma`` (not ``sigma*C``). Privacy is identical; it only rescales
    the update by 1/C (equivalent to lr/C), which decouples lr from C.
    """

    max_grad_norm: float
    noise_multiplier: float
    sample_rate: float
    dataset_size: int
    microbatch_size: int = 64
    noise_independent_of_C: bool = False
    accountant: str = "rdp"  # any opacus accountant mechanism: "rdp" | "prv" | "gdp"
    seed: Optional[int] = None

    @property
    def expected_batch_size(self) -> float:
        return self.sample_rate * self.dataset_size


def privatize(
    summed_clipped: Dict[str, Tensor],
    cfg: DPConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:
    """Add Gaussian noise to the sum of clipped grads and normalise by the expected batch size."""
    out = {}
    for n, s in summed_clipped.items():
        if cfg.noise_independent_of_C:
            s = s / cfg.max_grad_norm
            std = cfg.noise_multiplier
        else:
            std = cfg.noise_multiplier * cfg.max_grad_norm
        noise = torch.normal(0.0, std, size=s.shape, generator=generator, device="cpu").to(s.device, s.dtype) if std > 0 else 0.0
        out[n] = (s + noise) / cfg.expected_batch_size
    return out


class PoissonSampler:
    """Poisson subsampling (each example independently with prob. q) — what DP-SGD's accounting assumes."""

    def __init__(self, dataset_size: int, sample_rate: float, steps: int, generator: Optional[torch.Generator] = None):
        self.n, self.q, self.steps, self.g = dataset_size, sample_rate, steps, generator

    def __iter__(self) -> Iterator[Tensor]:
        for _ in range(self.steps):
            mask = torch.rand(self.n, generator=self.g) < self.q
            yield mask.nonzero(as_tuple=True)[0]

    def __len__(self):
        return self.steps


def _make_accountant(name: str):
    from opacus.accountants import create_accountant

    return create_accountant(mechanism=name)


class DPAugMult:
    """DP-SGD step with Augmentation Multiplicity for any ``nn.Module``.

    Usage::

        dp = DPAugMult(model, torch.optim.SGD(trainable, lr), cfg)
        for idx in PoissonSampler(n, q, steps, gen):
            dp.step(inputs[idx], labels[idx])     # inputs shaped (B, K, ...)
        dp.epsilon(delta)

    ``step`` processes the (variable-size) Poisson batch in microbatches of
    ``cfg.microbatch_size`` examples, accumulating only the *clipped sum*, so memory
    is bounded by one microbatch of per-example grads.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: DPConfig,
        per_example_loss: PerExampleLoss = mean_view_ce,
        logits_fn: Callable = _default_logits,
        randomness: str = "different",
    ):
        self.model, self.optimizer, self.cfg = model, optimizer, cfg
        self.per_example_loss, self.logits_fn, self.randomness = per_example_loss, logits_fn, randomness
        self.accountant = _make_accountant(cfg.accountant)
        self.noise_gen = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
        self.last_norms: Optional[Tensor] = None
        self.steps = 0

    def privatized_grads(self, inputs: Inputs, labels: Tensor, extras=None) -> Dict[str, Tensor]:
        named = trainable_parameters(self.model)
        params = {n: p.detach() for n, p in named.items()}
        summed = {n: torch.zeros_like(p) for n, p in params.items()}
        n_ex = _num_examples(inputs) if labels.numel() else 0
        norms = []
        mb = self.cfg.microbatch_size
        for start in range(0, n_ex, mb):
            sl = slice(start, start + mb)
            g = per_example_grads(
                self.model,
                _index(inputs, sl),
                labels[sl],
                None if extras is None else _index(extras, sl),
                self.per_example_loss,
                self.logits_fn,
                self.randomness,
                params,
            )
            clipped, nrm = clip_per_example(g, self.cfg.max_grad_norm)
            norms.append(nrm)
            for n in summed:
                summed[n] += clipped[n].sum(0)
        self.last_norms = torch.cat(norms) if norms else torch.empty(0)
        return privatize(summed, self.cfg, self.noise_gen)

    def step(self, inputs: Inputs, labels: Tensor, extras=None) -> Dict[str, float]:
        grads = self.privatized_grads(inputs, labels, extras)
        for n, p in trainable_parameters(self.model).items():
            p.grad = grads[n].to(p.dtype)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accountant.step(noise_multiplier=self.cfg.noise_multiplier, sample_rate=self.cfg.sample_rate)
        self.steps += 1
        nrm = self.last_norms
        return {
            "batch_size": int(labels.numel()),
            "mean_grad_norm": float(nrm.mean()) if nrm.numel() else math.nan,
            "frac_clipped": float((nrm > self.cfg.max_grad_norm).float().mean()) if nrm.numel() else math.nan,
        }

    def epsilon(self, delta: float) -> float:
        return float(self.accountant.get_epsilon(delta=delta))


def noise_multiplier_for(target_epsilon: float, delta: float, sample_rate: float, steps: int, accountant: str = "rdp") -> float:
    """Calibrate sigma for a target (eps, delta) with Opacus' search."""
    from opacus.accountants.utils import get_noise_multiplier

    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=delta, sample_rate=sample_rate, steps=steps, accountant=accountant
        )
    )
