"""Correctness of the AugMult per-example gradients.

The AugMult per-example gradient of example i must equal the gradient of
    L_i(theta) = (1/K) * sum_k CE(f_theta(view_{i,k}), y_i)
i.e. the gradient averaged over the K views, computed *before* clipping.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad

from conftest import flatten_views, multiview_batch, tiny_bert

from augmult.augmult_grad_sample_module import GradSampleModuleAugMult
from util.privacy_engine_util import prepare_gradsamplers

N = 3
ATOL = 1e-5
RTOL = 1e-4


def reference_per_example_grads(model, batch, k):
    """torch.func reference: grad of the mean-over-K-views loss, one example at a time."""
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}

    def loss_fn(prm, ids, mask, tt, y):
        out = functional_call(
            model, prm, args=(), kwargs={"input_ids": ids, "attention_mask": mask, "token_type_ids": tt}
        )
        return F.cross_entropy(out.logits, y.repeat(k))

    g = grad(loss_fn)
    per_ex = {n: [] for n in params}
    for i in range(batch["labels"].shape[0]):
        gi = g(
            params,
            batch["input_ids"][i],
            batch["attention_mask"][i],
            batch["token_type_ids"][i],
            batch["labels"][i : i + 1],
        )
        for n in params:
            per_ex[n].append(gi[n])
    return {n: torch.stack(v) for n, v in per_ex.items()}


def augmult_per_example_grads(model, batch, k):
    gsm = GradSampleModuleAugMult(model, batch_first=True, loss_reduction="mean")
    prepare_gradsamplers(k, gsm)
    out = gsm(**flatten_views(batch, k))
    out.loss.backward()
    res = {n: p.grad_sample.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    gsm.zero_grad()
    return res


@pytest.mark.parametrize("trainable", ["classifier_and_pooler", "last_output_block"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_augmult_matches_torch_func(trainable, k):
    model = tiny_bert(seed=0, trainable=trainable)
    batch = multiview_batch(N, k, seed=k)
    ref = reference_per_example_grads(model, batch, k)
    got = augmult_per_example_grads(model, batch, k)
    assert set(ref) == set(got)
    for name in ref:
        assert got[name].shape == ref[name].shape, name
        torch.testing.assert_close(got[name], ref[name], atol=ATOL, rtol=RTOL, msg=name)


@pytest.mark.parametrize("trainable", ["classifier_and_pooler", "last_output_block"])
def test_k1_matches_vanilla_opacus(trainable):
    from opacus import GradSampleModule

    batch = multiview_batch(N, 1, seed=7)
    flat = flatten_views(batch, 1)

    m1 = tiny_bert(seed=0, trainable=trainable)
    got = augmult_per_example_grads(m1, batch, 1)

    m2 = tiny_bert(seed=0, trainable=trainable)
    vanilla = GradSampleModule(m2, batch_first=True, loss_reduction="mean")
    vanilla(**flat).loss.backward()
    for name, p in m2.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(got[name], p.grad_sample, atol=ATOL, rtol=RTOL, msg=name)


def test_samplers_are_per_instance():
    """prepare_gradsamplers must not leak K into other GSM instances (class-level dict)."""
    m_a, m_b = tiny_bert(0), tiny_bert(0)
    gsm_a = GradSampleModuleAugMult(m_a)
    gsm_b = GradSampleModuleAugMult(m_b)
    prepare_gradsamplers(2, gsm_a)
    prepare_gradsamplers(4, gsm_b)
    batch = multiview_batch(N, 2, seed=3)
    gsm_a(**flatten_views(batch, 2)).loss.backward()
    ref = reference_per_example_grads(tiny_bert(0), batch, 2)
    for name, p in m_a.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad_sample, ref[name], atol=ATOL, rtol=RTOL, msg=name)
