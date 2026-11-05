"""Experimental bias-only (BiTFiT-like) AugMult GSM: biases everywhere + classifier."""

import torch

from legacy_helpers import flatten_views, multiview_batch, tiny_bert
from test_grad_samplers import ATOL, RTOL, reference_per_example_grads

from augmult.augmult_bias_grad_sample_module import GradSampleModuleAugMultBias
from util.privacy_engine_util import prepare_gradsamplers


def _bias_model():
    m = tiny_bert(0)
    for n, p in m.named_parameters():
        p.requires_grad = ("bias" in n) or ("classifier" in n)
    return m


def test_bias_only_gsm_matches_reference():
    k = 2
    batch = multiview_batch(3, k, seed=5)
    ref = reference_per_example_grads(_bias_model(), batch, k)
    m = _bias_model()
    gsm = GradSampleModuleAugMultBias(m)
    prepare_gradsamplers(k, gsm)
    gsm(**flatten_views(batch, k)).loss.backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad_sample is not None, n
            torch.testing.assert_close(p.grad_sample, ref[n], atol=ATOL, rtol=RTOL, msg=n)
