import math

import pytest
import torch
from torch import nn

from dpaug_helpers import make_inputs, make_model
from dpaug import (
    DPAugMult,
    DPConfig,
    PoissonSampler,
    clip_per_example,
    mean_view_ce,
    mean_view_ce_with_consistency,
    mean_view_distillation,
    per_example_grads,
    privatize,
)
from dpaug.augmult import noise_multiplier_for, trainable_parameters
from dpaug.masks import prepare_inputs

ATOL = 1e-5


def naive_per_example_grads(model, inputs, labels, loss_fn=mean_view_ce):
    """Reference: one ordinary forward/backward per example over its K views (2D HF masks)."""
    named = trainable_parameters(model)
    out = {n: [] for n in named}
    for i in range(labels.shape[0]):
        model.zero_grad(set_to_none=True)
        logits = model(**{k: v[i] for k, v in inputs.items()}).logits
        loss_fn(logits, labels[i], None).backward()
        for n, p in named.items():
            out[n].append(p.grad.detach().clone())
    return {n: torch.stack(v) for n, v in out.items()}


def assert_grads_close(a, b, atol=ATOL):
    assert a.keys() == b.keys()
    for n in a:
        torch.testing.assert_close(a[n], b[n], atol=atol, rtol=1e-4, msg=n)


# ---------------------------------------------------------------- masks
@pytest.mark.parametrize("arch", ["bert", "llama", "qwen2"])
def test_expanded_mask_matches_hf_2d_mask(arch):
    model = make_model(arch, "full").eval()
    x = make_inputs(3, 2)
    flat = {k: v.reshape(6, -1) for k, v in x.items()}
    ref = model(**flat).logits
    x4 = prepare_inputs(x, model)
    got = model(input_ids=flat["input_ids"], attention_mask=x4["attention_mask"].reshape(6, 1, 10, 10)).logits
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- (1) vmap == naive loop
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("arch,mode", [("bert", "full"), ("bert", "head"), ("bert", "lora"), ("llama", "lora"), ("qwen2", "full")])
def test_vmap_matches_naive_loop(arch, mode, k):
    model = make_model(arch, mode).eval()  # eval: no dropout -> deterministic comparison
    x = make_inputs(4, k)
    y = torch.tensor([0, 1, 2, 1])
    assert_grads_close(per_example_grads(model, x, y), naive_per_example_grads(model, x, y))


def test_vmap_matches_naive_loop_custom_objectives():
    model = make_model("bert", "full").eval()
    x = make_inputs(3, 3)
    y = torch.tensor([0, 1, 2])
    loss = mean_view_ce_with_consistency(0.7)
    assert_grads_close(per_example_grads(model, x, y, per_example_loss=loss), naive_per_example_grads(model, x, y, loss))
    # distillation with per-example extras (K, C) teacher logits
    teacher = torch.randn(3, 3, 3)
    kd = mean_view_distillation(alpha=0.5, temperature=2.0)
    got = per_example_grads(model, x, y, extras={"teacher_logits": teacher}, per_example_loss=kd)
    named = trainable_parameters(model)
    for i in range(3):
        model.zero_grad(set_to_none=True)
        kd(model(**{k: v[i] for k, v in x.items()}).logits, y[i], {"teacher_logits": teacher[i]}).backward()
        for n, p in named.items():
            torch.testing.assert_close(got[n][i], p.grad, atol=ATOL, rtol=1e-4)


# ---------------------------------------------------------------- (2) K=1 == standard DP-SGD per-sample grads (Opacus)
class ToyTokenClassifier(nn.Module):
    """Embedding -> Linear -> LayerNorm -> mean-pool -> Linear: all layers Opacus supports."""

    def __init__(self, vocab=50, d=16, c=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lin = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, c)

    def forward(self, ids):
        return self.out(self.ln(torch.tanh(self.lin(self.emb(ids)))).mean(-2))


def test_k1_matches_opacus_per_sample_grads():
    from opacus import GradSampleModule

    torch.manual_seed(0)
    model = ToyTokenClassifier()
    ids = torch.randint(0, 50, (6, 1, 7))
    y = torch.randint(0, 3, (6,))
    ours = per_example_grads(model, ids, y)  # tensor inputs (N, K=1, T)

    gsm = GradSampleModule(model, loss_reduction="sum")
    loss = nn.functional.cross_entropy(gsm(ids[:, 0]), y, reduction="sum")
    loss.backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(ours[n], p.grad_sample, atol=ATOL, rtol=1e-4, msg=n)
    gsm._close()


def test_augmult_is_mean_of_view_grads():
    """K views of the mean-view objective == average of K single-view per-sample grads."""
    torch.manual_seed(0)
    model = ToyTokenClassifier()
    ids = torch.randint(0, 50, (4, 3, 7))
    y = torch.randint(0, 3, (4,))
    multi = per_example_grads(model, ids, y)
    singles = [per_example_grads(model, ids[:, k : k + 1], y) for k in range(3)]
    for n in multi:
        torch.testing.assert_close(multi[n], torch.stack([s[n] for s in singles]).mean(0), atol=ATOL, rtol=1e-4)


# ---------------------------------------------------------------- dropout randomness
def test_dropout_views_differ_across_examples_and_views():
    model = make_model("bert", "full").train()
    x = make_inputs(1, 2, n_pad=0)
    x = {k: v.repeat(2, 1, 1) for k, v in x.items()}  # two identical examples
    y = torch.tensor([1, 1])
    g = per_example_grads(model, x, y)
    w = "classifier.weight"
    assert not torch.allclose(g[w][0], g[w][1]), "randomness='different' should give per-example dropout masks"


# ---------------------------------------------------------------- (3) clipping bound
def test_clip_per_example_bound():
    model = make_model("bert", "full").eval()
    g = per_example_grads(model, make_inputs(5, 2), torch.tensor([0, 1, 2, 0, 1]))
    c = 1e-3
    clipped, norms = clip_per_example(g, c)
    _, clipped_norms = clip_per_example(clipped, float("inf"))
    assert (norms > c).all()  # tiny C: everything clipped
    assert (clipped_norms <= c * (1 + 1e-4)).all()
    # unclipped examples untouched with large C
    same, _ = clip_per_example(g, 1e9)
    for n in g:
        torch.testing.assert_close(same[n], g[n])


def test_privatized_sum_respects_sensitivity_and_param():
    model = make_model("bert", "full").eval()
    x, y = make_inputs(6, 2), torch.tensor([0, 1, 2, 0, 1, 2])
    cfg = DPConfig(max_grad_norm=1e-2, noise_multiplier=0.0, sample_rate=0.5, dataset_size=12, microbatch_size=4)
    dp = DPAugMult(model, torch.optim.SGD(trainable_parameters(model).values(), lr=0.0), cfg)
    g = dp.privatized_grads(x, y)
    total = math.sqrt(sum(v.pow(2).sum().item() for v in g.values())) * cfg.expected_batch_size
    assert total <= 6 * cfg.max_grad_norm * (1 + 1e-4)
    # thesis parameterisation == standard / C
    cfg_t = DPConfig(**{**cfg.__dict__, "noise_independent_of_C": True})
    g_t = DPAugMult(model, torch.optim.SGD(trainable_parameters(model).values(), lr=0.0), cfg_t).privatized_grads(x, y)
    for n in g:
        torch.testing.assert_close(g_t[n], g[n] / cfg.max_grad_norm, atol=1e-5, rtol=1e-4)


def test_microbatching_is_invariant():
    model = make_model("bert", "full").eval()
    x, y = make_inputs(7, 2), torch.tensor([0, 1, 2, 0, 1, 2, 0])
    res = []
    for mb in (2, 64):
        cfg = DPConfig(max_grad_norm=0.1, noise_multiplier=0.0, sample_rate=0.5, dataset_size=14, microbatch_size=mb)
        res.append(DPAugMult(model, torch.optim.SGD(trainable_parameters(model).values(), lr=0.0), cfg).privatized_grads(x, y))
    for n in res[0]:
        torch.testing.assert_close(res[0][n], res[1][n], atol=1e-6, rtol=1e-5)


def test_noise_std():
    cfg = DPConfig(max_grad_norm=0.5, noise_multiplier=2.0, sample_rate=0.1, dataset_size=100)
    z = {"w": torch.zeros(200_000)}
    out = privatize(z, cfg, torch.Generator().manual_seed(0))["w"] * cfg.expected_batch_size
    assert abs(out.std().item() - 1.0) < 0.01  # sigma*C = 1.0
    cfg_t = DPConfig(**{**cfg.__dict__, "noise_independent_of_C": True})
    out_t = privatize(z, cfg_t, torch.Generator().manual_seed(0))["w"] * cfg.expected_batch_size
    assert abs(out_t.std().item() - 2.0) < 0.02  # sigma


def test_empty_poisson_batch_is_noise_only():
    model = make_model("bert", "head").eval()
    cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=1.0, sample_rate=0.01, dataset_size=10, seed=0)
    dp = DPAugMult(model, torch.optim.SGD(trainable_parameters(model).values(), lr=0.1), cfg)
    x = make_inputs(0, 2)
    stats = dp.step(x, torch.zeros(0, dtype=torch.long))
    assert stats["batch_size"] == 0 and dp.steps == 1


# ---------------------------------------------------------------- (4) accounting
@pytest.mark.parametrize("mech", ["rdp", "prv"])
def test_accounting_matches_opacus(mech):
    from opacus.accountants import create_accountant

    model = make_model("bert", "head").eval()
    n, q, sigma, steps, delta = 40, 0.25, 1.1, 5, 1e-5
    cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=sigma, sample_rate=q, dataset_size=n, accountant=mech, seed=0)
    dp = DPAugMult(model, torch.optim.SGD(trainable_parameters(model).values(), lr=0.1), cfg)
    x, y = make_inputs(n, 2), torch.randint(0, 3, (n,))
    for idx in PoissonSampler(n, q, steps, torch.Generator().manual_seed(0)):
        dp.step({k: v[idx] for k, v in x.items()}, y[idx])
    ref = create_accountant(mech)
    for _ in range(steps):
        ref.step(noise_multiplier=sigma, sample_rate=q)
    assert dp.epsilon(delta) == pytest.approx(ref.get_epsilon(delta=delta), rel=1e-6)


def test_noise_multiplier_for_target():
    from opacus.accountants import RDPAccountant

    sigma = noise_multiplier_for(3.0, 1e-5, 0.05, 200)
    acc = RDPAccountant()
    for _ in range(200):
        acc.step(noise_multiplier=sigma, sample_rate=0.05)
    assert acc.get_epsilon(1e-5) <= 3.0 + 1e-3


def test_poisson_sampler_seeded():
    a = [t.tolist() for t in PoissonSampler(1000, 0.1, 5, torch.Generator().manual_seed(1))]
    b = [t.tolist() for t in PoissonSampler(1000, 0.1, 5, torch.Generator().manual_seed(1))]
    assert a == b and 50 < len(a[0]) < 150


# ---------------------------------------------------------------- (5) LoRA path
@pytest.mark.parametrize("arch", ["bert", "llama", "qwen2"])
def test_lora_only_trains_adapters_and_head(arch):
    model = make_model(arch, "lora").train()
    g = per_example_grads(model, make_inputs(3, 2), torch.tensor([0, 1, 2]))
    assert set(g) == set(trainable_parameters(model))
    assert all(("lora_" in n) or ("modules_to_save" in n) for n in g), list(g)
    assert any("lora_" in n for n in g)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=1.0, sample_rate=0.5, dataset_size=6, seed=0)
    dp = DPAugMult(model, torch.optim.Adam(trainable_parameters(model).values(), lr=1e-2), cfg)
    dp.step(make_inputs(3, 2), torch.tensor([0, 1, 2]))
    for n, p in model.named_parameters():
        changed = not torch.equal(before[n], p.detach())
        assert changed == p.requires_grad, n


# ---------------------------------------------------------------- feature cache
def test_cache_and_head_match_full_model():
    from dpaug.cache import BertHead, cache_bert_cls

    model = make_model("bert", "head").eval()
    x = make_inputs(4, 3)
    feats = cache_bert_cls(model, x)
    assert feats.shape == (4, 3, 32)
    head = BertHead(model).eval()
    flat = {k: v.reshape(12, -1) for k, v in x.items()}
    torch.testing.assert_close(head(feats).reshape(12, -1), model(**flat).logits, atol=1e-5, rtol=1e-4)
    # dropout views: K' = K * (1 + m)
    model.train()
    assert cache_bert_cls(model, x, stochastic_passes=2).shape == (4, 9, 32)
    # DP-train the head on cached features (tensor inputs)
    head.train()
    cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=1.0, sample_rate=0.5, dataset_size=8, seed=0)
    dp = DPAugMult(head, torch.optim.SGD(head.parameters(), lr=0.1), cfg)
    stats = dp.step(feats, torch.tensor([0, 1, 2, 0]))
    assert stats["batch_size"] == 4
