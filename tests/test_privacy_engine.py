"""End-to-end Opacus integration of the AugMult grad_sample_mode, noise parameterization,
and empty-batch handling."""

import types

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from conftest import SEQ_LEN, multiview_batch, tiny_bert

from util.privacy_engine_util import (
    _prepare_model_modified,
    add_noise,
    empty_batch_handling,
    prepare_gradsamplers,
)


class MultiViewTensorDataset(Dataset):
    """Items are already-tokenized multi-view samples: {input_ids: [K, T], ..., label}."""

    def __init__(self, n, k, seed=0):
        b = multiview_batch(n, k, seed=seed)
        self.items = [
            {
                "input_ids": b["input_ids"][i],
                "attention_mask": b["attention_mask"][i],
                "token_type_ids": b["token_type_ids"][i],
                "label": int(b["labels"][i]),
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def stack_collate(batch):
    out = {k: torch.stack([s[k] for s in batch]) for k in ("input_ids", "attention_mask", "token_type_ids")}
    out["labels"] = torch.tensor([s["label"] for s in batch])
    return out


def make_private_augmult(model, loader, k, *, lr, C, sigma, poisson):
    from opacus import PrivacyEngine

    engine = PrivacyEngine()
    engine._prepare_model = types.MethodType(_prepare_model_modified, engine)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    dp_model, dp_opt, dp_loader = engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=C,
        grad_sample_mode="augmult",
        poisson_sampling=poisson,
    )
    dp_opt.add_noise = types.MethodType(add_noise, dp_opt)
    empty_batch_handling(mv_train_loader=loader, dp_train_loader=dp_loader, mv_collate=stack_collate)
    dp_model = prepare_gradsamplers(k, dp_model)
    return engine, dp_model, dp_opt, dp_loader


def flatten(batch, k):
    out = {}
    for key, v in batch.items():
        out[key] = torch.repeat_interleave(v, k, dim=0) if key == "labels" else v.reshape(-1, v.size(-1))
    return out


@pytest.mark.parametrize("k", [1, 3])
def test_single_example_step_is_clipped(k):
    """Batch of one example, sigma=0, tiny C: the update direction has norm exactly 1
    under the 1/C parameterization, i.e. the clipped per-example grad has norm C."""
    C = 1e-4
    model = tiny_bert(seed=0)
    loader = DataLoader(MultiViewTensorDataset(4, k), batch_size=1, collate_fn=stack_collate)
    _, dp_model, dp_opt, dp_loader = make_private_augmult(model, loader, k, lr=0.0, C=C, sigma=0.0, poisson=False)

    batch = next(iter(dp_loader))
    dp_opt.zero_grad(set_to_none=True)
    dp_model(**flatten(batch, k)).loss.backward()
    raw_norm = torch.sqrt(sum((p.grad_sample.flatten(1) ** 2).sum(1) for p in dp_opt.params))
    assert raw_norm.item() > 10 * C  # clipping is active
    dp_opt.step()
    total = torch.sqrt(sum((p.grad**2).sum() for p in dp_opt.params))
    # p.grad = clip_C(g)/C / expected_batch_size, expected_batch_size == 1
    assert total.item() == pytest.approx(1.0, rel=1e-3)
    assert total.item() <= 1.0 + 1e-4


def test_poisson_batch_memory_manager_step_runs():
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    k, C = 2, 0.5
    model = tiny_bert(seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    loader = DataLoader(MultiViewTensorDataset(16, k), batch_size=8, collate_fn=stack_collate)
    engine, dp_model, dp_opt, dp_loader = make_private_augmult(model, loader, k, lr=0.1, C=C, sigma=1.0, poisson=True)

    steps = 0
    with BatchMemoryManager(data_loader=dp_loader, max_physical_batch_size=4 // k, optimizer=dp_opt) as mem_loader:
        for batch in mem_loader:
            dp_opt.zero_grad(set_to_none=True)
            dp_model(**flatten(batch, k)).loss.backward()
            for p in dp_opt.params:  # per-example (not per-view) grads
                assert p.grad_sample.shape[0] == batch["labels"].shape[0]
            dp_opt.step()
            steps += 1
    assert steps > 0
    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert changed
    assert engine.accountant.get_epsilon(delta=1e-5) > 0


def test_add_noise_equals_standard_dpsgd_with_lr_over_C():
    """Patched add_noise (noise independent of C) with lr == standard DP-SGD with lr/C."""
    from opacus.optimizers import DPOptimizer

    torch.manual_seed(0)
    lr, C, sigma, B = 0.7, 0.05, 1.3, 4
    lin_a, lin_b = torch.nn.Linear(5, 3), torch.nn.Linear(5, 3)
    lin_b.load_state_dict(lin_a.state_dict())
    gs = {n: torch.randn(B, *p.shape) for n, p in lin_a.named_parameters()}

    def run(lin, lr_, patched):
        opt = DPOptimizer(
            torch.optim.SGD(lin.parameters(), lr=lr_),
            noise_multiplier=sigma,
            max_grad_norm=C,
            expected_batch_size=B,
            generator=torch.Generator().manual_seed(123),
        )
        if patched:
            opt.add_noise = types.MethodType(add_noise, opt)
        for n, p in lin.named_parameters():
            p.grad_sample = gs[n].clone()
        opt.step()

    run(lin_a, lr, patched=True)
    run(lin_b, lr / C, patched=False)
    for (n, pa), (_, pb) in zip(lin_a.named_parameters(), lin_b.named_parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-5, rtol=1e-5, msg=n)


# ---------------------------------------------------------------- empty batches


class RawTextDataset(Dataset):
    """Raw string rows, like the GLUE/MedNLI datasets fed to the multi-view collate."""

    def __init__(self):
        self.rows = [{"sentence": "a b c", "label": 1}, {"sentence": "d e", "label": 0}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def fake_mv_collate(batch, k=3):
    n = len(batch)
    return {
        "input_ids": torch.ones(n, k, SEQ_LEN, dtype=torch.long),
        "attention_mask": torch.ones(n, k, SEQ_LEN, dtype=torch.long),
        "token_type_ids": torch.zeros(n, k, SEQ_LEN, dtype=torch.long),
        "labels": torch.tensor([r["label"] for r in batch]),
    }


def test_empty_batch_shapes_come_from_collated_output():
    loader = DataLoader(RawTextDataset(), batch_size=2, collate_fn=fake_mv_collate)
    dp_loader = types.SimpleNamespace(collate_fn=None)
    empty_batch_handling(mv_train_loader=loader, dp_train_loader=dp_loader, mv_collate=fake_mv_collate)

    empty = dp_loader.collate_fn([])
    assert set(empty) == {"input_ids", "attention_mask", "token_type_ids", "labels"}
    assert empty["input_ids"].shape == (0, 3, SEQ_LEN)
    assert empty["input_ids"].dtype == torch.long
    assert empty["labels"].shape == (0,)
    # train.py flattening must work on the empty batch
    assert empty["input_ids"].view(-1, empty["input_ids"].size(-1)).shape == (0, SEQ_LEN)
    # non-empty batches are passed through unchanged
    full = dp_loader.collate_fn([RawTextDataset()[0]])
    assert full["input_ids"].shape == (1, 3, SEQ_LEN)


def test_empty_batch_with_real_multiview_collate(tmp_path):
    """Integration with augmult.data's tokenizing multi-view collate (offline tokenizer)."""
    pytest.importorskip("datasets")
    from transformers import BertTokenizer

    from augmult.data import init_mv_collate

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "d", "e"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizer(str(tmp_path / "vocab.txt"))
    transforms = [lambda x: x, lambda x: x]  # K = 2
    collate = init_mv_collate(tok, transforms, max_length=16)

    loader = DataLoader(RawTextDataset(), batch_size=2, collate_fn=collate)
    dp_loader = types.SimpleNamespace(collate_fn=None)
    empty_batch_handling(mv_train_loader=loader, dp_train_loader=dp_loader, mv_collate=collate)
    empty = dp_loader.collate_fn([])
    assert empty["input_ids"].shape == (0, 2, 16)
    assert empty["labels"].shape == (0,)
