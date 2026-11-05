"""Smoke test: a few DP-SGD + AugMult steps on 64 SST-2 examples with prajjwal1/bert-tiny.

K=2 views = identity text twice; with the model in train mode each view gets its own
dropout mask ("dropout views"). Skips gracefully if the HF hub is unreachable.

    python examples/sst2_smoke.py [--mode full|head|bitfit|lora] [--k 2] [--steps 8]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpaug import DPAugMult, DPConfig, PoissonSampler  # noqa: E402
from dpaug.augmult import noise_multiplier_for, trainable_parameters  # noqa: E402
from dpaug.models import count_params, load_seq_classifier  # noqa: E402
from dpaug.views import identity, tokenize_views  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "head", "bitfit", "lora"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        from transformers import BertForSequenceClassification, BertTokenizerFast

        tok = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
        # bert-tiny's config.json lacks `model_type`, so AutoModel cannot resolve it in transformers>=5.
        model = load_seq_classifier(
            "prajjwal1/bert-tiny", 2, mode=args.mode, model_cls=BertForSequenceClassification
        )
        train = load_dataset("nyu-mll/glue", "sst2", split=f"train[:{args.n}]")
        val = load_dataset("nyu-mll/glue", "sst2", split="validation[:128]")
    except Exception as e:  # offline / hub unavailable
        print(f"SKIP: could not load model/data ({type(e).__name__}: {e})")
        return 0

    torch.manual_seed(0)
    x = tokenize_views(tok, list(train["sentence"]), [identity] * args.k, max_length=64)
    y = torch.tensor(list(train["label"]))
    n = len(y)
    delta = 1.0 / n**1.1
    sigma = noise_multiplier_for(args.eps, delta, args.q, args.steps)
    cfg = DPConfig(max_grad_norm=args.clip, noise_multiplier=sigma, sample_rate=args.q, dataset_size=n, microbatch_size=16, seed=0)
    total, trainable = count_params(model)
    print(f"mode={args.mode} trainable={trainable}/{total} K={args.k} n={n} q={args.q} sigma={sigma:.3f} delta={delta:.2e}")

    opt = torch.optim.Adam(trainable_parameters(model).values(), lr=args.lr)
    dp = DPAugMult(model, opt, cfg)
    model.train()
    t0 = time.time()
    for step, idx in enumerate(PoissonSampler(n, args.q, args.steps, torch.Generator().manual_seed(0))):
        stats = dp.step({k: v[idx] for k, v in x.items()}, y[idx])
        print(f"step {step:2d} B={stats['batch_size']:3d} mean|g|={stats['mean_grad_norm']:.3f} clipped={stats['frac_clipped']:.2f}")
    dt = time.time() - t0

    model.eval()
    with torch.no_grad():
        enc = tok(list(val["sentence"]), padding=True, truncation=True, max_length=64, return_tensors="pt")
        acc = (model(**enc).logits.argmax(-1) == torch.tensor(list(val["label"]))).float().mean().item()
    print(f"done in {dt:.1f}s | epsilon={dp.epsilon(delta):.3f} (delta={delta:.1e}, {cfg.accountant}) | val acc (128 ex)={acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
