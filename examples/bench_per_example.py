"""Speed: vmap per-example AugMult grads vs a naive per-example loop (offline, random bert-tiny-sized model).

    python examples/bench_per_example.py [--n 32] [--k 4] [--len 64] [--mode full]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import BertConfig  # noqa: E402

from dpaug import mean_view_ce, per_example_grads  # noqa: E402
from dpaug.augmult import trainable_parameters  # noqa: E402
from dpaug.models import load_seq_classifier  # noqa: E402


def naive(model, x, y):
    named = trainable_parameters(model)
    outs = []
    for i in range(y.shape[0]):
        model.zero_grad(set_to_none=True)
        mean_view_ce(model(**{k: v[i] for k, v in x.items()}).logits, y[i]).backward()
        outs.append([p.grad.clone() for p in named.values()])
    return outs


def timeit(fn, reps=3):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)  # min is the most robust estimate on a shared/contended machine


def compare(fa, fb, rounds=3):
    """Alternate A/B timing rounds so background load hits both equally; return per-fn minima."""
    a, b = [], []
    for _ in range(rounds):
        a.append(timeit(fa, reps=1))
        b.append(timeit(fb, reps=1))
    return min(a), min(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--freeze-embeddings", action="store_true", help="common in DP fine-tuning; avoids N copies of the vocab matrix")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512)  # bert-tiny shape
    model = load_seq_classifier(cfg, 2, mode=args.mode).train()
    if args.freeze_embeddings:
        for n, p in model.named_parameters():
            if "embeddings" in n:
                p.requires_grad = False
    x = {
        "input_ids": torch.randint(1000, 30000, (args.n, args.k, args.len)),
        "attention_mask": torch.ones(args.n, args.k, args.len, dtype=torch.long),
    }
    y = torch.randint(0, 2, (args.n,))
    t_vmap, t_loop = compare(lambda: per_example_grads(model, x, y), lambda: naive(model, x, y))
    print(
        f"threads={torch.get_num_threads()} N={args.n} K={args.k} L={args.len} mode={args.mode} "
        f"freeze_emb={args.freeze_embeddings}: "
        f"vmap {t_vmap:.3f}s | naive loop {t_loop:.3f}s | speedup x{t_loop / t_vmap:.2f}"
    )


if __name__ == "__main__":
    main()
