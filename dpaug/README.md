# dpaug — functional AugMult for DP-SGD (prototype)

Opacus-fork-free re-implementation of the thesis' Augmentation Multiplicity. It works for HF encoders and decoder LLMs, with full, head-only, BitFiT or LoRA fine-tuning.

```python
from dpaug import DPAugMult, DPConfig, PoissonSampler, mean_view_ce
from dpaug.augmult import noise_multiplier_for, trainable_parameters
from dpaug.models import load_seq_classifier
from dpaug.views import tokenize_views, identity

model = load_seq_classifier("Qwen/Qwen3-0.6B", num_labels=3, mode="lora")      # or BERT, mode="head"/"full"/"bitfit"
x = tokenize_views(tok, premises, [identity, paraphrase_fn, bt_fn], texts_b=hypotheses)  # (N, K, L) tensors
sigma = noise_multiplier_for(target_epsilon=8, delta=1/n**1.1, sample_rate=q, steps=T)
dp = DPAugMult(model, torch.optim.AdamW(trainable_parameters(model).values(), 1e-3),
               DPConfig(max_grad_norm=1.0, noise_multiplier=sigma, sample_rate=q, dataset_size=n, seed=0),
               per_example_loss=mean_view_ce)                                   # pluggable per-example objective
for idx in PoissonSampler(n, q, T, torch.Generator().manual_seed(0)):
    dp.step({k: v[idx] for k, v in x.items()}, y[idx])
print(dp.epsilon(delta))
```

| Module | What it does |
|---|---|
| `augmult.py` | `per_example_grads` (`vmap(grad(·))` over examples; each example's K views are one inner batch), `clip_per_example` (flat clipping), `privatize` (Gaussian noise; `noise_independent_of_C=True` gives the thesis parameterisation), `PoissonSampler`, and `DPAugMult` (microbatched step with an Opacus RDP/PRV/GDP accountant). |
| `losses.py` | Per-example objectives over `(K, C)` logits: `mean_view_ce` (the thesis), `ce_of_mean_logits`, consistency-regularised CE, and distillation from per-view teacher logits passed via `extras`. |
| `masks.py` | Expands `(N, K, L)` padding masks to 4D additive masks before `vmap`, because the transformers-5 mask utilities use data-dependent control flow. Encoders get a bidirectional mask; decoders get a causal one. |
| `models.py` | `load_seq_classifier(name_or_config, num_labels, mode)` for any `*ForSequenceClassification` (BERT/RoBERTa/…/Llama/Qwen/Gemma). |
| `cache.py` | Cached `[CLS]` features plus `BertHead` for the head-only regime (≈100× faster per step). **Encoder dropout is gone** unless you add `stochastic_passes`. |
| `ghost_norm.py` | Reference per-example Linear grad norms over K views via the Gram trick on the concatenated `K·T` axis. This is the seed of ghost-clipping AugMult. |

Tests: `pytest tests/dpaug` (CPU, offline, tiny random models). Examples: `examples/sst2_smoke.py`, `examples/bench_per_example.py`.

Known limitations:
- Per-example grads are materialised, one microbatch at a time. With trainable embeddings, that means N copies of the vocabulary matrix, so freeze embeddings or use LoRA for big models.
- Sliding-window layers get a full mask.
- ModernBERT's unpadding path and multi-label heads are untested.
- No mixed precision, distributed or GPU benchmarks yet.
