# `views/` — offline view generation for AugMult

Precomputes K augmented **views** per training example with local models. The views go into a versioned Parquet store, together with provenance and label-preservation scores. The training code can then read K views per example for AugMult. The augmentation pipeline stays out of the training loop, so the views are cheap, reproducible and auditable.

**Everything runs locally.** No hosted LLM APIs are called. This matters for credentialed data such as MedNLI under the PhysioNet DUA. Generated stores land in `outputs/`, which is git-ignored, because they contain dataset text.

```bash
uv pip install -e ".[views]"            # from the repo root
python -m views.generate --config configs/views/sst2_qwen.yaml --limit 100
python -m views.filter --input outputs/views/<run> --k 8 --min-entail 0.8 --min-sim 0.7
python -m pytest tests/views -q          # offline unit + end-to-end tests (no downloads)
```

## Pieces

| Module | What it does |
|---|---|
| `generators.py` | `identity` (always view 0), `rule` (AEDA punctuation insertion, word dropout, adjacent swap; no deps), `llm_paraphrase` (local instruct LLM via transformers, or vLLM if installed with CUDA; chat template; temperature/nucleus sampling; dedupe), `backtranslation` (NLLB-200 or Opus-MT pivots), `mock` (tests only). |
| `scorers.py` | `lexical` (word edit distance, length ratio), `similarity` (sentence-transformers cosine), `nli` (bidirectional entailment original↔view with a public cross-encoder), `task` (optional public task classifier: label-free agreement, and P(own label) only if `use_label: true`). |
| `generate.py` / `filter.py` | CLIs. The filter keeps view 0 plus the passing views and emits exactly K views per example, padding by resampling (falling back to identity copies). Padding is recorded per row and in the manifest. |
| `store.py` / `schema.py` | Parquet schema with columns `example_id, split, view_id, method, generator_id, text_a, text_b, augmented_fields, label, created_at, score_*`. Filtered stores add `source_view_id, padded, pad_source`. `load_grouped(path, k)` is the training-side loader. |

Each store directory holds `views.parquet` and `manifest.json`. The manifest records:
- the resolved config and its hash;
- the dataset id and revision, or a local fingerprint;
- every generator's provenance: model id, resolved commit sha, decoding parameters, seed, prompt-template sha256 and backend;
- each scorer's model and revision, with `saw_label`;
- counts and any shortfall of views;
- timing;
- per-method score quantiles;
- a **label-flip-rate estimate**;
- the git commit and package versions.

## Privacy contract

- **Generators:** they see one text field of one example at a time. A generator never sees the label or the other side of a sentence pair. In NLI terms, a rewritten hypothesis cannot "know" the premise or the answer.
- **Scorers:**
  - They use public models only.
  - They see one example at a time, and only that example's own label, and only when `saw_label` is set.
  - Anything computed this way (views, scores, filters, weights) happens per example, *before* clipping. The downstream model's DP-SGD guarantee therefore still holds.
- **Not covered:**
  - **Filter thresholds** are global hyperparameters. Fix them a priori or tune them on public data. Tuning them on private score distributions is unaccounted data-dependent tuning.
  - The **views themselves are private data**. Treat them like the raw data and never commit them.
  - A **task classifier counts as "public" only if it was not trained on the private set**. The SST-2 example uses an SST-2-finetuned model, which is fine only because SST-2 is public.

## The distillation confound (why provenance is per row)

LLM- or MT-generated views inject the generator's distribution into training. A model trained with AugMult on LLM paraphrases is partly **distilled from the generator**, through its wording, its errors and its world knowledge. Any gain could therefore come from "more public knowledge" rather than from "AugMult reduces clipping variance". This is the same issue Tramèr, Kamath and Carlini (ICML 2024) raise for public pretraining.

The store is built so this can be measured:

- `generator_id` and `method` on every row let you ablate by generator (rule vs MT vs LLM, small vs large LLM) at a fixed K.
- **The label-flip estimate** (`summary.label_flip_rate_estimate`) quantifies the *inherent error* each generator injects. On the 5-sentence smoke test:

  | Generator | Fraction with bidirectional entailment < 0.5 |
  |---|---|
  | SmolLM2-135M | 0.90 |
  | Qwen2.5-0.5B | 0.50 |
  | Word swap | 0.0 |
  | Word dropout | 0.8 |

  These are proxies, not ground truth.
- **Suggested controls for experiments:**
  - Use rule views and back-translation at the same K as the "low-knowledge" baselines.
  - Compare training on views only vs original plus views.
  - Measure **teacher agreement**: how often the trained classifier agrees with the generator LLM's zero-shot predictions on held-out data, compared with a no-augmentation model.
  - Check generator **contamination**: whether its pretraining likely included the benchmark (SST-2/MNLI are on the web; MedNLI is not public).
  - Scale the generator (135M → 0.5B → 3B+) at fixed K and filtering.

## Measured CPU throughput (4 cores, no GPU)

Measured on 5 SST-2 sentences, 4 LLM views and 3 rule views each:

| Component | Time per example |
|---|---|
| SmolLM2-135M-Instruct (8 samples × 64 tokens) | about 55 s |
| Qwen2.5-0.5B-Instruct (6 samples) | about 26 s |
| Rule views | effectively free |
| MiniLM similarity + DeBERTa-v3-xsmall NLI scoring | about 0.6 s per view |

CPU is enough for tests and pilots on a few hundred examples. Full datasets with K ≥ 8 LLM views need a GPU. vLLM on a single GPU is the intended path; it also makes sampling seeds per request, so results do not depend on batching.

**Reproducibility caveat:**
- Transformers-backend sampling is seeded per batch, so the outputs depend on `batch_size`, which is recorded in provenance.
- The vLLM backend and the `rule` generator are seeded per example.
