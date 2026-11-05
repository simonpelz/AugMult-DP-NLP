# CLAUDE.md — orientation for coding agents

Research code for a TUM BSc thesis: **Augmentation Multiplicity (AugMult) for DP-SGD fine-tuning of BERT** (SST-2, MedNLI).
AugMult is K augmented views per example, with per-example gradients **averaged over views before clipping**, so it has no extra privacy cost.
Last code change: Nov 2024. The continuation plan, code audit, literature review and roadmap are in **`docs/CONTINUATION.md`**. Read it first.

## Layout
- `run_experiment.py` — `run(...)` entry point: model, data, Opacus `PrivacyEngine` (monkey-patched), and the train/eval loop. CLI via `python run_experiment.py --help`.
- `train.py` — training loop. Batches are `[N, K, L]`, flattened to `[N*K, L]`, with labels `repeat_interleave`d K times.
- `augmult/augmult_grad_sample_module.py` — fork of Opacus `GradSampleModule` (`max_batch_len = batch // K`).
- `augmult/augmented_grad_samplers.py` — K-aware per-sample grads for `nn.Linear` and `nn.LayerNorm` only.
- `util/privacy_engine_util.py` — adds the `"augmult"` grad_sample_mode, empty-batch collate, and `add_noise`, which divides by C so the noise is independent of C (equivalent to lr/C).
- `augmult/data.py`, `augmult/augmentations.py`, `util/local_datasets.py` — data, nlpaug augmentations, and the offline precomputation of back-translations.
- `init_sweep.py` / `execute_sweep.py` — W&B sweep and SLURM agent. `non_dp.py` — non-private baseline.

## Running
- Modern env: `pyproject.toml` (Python 3.11/3.12, torch>=2.6, opacus>=1.6, transformers 5). `pip install -e ".[dev]"` (+ `.[aug]` for nlpaug/EDA). Tests: `pytest tests -q` (CPU, offline tiny models). The legacy `environment.yml` is kept for thesis reproduction only.
- Smoke run (CPU, public data): `WANDB_MODE=disabled python run_experiment.py --dataset_name sst2 --model_name google/bert_uncased_L-2_H-128_A-2 --dataset_size 128 --epochs 1 --batch_size 32 --target_epsilon 8 --transforms_name identity2 --seed 0`. `prajjwal1/bert-tiny` fails to load its tokenizer on transformers 5 (it has only a vocab.txt).
- Any public HF dataset: `--hf_path/--hf_config/--text_columns/--label_column/--splits`. Privacy knobs: `--delta {1/n,n^-1.1,<float>}`, `--accountant {rdp,prv}`, `--seed`. Precomputed views: `precomputed:<view,...>[*B]`.
- Env vars: `DATASET_DIR` (local TSVs), `MODEL_DIR` (word vectors), `CKPT_PATH`, `WANDB_DIR`, `JOBS_DIR`, `MAX_PHYS_BATCHSIZE`.
- Data is **not in the repo**, and the owner has **no PhysioNet access**: use public datasets (see docs/research/datasets_models_compute.md). **Never commit dataset files or outputs containing dataset text** (clear notebook outputs before committing), and never send private data to hosted third-party APIs. Generate LLM views with local models only.

## Status / known issues
- The AugMult grad samplers are verified against `torch.func` per-example grads (tests/test_grad_samplers.py). The audit items in docs/CONTINUATION.md §2 are fixed (see the commit log), except the history issue below.
- `scrap.ipynb` contains restricted MedNLI text in its outputs, including in git history. The cleanup is pending the owner's decision. Do not copy its contents anywhere.
- The Opacus `_prepare_model` is still monkey-patched (Opacus 1.6 has no public custom grad-sample-mode hook); `wrap_model=False` is unsupported for "augmult".
- Research direction and reports: docs/research/*.md.

## Conventions for changes
- Anything touching clipping, noise, sampling or accounting is privacy-critical. Add or extend a test that compares against vanilla Opacus or `torch.func` per-sample gradients, and state the privacy unit, ε, δ and accountant in the PR.
- Per-example computations (views, filters, weights) may use only that example plus public models. Never mix examples before clipping.
- Keep old behaviour reachable behind a flag when it affects comparability with the thesis results.
