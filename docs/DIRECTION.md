# Direction: what we do next, and why

*Decided 2026-11-05 from the eight workstreams in `docs/research/` and the merged code on this branch. This document replaces the open brainstorm in `docs/CONTINUATION.md` §4–5 as the plan of record. That file stays as background.*

## 1. The decision

**Primary paper:** *"Variance or distillation? Augmentation multiplicity for differentially private text classification."*

**Fast first result (paper or workshop):** *"AugMult fixes the miscalibration of small-clip DP-SGD."*

**Side probe, kept only if the effect beats seed noise:** K-view averaging for DP zeroth-order fine-tuning of a small decoder.

### Why this direction
1. **Open niche.** We screened 334 papers citing De et al. 2022 and TAN, and every augmentation-related one is about images or diffusion. Nobody has studied AugMult for text (`docs/research/novelty_impact.md`). Searches found no counter-example, but that is not proof.
2. **It turns the distillation worry into the research question.** LLM-generated views mean the student partly learns the generator's distribution (`docs/research/distillation_llm_augmentation.md`). Privacy is unaffected as long as the generator is public and fixed and sees only its own example. The *conclusions*, however, are at risk. We measure that split instead of ignoring it:
   * (i) variance reduction before clipping;
   * (ii) knowledge transfer from the public model;
   * (iii) the label-smoothing and calibration effect of noisy views.
3. **It builds on the thesis's strongest finding**, the rising loss at small C, and reuses the thesis's measurements:
   * the "inherent error" becomes the view label-flip rate;
   * the K-scaling plot becomes the K ablation;
   * MedNLI becomes the motivating pilot.
4. **It runs on modest compute.**
   * The calibration study and the head-only regime run on a CPU laptop with cached features.
   * The core study needs one 12–24 GB GPU, or roughly $75–450 of rented consumer GPUs. These are agent estimates (`docs/research/datasets_models_compute.md`).
5. **Public data only.** No PhysioNet access is needed.

### Hypotheses (falsifiable)
All comparisons are at matched ε, δ = 1/n^1.1 and compute (forward passes), with at least 10 paired seeds.

* **H1:** teacher-free views (dropout masks, token cutoff) beat the compute-matched K=1 baseline by at least 0.5 pt only in data-scarce settings (n ≤ 5k) or when LoRA-scale or larger parameter sets train.
* **H2:** LLM-paraphrase views beat teacher-free views. When the same public model is also added as an explicit per-example teacher (KL on the same views), the extra gain shrinks to at most a third. That would mean the LLM-view gain is mostly distillation.
* **H3:** at small C, AugMult lowers ECE and NLL about as much as label smoothing tuned to the same accuracy. That would mean "rising loss" is a calibration effect.

### Kill and pivot criteria
* **No view source beats compute-matched K=1 by more than 0.5 pt anywhere with n ≤ 5k:** write the negative result plus the calibration paper.
* **The teacher's zero-shot accuracy beats every DP student:** reframe as "when is DP fine-tuning worth it versus a public model", or move to tasks where a public LLM is weak.

## 2. Mandatory controls (from the distillation review)
* **Public-only baselines:** the teacher zero-shot and few-shot, and a student trained only on generated or public data. Always report **Δ_priv**: the DP student minus the best public-only baseline.
* **Generator ladder:** none → dropout/cutoff → back-translation (NLLB) → small LM → strong instruction-tuned LLM. Record each generator's own task accuracy.
* **Generator and student come from different model families.** Shared initialization transfers traits.
* **Equal-compute non-LLM views and a K×-batch K=1 baseline**, both at the same ε.
* **View diagnostics using public models only:** label-flip rate, similarity, diversity.
* **Contamination checks** with fully open-data models (OLMo, Pythia, Ettin), since SST-2, MNLI and QNLI sit in common instruction-tuning mixes.
* **Label smoothing, temperature scaling and larger-C calibration baselines.**

## 3. Setup

| Role | Choice | Notes |
|---|---|---|
| Comparability tasks | SST-2, QNLI (MNLI) | Same as Li et al. 2022 / Yu et al. 2022 |
| Domain / "LLM-weak" task | HoC or ChemProt (BLURB); NLI4CT as a clinical-NLI stand-in for MedNLI | **Check licences first.** NLI4CT has no declared licence. |
| Data-scarcity axis | n ∈ {500, 1k, 5k, full} subsamples | |
| Student, encoder | RoBERTa-base (comparability), Ettin/ModernBERT-base (open data, modern) | **Avoid** BioClinical/Clinical ModernBERT: pretrained on MIMIC |
| Student, decoder | Qwen3-0.6B or Gemma-3-1B with DP-LoRA. VaultGemma-1B as the "DP-pretrained" variant. | "Jav" is read as Gemma; no small model family by that name exists |
| View generator | A different-family local instruct model of about 4–8B via vLLM, plus NLLB back-translation | Local only; never a hosted API |
| Label filter / teacher | Public DeBERTa-v3 NLI cross-encoder | Disclose that it was trained on MNLI when used on MNLI |
| Privacy | ε ∈ {1, 8}, δ = 1/n^1.1, PRV accountant, Poisson sampling | Disclose that hyperparameter tuning was not privacy-accounted, or use TAN-style tuning |

## 4. What is already built on this branch
* **`dpaug/`, the new experiment engine** (34 tests):
  * AugMult per-example gradients via `torch.func`, for any HF `*ForSequenceClassification` model (BERT/Llama/Qwen tested) in full, head, BitFit or LoRA mode.
  * Pluggable per-example losses: mean over views (thesis), averaged logits, consistency, and distillation from teacher logits.
  * Clipping and noise (including the thesis's noise-independent-of-C variant), Opacus RDP/PRV accounting, seeded Poisson sampling.
  * A cached-feature head trainer, about 112× faster per step than end to end.
  * A reference implementation of the ghost-norm identity.
* **`views/`, offline view generation** (30 tests): identity, rule (AEDA/dropout/swap), LLM paraphrase (transformers or vLLM) and back-translation generators. Each view row records its generator. Scoring covers similarity, two-way NLI and label-flip estimates; filtering pads to exactly K. Output is a Parquet store plus a manifest.
* **Legacy pipeline** (`run_experiment.py`, etc.): kept for reproducing the thesis results.
  * Its AugMult grad samplers are verified against `torch.func` (19 tests). **The thesis gradients were correct.**
  * All audit bugs are fixed, and it runs on Opacus 1.6 and transformers 5 with seeding and test-split evaluation.
  * It can now load any public HF dataset.
* **CI workflow** (`.github/workflows/tests.yml`). It has not run on GitHub yet.

## 5. Next engineering milestone (before the first real experiment)
Roughly 1–2 weeks of work. Every item is concrete:
1. **`experiments/` runner**: a Hydra or plain-YAML config that loads a task, loads a view store from `views/`, builds a `dpaug` model and trains. It logs to W&B or CSV and writes one results row per run (seed, ε, δ, accountant, K, view source, regime, n, test metrics). This is the bridge between `views/` and `dpaug/`.
2. **Metrics**: ECE (15-bin), NLL, Brier score, plus per-example gradient-norm statistics (fraction clipped, norm concentration).
3. **Baselines in `dpaug.losses`**: label smoothing; a compute-matched K×-batch K=1 arm; and DP temperature scaling computed afterwards on a held-out split.
4. **Teacher logits**: precompute a public teacher's logits per view into the view store, so the distillation-loss arm needs no teacher at training time.
5. **The `dropout-K` and `cutoff-K` view sources** inside `dpaug` (model-space views, no store needed).
6. **Reproduce the thesis pattern on public data**: head-only, cached features, SST-2/QNLI, rising loss across C. This is the gate before spending GPU money.

## 6. Timeline
Venue dates are estimates except where confirmed; see `docs/research/novelty_impact.md` §4.

| When | What |
|---|---|
| Nov 2026 | §5 milestone; reproduce rising loss on public data (CPU/laptop). |
| Dec 2026 | Calibration study (H3) → fast-first paper draft. Possible targets: ACL 2027 short paper (ARR deadline 2027-01-04, tight), or TPDP/PrivateNLP 2027 (not yet announced). |
| Jan–Apr 2027 | Primary study (H1/H2) with LLM views generated once on a rented A100/H100 and DP training on one GPU. DP-ZO side probe. |
| ~May 2027 | EMNLP 2027 submission (ARR, estimated date). |

## 7. Decisions needed from the owner
1. **Restricted data in a public repo.** `scrap.ipynb` contains MedNLI text in its outputs, in history from `f72434b` on both branches.
   * The full audit is held outside the repo, so the leak isn't advertised further.
   * Options, in order: make the repo private; rewrite history with `git filter-repo` and force-push; ask GitHub Support to purge its caches; decide whether to notify PhysioNet.
   * No secrets or keys were found.
2. **Compute.** What is the local GPU (model/VRAM, or Apple Silicon)? What cloud budget is acceptable?
3. **Domain task.** Choose HoC/ChemProt, NLI4CT, or a legal task, after checking licences.
4. **Scope.** Primary plus fast-first, as above, or the fast-first calibration paper alone?
