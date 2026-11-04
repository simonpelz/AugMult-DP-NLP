# AugMult-DP-NLP — Continuation Plan, Literature Review & Brainstorm

*Written November 2026, about two years after the last code commit (Nov 2024). This is a planning document only. No code was changed while writing it.*

---

## 0. TL;DR

* **What the project is:** a BSc thesis (TUM). It ports **Augmentation Multiplicity (AugMult)** from DP image classification (De et al., 2022) to **DP-SGD fine-tuning of BERT** on SST-2 and MedNLI. AugMult averages the gradients of K augmented views *per example, before clipping*, so it costs no extra privacy. It is implemented by forking Opacus' hook-based `GradSampleModule` and reshaping activations and backprops to `(N, K, ...)`.
* **What it found:**
  * Gains are small and mostly inside seed noise. On MedNLI at ε=1, back-translation with K=16 gives about +1 pt; EDA hurts; SST-2 shows no effect.
  * There is a weak positive trend in K.
  * The most robust finding is qualitative. **AugMult prevents the "rising validation loss" seen at tiny clipping norms.** This lines up with the literature on how DP-SGD's small clipping norms cause *miscalibration*.
* **Biggest limitation of the existing results:** only the **pooler and classifier** were trained (about 0.5% of parameters, on a frozen BERT-base encoder). The DP gap and the room for AugMult were both measured in a regime close to a linear probe. Modern DP-NLP practice (full fine-tuning or LoRA with ghost clipping) was not explored.
* **Novelty status:** in my searches I found **no published study of AugMult for text or language models**. Every AugMult or DP-augmentation follow-up I found is about images or diffusion. That is still an open niche, but verify it with a targeted citation search before claiming it.
* **Recommended path:**
  1. Revive and make it reproducible (fix two regressions, add tests, add seeds and a test split).
  2. Make it cheap (feature caching; a functional or ghost-clipping AugMult).
  3. Re-ask the question where it can matter (LoRA/full fine-tuning, better and filtered augmentations, compute-matched baselines, calibration).
  4. Then pick one *new* idea to make the paper: per-sample multi-view objectives, patient-level DP via the same code path, or AugMult for DP zeroth-order fine-tuning.

---

## 1. What exists today

### 1.1 Code map

| File | Role |
|---|---|
| `run_experiment.py` | Main entry point. `run(...)` builds the model, data and Opacus `PrivacyEngine`, monkey-patches it to use the AugMult GSM, and trains and evaluates per epoch. Also has a CLI (`commandline()`) and a `debug()` helper. |
| `train.py` | Training loop with `BatchMemoryManager`. It flattens `(N, K, L)` to `(N*K, L)` and uses `repeat_interleave` on the labels. Also eval and metrics. |
| `augmult/augmult_grad_sample_module.py` | Fork of Opacus `GradSampleModule`. The key change is that `max_batch_len = batch // K`. |
| `augmult/augmult_bias_grad_sample_module.py` | Variant for bias-only fine-tuning (BiTFiT-like). Experimental. |
| `augmult/augmented_grad_samplers.py` | Per-sample gradient rules for `nn.Linear` and `nn.LayerNorm` that **sum over the K and sequence axes**. With mean-reduced loss scaled by N, this gives the *average* over the K views. I checked this by reading; there is no test yet. |
| `augmult/data.py` | Dataset loading (GLUE via HF, MedNLI/HoC from local TSV). Multi-view collate builds `[B, K, L]` tensors, on the fly or from precomputed columns. |
| `augmult/augmentations.py` | nlpaug-based EDA family (WordNet synonyms, swap and delete, GloVe/w2v/BERT insert and replace) plus parsing of `precomputed_AxB`. |
| `util/local_datasets.py` | Minimal HF-like TSV datasets. Also contains the **offline precomputation script** for back-translation (Opus-MT, 11 pivot languages) and BlueBERT insert/replace. |
| `util/privacy_engine_util.py` | Monkey-patches: `_prepare_model_modified` (adds the `"augmult"` grad-sample mode), empty-batch collate, and `add_noise`. **The noise is independent of C**: summed grads are divided by C, which is equivalent to rescaling the learning rate by 1/C, as in De et al. and automatic clipping. |
| `util/different_finetune_modes.py` | Which parameters to train (`classifier_and_pooler` was used for the reported results). |
| `init_sweep.py` / `execute_sweep.py` | W&B sweep definition and a SLURM agent with requeue and checkpoint handling. |
| `non_dp.py` | Non-DP baseline using the same pipeline. |
| `scrap.ipynb` | Scratch notebook: prints augmentations of MedNLI samples. **Contains MedNLI text in its outputs** (see §3.4). |
| `figures/` | Result plots used in the README. |

Final hyperparameters, inferred from `init_sweep.py`:
* B = 512, 1,500 steps (about 68 epochs on MedNLI)
* lr = 133.33 under the 1/C parameterization, C = 0.0075
* ε ∈ {1, 8}, δ = 1/n
* max sequence length 128, plain SGD, no momentum, no EMA

### 1.2 Results, as read off the committed figures

**MedNLI (validation accuracy)**

| | ε=1 | ε=8 |
|---|---|---|
| No augmentation | 0.589 | 0.631 |
| EDA (K=5) | 0.581 | 0.615 |
| Back-translation (K=16) | **0.600** | 0.630 |
| Non-DP (same head-only regime) | 0.672 | 0.672 |

*MedNLI ε=1 detail, 5 seeds:*
* Validation: NoAug 0.578±0.007, EDA 0.568±0.012, BT 0.587±0.011.
* Training accuracy on unaugmented samples: NoAug 0.509, EDA 0.495, BT 0.509.
* Training accuracy on augmented samples: EDA 0.472, BT 0.493. This is the "inherent error" of the augmentations.

*K-scaling, BT, 4 seeds:*

| K | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Accuracy | 0.572 | 0.578 | 0.578 | 0.580 | 0.582 | 0.587 |

Standard deviation is about 0.006–0.011. NoAug is 0.578.

**SST-2:**

| | ε=0.1 | ε=8 |
|---|---|---|
| NoAug | 0.830 | 0.851 |
| EDA (K=5) | 0.830 | 0.854 |
| Non-DP | 0.859 | 0.859 |

**Rising loss:**
* Without AugMult, validation loss on MedNLI turns upward after about 250 steps.
* It rises further the smaller C is, from about 1.5 at C=0.5 to about 2.9 at C=1e-4.
* With AugMult, it stays between 1.1 and 1.3 for every C.
* The README notes that accuracy is best where loss is highest. That is the signature of over-confidence, not of worse ranking.

### 1.3 An honest read of the evidence

* Most accuracy differences are about one standard deviation, with 4–5 seeds and no paired test. **Statistically, "BT ≥ NoAug > EDA" on MedNLI at ε=1 is suggestive, not established.**
* All numbers are **validation** accuracy, and the hyperparameters were tuned on the same validation set. The test split is loaded but never evaluated.
* The non-DP ceiling (67%) is about 10–15 pts below what full fine-tuned clinical encoders reach on MedNLI (about 80%+ in the literature). The whole study lives in a constrained, linear-probe-like regime.
* **Hidden confounder:** `model.train()` leaves **dropout active inside the frozen encoder**. Every run, including "No Augmentation", therefore already sees stochastic views of each input. This probably also explains why training accuracy (about 0.51) is *below* validation accuracy (about 0.58): eval runs with dropout off.
* The rising-loss result is the most distinctive and most explainable finding. It deserves to be a central thread rather than a side note (§4.3).

---

## 2. Code audit: bugs and caveats to fix before any new experiment

Ordered by severity.

1. **MedNLI loading is broken at HEAD.** In `augmult/data.py:get_dataset`, the `if dataset_name == "mednli": ...` block is followed by `if dataset_name == "hoc": ... else: raise NotImplementedError`. Loading MedNLI therefore always raises. This was introduced in `a48e364` and needs `elif`.
2. **No seeding anywhere.** There are no torch, numpy, random or Python hash seeds. In `collate_precomputed`, `precomp_augs = set(...)`, so view order depends on `PYTHONHASHSEED`. That is harmless for the averaged gradient but bad for reproducibility and debugging.
3. **The test split is never evaluated.** Reported numbers are validation numbers after tuning on validation.
4. **δ = 1/n is too loose.** Ponomareva et al. (2023, "How to DP-fy ML") and common practice recommend δ ≪ 1/n, e.g. 1/n^1.1 or 1e-5 for MedNLI's n≈11k. Also report ε under PLD/PRV accounting; Opacus has `accountant="prv"`.
5. **Hyperparameter search privacy is not accounted for.** This is standard in papers but should be stated, or handled per Papernot & Steinke (ICLR 2022). TAN-style tuning at small batch size is one way to reduce it (Sander et al., 2023).
6. `run_experiment.py`:
   * `from uu import Error`: the `uu` module was removed in Python 3.13, and it is the wrong exception type anyway.
   * `"..." + e` would raise a `TypeError`.
   * `--glue type=bool` treats any string as True.
7. `train.py`: the timer reset `for times in (...): times = {...}` only rebinds a local, so timing logs accumulate forever.
8. `empty_batch_handling` builds empty-batch shapes from the *raw* dataset row (string columns), not the tokenized keys. An empty Poisson batch would crash. This is practically never hit at B=512.
9. **The precomputed BlueBERT columns are mislabeled.** `bert_insert_zh_*`, `bert_replac_de_*` and `bert_replac_ru_*` are all computed from the *original* sentences, not the zh/de/ru back-translations.
10. **K-scaling runs (K = 2…12) are not reproducible from the code.** `PrecomputedAugsDataset` always loads all 16 columns, and `get_transforms_from_str("precomputed_AxB")` asserts `len == A*B`. The subsets must have been made by local edits.
11. **HoC (multi-label) support is not working.** It is enabled by a `num_labels == 5` heuristic (HoC has 10 classes). The metric uses argmax, and BCE needs float labels.
12. **Only the `Linear` and `LayerNorm` grad samplers are AugMult-aware.** Training embeddings, or any other layer type, would silently need more samplers (and `strict` validation).
13. The environment pins Python 3.9, torch 2.4, Opacus 1.5.2 and transformers 4.41, with absolute cluster paths in `execute_sweep.py`, `local_datasets.py` and `augmentations.py`.

---

## 3. Literature review: what has happened since late 2024, and what was missed

### 3.1 AugMult and augmentation under DP

* **De et al. 2022**, *Unlocking High-Accuracy DP Image Classification through Scale* ([arXiv 2204.13650](https://arxiv.org/abs/2204.13650)). This is the origin of AugMult in DP. It also introduces EMA of parameters, weight standardization, and large batches.
* **Sander, Stock, Sablayrolles 2023**, *TAN Without a Burn: Scaling Laws of DP-SGD* ([arXiv 2210.03403](https://arxiv.org/abs/2210.03403)). Its argument for why AugMult helps is that hard examples stay learnable and **per-example gradient norms stay concentrated** instead of collapsing to 0. Its TAN trick does hyperparameter search at small batch size with constant σ/B, which makes sweeps far cheaper. *The original conda env here was named `tan`, so this paper was clearly on the radar.*
* **Fort et al. 2021**, *Drawing Multiple Augmentation Samples per Image* ([arXiv 2105.13343](https://arxiv.org/abs/2105.13343)), and **Hoffer et al. 2020**, *Augment Your Batch* (CVPR). These are the non-DP roots: multiplicity keeps the augmentation's bias while reducing its variance.
* **Bao et al. 2023**, *DP-Mix* (NeurIPS) ([arXiv 2311.01295](https://arxiv.org/abs/2311.01295)). Self-mixup and diffusion-generated views inside the AugMult framework. It also states that multi-sample augmentations (mixup *across* examples) break per-example sensitivity, so only single-input views are allowed.
* **Dockhorn et al. 2022**, *DP Diffusion Models* ([arXiv 2210.09929](https://arxiv.org/abs/2210.09929)). Applies "noise multiplicity" and AugMult to diffusion training.
* **Wu et al. 2023**, *Augment then Smooth* ([arXiv 2306.08656](https://arxiv.org/abs/2306.08656)). AugMult for DP and certified robustness.
* **Why clipping hurts (theory hooks for an AugMult explanation):**
  * Chen, Wu, Hong 2020, *Understanding Gradient Clipping in Private SGD: A Geometric Perspective* (NeurIPS).
  * Koloskova et al. 2023, *Revisiting Gradient Clipping: Stochastic Bias…* ([arXiv 2305.01588](https://arxiv.org/abs/2305.01588)).
  * Zhang et al. 2023, *DP-SGD without Clipping Bias: An Error-Feedback Approach* ([arXiv 2311.14632](https://arxiv.org/abs/2311.14632)).
  * The natural story is that averaging K views *before* clipping reduces the within-example gradient variance, and with it the clipping bias.
  * I found **no 2024–26 paper that formally analyses AugMult**. A clean theoretical note would be a contribution in itself.
* **Text:** I found **no published AugMult-for-NLP study** (searches as of Nov 2026). This thesis is still, as far as I can tell, the only one.

### 3.2 DP fine-tuning of language models

* **Li, Tramèr, Liang, Hashimoto 2022**, *Large Language Models Can Be Strong DP Learners* (ICLR) ([arXiv 2110.05679](https://arxiv.org/abs/2110.05679)). **Full** fine-tuning with ghost clipping beats head-only and PEFT. It also showed that sensible hyperparameters (large batch, small C) close most of the gap on SST-2, QNLI, MNLI and QQP. *This is the most important baseline the thesis did not use.*
* **Yu et al. 2022**, *DP Fine-tuning of Language Models* (ICLR) ([arXiv 2110.06500](https://arxiv.org/abs/2110.06500)). DP-LoRA and adapters.
* **Bu et al.**:
  * *DP-BiTFiT* (bias-only, [arXiv 2210.00036](https://arxiv.org/abs/2210.00036)). Directly relevant to the half-built `bias_only` mode.
  * *Automatic Clipping* (NeurIPS 2023, [arXiv 2206.07136](https://arxiv.org/abs/2206.07136)). This is the C→0 normalization regime the thesis effectively used.
  * *Book-keeping / DP at small cost* (ICML 2023, [arXiv 2210.00038](https://arxiv.org/abs/2210.00038)).
  * The `fast-differential-privacy` library.
* **DP zeroth-order fine-tuning:**
  * DP-ZO, Tang et al. 2024 ([arXiv 2401.04343](https://arxiv.org/abs/2401.04343)).
  * DPZero, Zhang et al. 2023 ([arXiv 2310.09639](https://arxiv.org/abs/2310.09639)).
  * DP-ZOSO/ZOPO ([arXiv 2402.07818](https://arxiv.org/abs/2402.07818)).
  * Per-example sensitivity here is a *scalar* loss difference, so there is no backprop, and memory equals inference.
* **User-level DP for LLMs:**
  * Chua et al. 2024, *Mind the Privacy Unit!* ([arXiv 2406.14322](https://arxiv.org/abs/2406.14322)).
  * Charles et al. 2024, *Fine-tuning LLMs with User-Level DP* ([arXiv 2407.07737](https://arxiv.org/abs/2407.07737)).
  * The user-wise DP-SGD in Chua et al. **averages each user's gradients, then clips and adds noise**. That is structurally the same as AugMult, with "views" replaced by "that user's records".
* **Scaling laws and DP pre-training:**
  * McKenna et al. 2025, *Scaling Laws for DP Language Models* (ICML) ([arXiv 2501.18914](https://arxiv.org/abs/2501.18914)). The **noise-batch ratio** is the key quantity, and compute-optimal DP models are much smaller.
  * **VaultGemma** (Google, Sep 2025; [arXiv 2510.15001](https://arxiv.org/abs/2510.15001)). A 1B open-weights model DP-trained from scratch, with sequence-level ε≤2.
* **Correlated-noise mechanisms (DP-FTRL / matrix factorization):**
  * Choquette-Choo et al., multi-epoch MF ([arXiv 2211.06530](https://arxiv.org/abs/2211.06530)).
  * BLT mechanisms ([arXiv 2408.08868](https://arxiv.org/abs/2408.08868)).
  * JAX-Privacy 1.0 supports DP-FTRL alongside DP-SGD (check which MF variants it ships). These mechanisms can beat DP-SGD at the same ε, and they combine orthogonally with AugMult.
* **Surveys and benchmarks:**
  * *DP Fine-Tuning of LLMs: A Survey* (ADMA 2025).
  * *Benchmarking Empirical Privacy Protection for Adaptations of LLMs* ([arXiv 2606.09401](https://arxiv.org/abs/2606.09401)). Compares head-only, LoRA and full DP-SGD.
  * PrivLM-Bench (ACL 2024).
  * Ponomareva et al. 2023, *How to DP-fy ML* ([arXiv 2303.00654](https://arxiv.org/abs/2303.00654)). The reporting checklist to follow.
  * Its companion, *How to DP-fy your data* ([arXiv 2512.03238](https://arxiv.org/abs/2512.03238)), covers synthetic data.

### 3.3 Calibration: the probable explanation for "rising loss"

* Bu et al., *On the Convergence and Calibration of Deep Learning with DP* (TMLR 2023) ([arXiv 2106.07830](https://arxiv.org/abs/2106.07830)). Small C makes DP models **over-confident**; clipping, not noise, drives this. A larger C gives about the same accuracy but much better calibration.
* *A Closer Look at the Calibration of DP Learners* ([arXiv 2210.08248](https://arxiv.org/abs/2210.08248)). Per-example clipping is a major cause of miscalibration in DP fine-tuning of language models. Standard fixes help only marginally. It proposes DP temperature scaling and DP Platt scaling.
* **Hypothesis for this project:**
  * AugMult with imperfect, partly label-breaking views acts like **label smoothing** or a confidence penalty inside each clipped per-example gradient.
  * That would explain why loss stays low (calibrated) while accuracy is unchanged.
  * It is testable cheaply (§4.3), and if it holds it is a real contribution: "AugMult as a free calibration fix for DP-NLP".

### 3.4 Privacy hygiene and auditing

* **Tramèr, Kamath, Carlini**, *Considerations for DP Learning with Large-Scale Public Pretraining* (ICML 2024 position paper) ([arXiv 2212.06470](https://arxiv.org/abs/2212.06470)).
  * This bites for clinical text. Encoders pre-trained on **MIMIC** (Bio_ClinicalBERT, BlueBERT-MIMIC variants, some clinical ModernBERTs) overlap with MedNLI's source, so "public" pretraining is not public here.
  * The thesis used PubMed-only BlueBERT for augmentation, which is fine. Keep auditing this for every model.
* **Empirical auditing:**
  * Steinke, Nasr, Jagielski 2023, *Privacy Auditing with One Training Run* (NeurIPS).
  * Steinke et al. 2024, *The Last Iterate Advantage* ([arXiv 2410.06186](https://arxiv.org/abs/2410.06186)).
  * LiRA (Carlini et al. 2022) for membership inference.
  * An interesting AugMult-specific question: **does AugMult change *empirical* leakage at fixed ε?** It might reduce it, since there is less per-view memorization. It might also *increase* the success of augmentation-aware MIAs.
* **Clinical DP-NLP reality check (2024–26):**
  * DP fine-tuning for synthetic clinical notes reduced, but did not eliminate, canary leakage (Findings of EMNLP 2024).
  * DP ICD coding on MIMIC lost a large share of F1 and showed fairness gaps.
  * These are good motivating citations.
* **Data-use agreement:**
  * MedNLI is PhysioNet-credentialed. PhysioNet's DUA restricts sharing with third parties, and PhysioNet has published guidance on using online LLM services with credentialed data.
  * **Generate any LLM paraphrases of MedNLI with locally hosted models**, unless a compliant setup is confirmed.
  * Also check that `scrap.ipynb`'s committed outputs, which contain MedNLI sentences, are acceptable under the DUA. If not, clear its outputs and consider history rewriting. *(I did not change anything; this is for the owner to decide.)*

### 3.5 Alternatives to DP training for private NLP

These are useful as baselines or framing.

* **DP synthetic text:**
  * Yue et al. 2023, ACL ([arXiv 2210.14348](https://arxiv.org/abs/2210.14348)): DP-fine-tune a generator, then train non-privately on its samples.
  * **Aug-PE**, Xie et al. 2024, ICML ([arXiv 2403.01749](https://arxiv.org/abs/2403.01749)): Private Evolution with API-only LLMs. Benchmarks are Yelp, OpenReview and PubMed.
* **DP in-context learning / private prediction:**
  * DP-ICL, Wu et al. 2023 ([arXiv 2305.01639](https://arxiv.org/abs/2305.01639)).
  * DP few-shot generation, Tang et al. ICLR 2024.
  * Synthetic DP demonstrations ([arXiv 2509.13625](https://arxiv.org/abs/2509.13625)).
* **Local DP text rewriting**, with a strong TUM line of work:
  * Meisenbacher & Matthes et al.: DP-MLM ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)), *Just Rewrite It Again*, the impact of noise, dataset-size effects (IJCNLP-AACL 2025).
  * Igamberdiev & Habernal, DP-BART.
  * Habernal 2021, *When DP meets NLP: the devil is in the detail*, a must-read on getting DP claims right in NLP.
  * Klymenko et al. 2022, *DP in NLP: The Story So Far*.
  * *Possible link:* DP-rewritten text could serve as *augmentation views* (although views must not be published, so they are only needed for per-example computation).

### 3.6 Text augmentation since EDA

* EDA (Wei & Zou 2019) is now considered weak. AEDA (punctuation insertion, Karimi et al. 2021) is cheaper and more label-preserving.
* **Embedding and model-space views** cost almost nothing with AugMult:
  * Cutoff (Shen et al. 2020: token, span or feature cutoff).
  * Dropout as augmentation (R-Drop, Liang et al. 2021; SimCSE, Gao et al. 2021).
  * Gaussian or adversarial noise on embeddings.
* **Back-translation with modern MT:** NLLB-200 and MADLAD-400 are much better than the 2020-era Opus-MT, especially on clinical jargon. Diverse decoding (sampling, temperature) gives more views per pivot.
* **LLM paraphrasing:**
  * Cegin et al., NAACL 2025, *LLMs vs Established Text Augmentation Techniques for Classification*: gains over classic methods are inconsistent and need to be weighed against cost.
  * An ACL 2025 paraphraser fine-tuned for diversity while preserving labels.
  * Ding et al. 2024 survey ([arXiv 2403.02990](https://arxiv.org/abs/2403.02990)).
  * **Label-preservation filtering** (with an NLI or consistency model, or the task model itself) is the standard fix for the "inherent error" the thesis measured.
* **Domain-specific views** for clinical text: UMLS/SNOMED synonym substitution, abbreviation expansion and contraction ("pt" ↔ "patient", "SOB" ↔ "shortness of breath"), and unit and number formatting.

### 3.7 Models, datasets, tools (2026 state)

**Encoders**
* RoBERTa and DeBERTa-v3 (stronger than BERT-base for NLI).
* **ModernBERT** (Warner et al. 2024): faster, 8k context.
* **BioClinical ModernBERT** ([arXiv 2506.10896](https://arxiv.org/abs/2506.10896)) and Clinical ModernBERT ([arXiv 2504.03964](https://arxiv.org/abs/2504.03964)).
* BiomedBERT/PubMedBERT, Bio_ClinicalBERT (MIMIC-trained, so the overlap caveat applies), GatorTron, NeoBERT, EuroBERT.

**Decoders (DP-LoRA)**
* Small Qwen, Llama, Gemma and SmolLM-class models.
* **VaultGemma 1B** as a DP-pretrained starting point. This makes a cleaner "nothing private in pretraining" story, which pairs well with the Tramèr et al. critique.

**Datasets**
* Keep SST-2 (as the "easy" control) and MedNLI.
* Add the standard DP-NLP set (SST-2/QNLI/MNLI/QQP), which makes results comparable to Li et al. 2022.
* Clinical and biomedical: BLURB tasks (incl. HoC, ChemProt), n2c2 challenges, MIMIC-IV-Note-derived tasks (credentialed).
* Privacy-flavoured: TAB/ECHR (anonymization), Enron, Yelp/PubMed/OpenReview (DP synthetic text benchmarks), PrivLM-Bench.
* **Low-resource subsets** (e.g. 1k/2k/5k examples). The thesis's own conclusion says AugMult should help most when data are scarce, but this was never tested directly.

**Libraries**
* **Opacus 1.6.0** (May 2026, after v1.5.4 in May 2025):
  * FSDP and tensor parallelism for fast gradient clipping.
  * Mixed precision.
  * `wrap_model=False` for HF Transformers/Accelerate compatibility.
  * Custom accountants.
  * Requires torch ≥ 2.6.
  * v1.5.3/1.5.4 added ghost clipping for embeddings, adaptive clipping, per-sample grads under ghost clipping, a LoRA/PEFT tutorial, and an RMSNorm sampler.
* **JAX-Privacy 1.0** (Nov 2025; DP-SGD and DP-FTRL/BLT, Keras API, auditing; it powered VaultGemma). Google `dp_accounting`.
* `fast-differential-privacy` (book-keeping, BiTFiT), `private-transformers`, `dp-transformers`.
* HF `peft`, `datasets` (Parquet), `vllm` (fast local paraphrase generation), `sentence-transformers` / NLI cross-encoders for filtering.
* Infra: `uv` for environments, Hydra/OmegaConf for configs, W&B (still fine), pytest.

---

## 4. Brainstorm: research directions

Ordered roughly from "fixing the original question" to "new contributions". Each idea notes *why* it matters and *how cheap* it is.

### 4.1 Re-ask the original question properly

1. **Fine-tuning regime sweep:** head-only → BiTFiT → LoRA (rank 8/16) → full fine-tuning (ghost clipping), each ± AugMult. *This is the single most important missing experiment.* De et al.'s gains came from training the whole network.
2. **Compute-matched comparisons.** AugMult with K views costs about K× compute. Compare it against using the same compute for (a) a K× larger batch at the same ε, (b) more steps, or (c) a bigger model. The thesis's `honest_k_scaling.png` hints at this but stops short. Use the noise-batch-ratio lens of McKenna et al. 2025.
3. **Data-scarcity curve:** subsample MedNLI and SST-2 to 500/1k/2k/5k/full and plot the AugMult gain vs n. This tests the thesis's own "best use case" claim.
4. **Stronger statistics:** ≥10 seeds, paired across conditions (same seed means same Poisson batches), bootstrap CIs, and test-set reporting.
5. **Modern encoders:** RoBERTa-base/large, DeBERTa-v3, ModernBERT, BioClinical ModernBERT (checking MIMIC overlap).

### 4.2 Better views ("quality > quantity" was the thesis's main lesson)

6. **LLM paraphrases**, generated locally with vLLM:
   * Use a clinical-aware instruction model to produce K paraphrases per premise/hypothesis at controlled edit distance.
   * Precompute and version them as a Parquet dataset.
   * Sweep K up to 32–64, since this becomes cheap once views are cached.
7. **Label-preservation filtering and weighting.** Score each view with a *public* NLI model or a consistency check and drop or down-weight label-breaking views. This attacks the measured "inherent error" directly. Privacy note: any per-example computation that uses only that example and public models is free under DP, because it happens before clipping.
8. **Modern back-translation** (NLLB/MADLAD with sampling) vs Opus-MT. Also domain views: UMLS synonym swaps and abbreviation toggles.
9. **Free "model-space" views:**
   * Dropout masks (K forward passes, different masks).
   * Token/span cutoff.
   * Embedding Gaussian noise.
   * These cost no augmentation pipeline at all. Also *control for* the encoder-dropout confounder (§1.3).
10. **Learned or adaptive view selection:**
    * "Hard views" (MaxUp-style max over views, or top-k by loss), still per-example.
    * A curriculum over view strength.
    * Mix strong and weak views.

### 4.3 Understand the mechanism: calibration and gradient geometry

11. **Is AugMult's effect on rising loss a calibration effect?**
    * Log ECE, NLL, Brier score and confidence histograms.
    * Compare AugMult vs label smoothing vs a confidence penalty vs DP temperature scaling at matched ε across C ∈ {1e-4 … 1}.
    * Prediction: label smoothing reproduces most of the "loss stabilization".
12. **Gradient-norm diagnostics:**
    * Track the distribution of per-example gradient norms (fraction clipped, cosine to the batch mean, norm concentration), as in Sander et al. 2023, for K = 1…16.
    * Do the same analysis for text that TAN did for images.
13. **Theory note:**
    * Bound the clipping bias as a function of within-example view variance, K, and C (building on Chen et al. 2020 and Koloskova et al. 2023).
    * Show when AugMult helps or hurts with label-noisy views. This explains *why EDA hurts*: its views shift the per-example mean (bias) more than they reduce variance.

### 4.4 Generalize AugMult (new methods)

14. **Per-example multi-view objectives.** AugMult averages the loss over views, but *any* per-example function of the K views is equally private:
    * Consistency regularization (R-Drop/UDA-style KL between views).
    * Within-example contrastive terms.
    * Prediction averaging before the loss (averaging logits over views, i.e. a "test-time-augmentation-style" training loss).
    * This opens a whole method family with zero extra privacy cost.
15. **Ghost-clipping AugMult:** for linear layers, the per-example gradient norm can be computed without materializing it. **Concatenating the K views along the sequence axis** gives exactly the norm of the view-summed gradient (the Gram trick over `(K·T)` tokens). AugMult then works with Opacus/fastDP ghost clipping and book-keeping, which makes full fine-tuning feasible. *This is a neat engineering and paper contribution.*
16. **AugMult for DP zeroth-order fine-tuning** (DP-ZO/DPZero):
    * The per-example quantity is a scalar loss difference, so averaging over K views is trivial.
    * Memory equals inference.
    * Views reduce the variance of the scalar before clipping.
    * Very cheap to try on top of an existing DP-ZO implementation, and it scales to billion-parameter decoders.
17. **Patient-level / user-level DP via the same code path.**
    * MIMIC-derived data have many records per patient, so example-level DP under-protects patients.
    * The AugMult reshape `(N, K, …)` *is* user-wise DP-SGD (Chua et al. 2024) when "views" are a patient's records.
    * A combined "user-level DP with within-record AugMult" pipeline (two-level averaging) is novel and well-motivated clinically.
    * MedNLI does not expose patient IDs, so this needs a MIMIC-IV-Note-derived task, or a public proxy (e.g. reviews grouped by author).
18. **Correlated noise (DP-MF/BLT) + AugMult.** These are orthogonal. Check whether the gains stack, especially for many-epoch training on small clinical sets.
19. **EMA / weight averaging** (De et al.'s other trick). It is cheap and was never tried here.

### 4.5 Privacy evaluation beyond ε

20. **Empirical auditing:** one-run canary audits and LiRA on models trained ± AugMult at equal ε. Is empirical leakage lower? Do augmentation-aware MIAs, which query with augmented inputs, do better?
21. **Public-pretraining hygiene study:** MIMIC-pretrained vs PubMed-only vs VaultGemma backbones on MedNLI. How much of "DP utility" is actually leakage of the private distribution into pretraining?

### 4.6 Put it in the 2026 landscape (bigger reframing)

22. **DP-LoRA on small decoders** (Qwen/Gemma/VaultGemma) for MedNLI as classification or generation, ± AugMult with LLM views.
23. **Head-to-head pipelines at fixed ε:** DP fine-tuning + AugMult vs DP synthetic data (Aug-PE / DP-finetuned generator) + non-private training vs DP-ICL.
    * Which wins on small clinical data?
    * Augmentations could also *help* the synthetic-data route; Aug-PE already uses LLM variation.
24. **Fairness under DP + AugMult:** subgroup accuracy (DP is known to hurt minority subgroups disproportionately). Do label-preserving views help or hurt them?

---

## 5. Proposed roadmap

### Phase 0: revive & reproduce (about 1–2 weeks)

- [ ] Decide on a target (paper venue, portfolio, or follow-up thesis) and the compute budget (see §6).
- [ ] Recover W&B run data (projects `mednli`, `sst2`, `hoc` on the old account) and export summary CSVs into `results/`, so the figures become regenerable.
- [ ] New environment: `uv` + `pyproject.toml`, Python 3.11/3.12, torch ≥ 2.6, Opacus 1.6, a current transformers/datasets. Remove absolute paths and move to env/config.
- [ ] Fix §2 items 1, 2, 3, 6, 7 and 9 (the regressions and hygiene fixes). Keep the old behaviour behind flags where it affects comparability.
- [ ] **Tests:**
  1. The AugMult grad sampler equals `torch.func` per-example grads averaged over K, on a tiny BERT.
  2. With K=1 it equals vanilla Opacus.
  3. The collate shapes are correct.
  4. The privacy accounting matches Opacus for the same q, σ and steps.
- [ ] Reproduce two anchor numbers (MedNLI ε=1 NoAug and BT-16), now with test-set reporting and δ=1/n^1.1. Write down any drift.

### Phase 1: make experiments cheap (about 1–2 weeks)

- [ ] **Feature caching** for the head-only regime: run the frozen encoder once per view and cache the [CLS] states. Training then becomes a tiny MLP, with orders of magnitude more seeds and K.
  * Handle the dropout confounder explicitly: cache with dropout off, and add "dropout views" as a separate condition.
- [ ] **A clean AugMult implementation** that does not fork Opacus. Two options:
  * (a) `torch.func.vmap(grad(per_example_loss_over_K_views))`, which is simplest and generic;
  * (b) ghost-clipping AugMult via sequence-axis concatenation (§4.4 idea 15) for full fine-tuning and LoRA.
  * Keep the old GSM for regression comparison.
- [ ] Precomputed view store: Parquet with `(example_id, view_id, method, text_a, text_b, filter_score)`, generated by local models.
- [ ] Hydra configs, deterministic seeds, and a small sweep launcher (SLURM or local).

### Phase 2: core science (about 3–6 weeks)

- [ ] §4.1 ideas 1–4: regime × AugMult × ε ∈ {1, 3, 8} × n-subsets, compute-matched, ≥10 seeds, on MedNLI + SST-2 (+ QNLI/MNLI for comparability).
- [ ] §4.2 ideas 6–9: view quality (LLM paraphrase ± filter, NLLB BT, dropout/cutoff views).
- [ ] §4.3 ideas 11–12: calibration and gradient-geometry diagnostics. **Likely the headline of a short paper.**

### Phase 3: pick one novel thread (about 4–8 weeks)

- [ ] Choose one or two of: multi-view per-example objectives (idea 14), ghost-clipping AugMult (15), DP-ZO + AugMult (16), patient-level DP (17), auditing (20).

### Phase 4: write-up

- [ ] Possible venues: PrivateNLP workshop (ACL/NAACL), TPDP, ClinicalNLP/ML4H, SaTML, PETS, Findings of ACL/EMNLP.
- [ ] Follow the *How to DP-fy ML* reporting checklist: privacy unit, ε/δ, accountant, hyperparameter-tuning disclosure, and public-data provenance.

### Concrete "first five experiments" once Phase 0 is done

1. MedNLI, head-only, cached features: NoAug vs BT-16 vs dropout-16, 20 seeds, ε=1. Does the +1 pt survive, and is dropout alone as good?
2. The same with ECE/NLL logged, across C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} ± label smoothing. Is rising loss a calibration effect?
3. MedNLI LoRA (r=8) on RoBERTa-base with Opacus ghost clipping, ± BT-16 at ε=1/8. Does AugMult matter more when more parameters train?
4. Data-scarcity curve on SST-2 (n = 500 … 67k), ± AugMult. Where, if anywhere, is the crossover?
5. Compute-matched: BT-16 at B=512 vs NoAug at B=8192 (same forward passes), same ε.

---

## 6. Open questions for the project owner

1. **Goal:** a publishable paper, a portfolio refresh, or a handover to a new student? This decides how much of Phases 3–4 matters.
2. **Compute:** is the old SLURM cluster still accessible? Otherwise, what GPU budget is there (one consumer GPU is enough for the head-only regime with caching and for LoRA on base-size models)?
3. **Data access:** is the PhysioNet credential (MedNLI, MIMIC) still valid? Are the precomputed `train_precomputed_16.tsv` and the W&B logs still available?
4. **Supervision and collaboration:** stay aligned with the original TUM group, or contact the TUM DP-text-rewriting group (Matthes/Meisenbacher), whose work is adjacent (§3.5)?
5. **Scope:** keep the classification framing (encoders), or move to decoder LLMs with DP-LoRA, where the field's attention now is?
6. **`scrap.ipynb` outputs** contain MedNLI text. Is that acceptable under the DUA, or should it be cleaned (possibly with history rewriting)?
